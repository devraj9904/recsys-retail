# Phase 3: streaming features and live user vectors

Kafka (KRaft, single broker) -> consumer -> Redis online store + two-tower
user tower. Freshness is per event: producer timestamp to the moment the
visitor's re-encoded vector is readable in Redis. One consumer process,
CPU only.

## Freshness and throughput

| replay rate | events | consumer events/s | p50 ms | p95 ms | p99 ms | batch p50 ms | mean batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| max (capacity) | 60,000 | 5,709 | 4571 | 7429 | 7716 | 147.5 | 1000 |
| 2,000/s | 60,000 | 2,000 | 80 | 121 | 132 | 33.7 | 169 |
| 500/s | 60,000 | 500 | 46 | 74 | 80 | 10.6 | 31 |

## Does fresh state help? Next-event prediction

| user state | hit@20 | mrr@20 | hit@20 new item | mrr@20 new item |
|---|---:|---:|---:|---:|
| popularity | 0.0082 | 0.0024 | 0.0071 | 0.0015 |
| stale vector (batch, frozen at cutoff) | 0.1164 | 0.0482 | 0.0527 | 0.0107 |
| live vector (streaming) | 0.2161 | 0.0689 | 0.1246 | 0.0266 |

23,591 test events from 11,376 visitors with pre-test history (19,791 on items not in the visitor's last 30 steps).

## Example: one visitor before and after the stream

Visitor 689613, first stream events: view 206673, view 285192, view 350525, view 92341, view 283100, view 62599, view 134647, view 89639, view 89639, view 239145

| # | before (batch state) | after (live state) |
|---:|---|---|
| 1 | 52631 | 253024 |
| 2 | 71336 | 221317 |
| 3 | 301656 | 62300 |
| 4 | 389436 | 465522 |
| 5 | 264756 | 155943 |
| 6 | 351841 | 418317 |
| 7 | 188977 | 302098 |
| 8 | 306448 | 378593 |
| 9 | 300925 | 368403 |
| 10 | 228793 | 448494 |

## Notes

- Each replay run is the first N events of the 14-day test window, sent to a fresh
  3-partition topic; Redis is flushed and warm-started with the replayed visitors'
  last 30 pre-test events first (the batch to online hand-off).
- The `max` run measures capacity: the producer outruns the single consumer, so
  its freshness is queueing delay, not processing time. Per-batch cost (Redis
  pipeline + Lua session/affinity script + user-tower forward pass) is in the
  batch columns.
- For scale: RetailRocket's 2.7M events over 4.5 months average about 0.25
  events/s, so the paced runs are thousands of times the dataset's real traffic.
- The prediction table is offline and deterministic (no Kafka timing involved):
  every test event of a visitor with pre-test history, predicted from the
  vector frozen at the cutoff vs the vector after all earlier events. Events on
  items outside the trained catalogue are dropped from history and targets
  (Phase 4 handles them), so these hit rates are per warm next event and not
  comparable to Phase 1 recall, which counts cold targets as misses.
- Single machine (container CPU), single broker and consumer, no replication;
  numbers will differ on other hardware.
