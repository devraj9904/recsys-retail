# Phase 5: serving latency and throughput

API: 4 uvicorn workers, HNSW index, 4 vCPUs shared with the load generator, Redis on the same machine. Mix per client loop: 70% recommendations for returning visitors, 10% for unknown visitors, 20% click events.

## Latency vs load (cache bypassed: every request runs the full pipeline)

| clients | requests/s | p50 ms | p95 ms | p99 ms | returning-visitor recs p95 ms | event p95 ms | errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 234 | 4.2 | 5.6 | 8.1 | 5.7 | 5.1 | 0 |
| 4 | 523 | 7.4 | 10.8 | 14.2 | 10.6 | 11.9 | 0 |
| 8 | 685 | 10.0 | 23.7 | 31.5 | 22.3 | 29.1 | 0 |
| 16 | 722 | 16.3 | 57.2 | 93.6 | 56.5 | 59.7 | 0 |

## With the Redis response cache

8 clients: 875 requests/s, p50 7.9 ms, p95 18.8 ms, p99 26.1 ms, cache hit rate 57%. Hits skip the pipeline; every click event drops that visitor's cached response.

## Where the time goes (server-side mean, one client)

| stage | ms |
|---|---:|
| retrieval: Redis reads + HNSW top-100 + candidate merge | 0.78 |
| 24 ranking features | 0.48 |
| LambdaRank + conversion model | 0.46 |
| MMR over top 50 | 0.21 |
| ranked route, total | 2.35 |
| cold-start content kNN (TF-IDF over 233k items) | 9.34 |

## Notes

- Latency is client-side and includes HTTP, JSON and queueing. The gap between the
  ~2-3 ms pipeline and the client p50 at higher load is queueing for CPU: the
  load generator runs on the same 4 vCPUs as the API.
- Throughput flattens past 8 clients because the machine is CPU-bound; more
  workers or machines scale it horizontally (the API keeps no state in process).
- Single-node Redis without persistence; no network hop between services.
