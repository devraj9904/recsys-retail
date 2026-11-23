"""Replay prepared events into Kafka as a live click stream.

    python -m recsys.streaming.producer --split test --rate 2000 --limit 100000

Events go out in event-time order, keyed by visitor so one visitor's events
stay ordered within a partition. ``--rate`` caps events per second (0 = as
fast as the broker accepts); ``--speedup`` instead replays with the original
gaps compressed by that factor. Each message carries ``sent_at`` so the
consumer can measure end-to-end freshness.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator

import pandas as pd

from recsys.config import processed_dir
from recsys.streaming.events import TOPIC, Event


def iter_events(df: pd.DataFrame) -> Iterator[Event]:
    df = df.sort_values("ts", kind="stable")
    # Parquet round-trips can give datetime64[ms]; normalise the unit before
    # converting, or the epoch seconds come out 1000x too small.
    ts = df["ts"].astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
    for t, v, i, e in zip(
        ts, df["visitor_id"].to_numpy(), df["item_id"].to_numpy(), df["event"].astype(str),
        strict=True,
    ):  # fmt: skip
        yield Event(visitor_id=int(v), item_id=int(i), event=e, ts=float(t))


def make_producer(bootstrap: str):
    from confluent_kafka import Producer

    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "linger.ms": 5,
            "batch.num.messages": 10000,
            "compression.type": "lz4",
            "enable.idempotence": True,
        }
    )


def replay(
    events: Iterator[Event],
    producer,
    topic: str = TOPIC,
    rate: float = 0.0,
    speedup: float = 0.0,
    limit: int | None = None,
) -> dict[str, float]:
    sent, t0, first_ts = 0, time.perf_counter(), None
    for e in events:
        if limit is not None and sent >= limit:
            break
        if rate > 0:
            wait = t0 + sent / rate - time.perf_counter()
        elif speedup > 0:
            first_ts = e.ts if first_ts is None else first_ts
            wait = t0 + (e.ts - first_ts) / speedup - time.perf_counter()
        else:
            wait = 0.0
        if wait > 0:
            time.sleep(wait)
        msg = Event(e.visitor_id, e.item_id, e.event, e.ts, sent_at=time.time())
        while True:
            try:
                producer.produce(topic, key=str(e.visitor_id), value=msg.to_bytes())
                break
            except BufferError:  # local queue full: let it drain
                producer.poll(0.05)
        producer.poll(0)
        sent += 1
    producer.flush()
    secs = time.perf_counter() - t0
    return {"sent": sent, "seconds": secs, "events_per_s": sent / max(secs, 1e-9)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default=TOPIC)
    p.add_argument("--processed", default=None)
    p.add_argument("--split", default="test", choices=["train", "tune", "test"])
    p.add_argument("--rate", type=float, default=0.0)
    p.add_argument("--speedup", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    path = processed_dir() if a.processed is None else a.processed
    df = pd.read_parquet(f"{path}/{a.split}.parquet")
    out = replay(iter_events(df), make_producer(a.bootstrap), a.topic, a.rate, a.speedup, a.limit)
    print(f"sent {out['sent']:,} events in {out['seconds']:.1f}s "
          f"({out['events_per_s']:,.0f}/s)")  # fmt: skip


if __name__ == "__main__":
    main()
