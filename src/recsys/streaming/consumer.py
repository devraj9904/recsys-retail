"""Consume the click stream and keep the online state current.

    python -m recsys.streaming.consumer --group live-features

Each poll takes up to ``batch`` messages, writes them to Redis in one
pipeline, re-encodes the touched visitors' user vectors and commits the
offsets. Freshness is measured per event as wall time from ``sent_at``
(producer) to the moment the visitor's new vector is in Redis, i.e. the
point where a recommendation request would see the click.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from recsys.streaming.events import TOPIC, Event


@dataclass
class ConsumerStats:
    events: int = 0
    batches: int = 0
    bad: int = 0
    freshness_ms: list[float] = field(default_factory=list)
    batch_ms: list[float] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    first_event: float | None = None
    last_event: float | None = None

    def summary(self) -> dict[str, float]:
        f = np.asarray(self.freshness_ms) if self.freshness_ms else np.zeros(1)
        b = np.asarray(self.batch_ms) if self.batch_ms else np.zeros(1)
        span = (self.last_event or 0) - (self.first_event or 0)
        return {
            "events": self.events,
            "batches": self.batches,
            "bad_messages": self.bad,
            "events_per_s": self.events / span if span > 0 else 0.0,
            "freshness_p50_ms": float(np.percentile(f, 50)),
            "freshness_p95_ms": float(np.percentile(f, 95)),
            "freshness_p99_ms": float(np.percentile(f, 99)),
            "freshness_max_ms": float(f.max()),
            "batch_p50_ms": float(np.percentile(b, 50)),
            "batch_p95_ms": float(np.percentile(b, 95)),
            "mean_batch_size": self.events / max(self.batches, 1),
        }


def make_consumer(bootstrap: str, group: str, topic: str = TOPIC):
    from confluent_kafka import Consumer

    c = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "fetch.wait.max.ms": 10,
        }
    )
    c.subscribe([topic])
    return c


def run(
    consumer,
    live,
    max_events: int | None = None,
    idle_timeout_s: float = 10.0,
    batch: int = 1000,
    stop=None,
) -> ConsumerStats:
    """Poll until ``max_events`` are processed, ``stop`` is set, or the topic goes idle."""
    st = ConsumerStats()
    idle_since = time.perf_counter()
    while True:
        if stop is not None and stop.is_set():
            break
        if max_events is not None and st.events >= max_events:
            break
        msgs = consumer.consume(num_messages=batch, timeout=0.05)
        if not msgs:
            if time.perf_counter() - idle_since > idle_timeout_s and st.events > 0:
                break
            continue
        idle_since = time.perf_counter()
        t0 = time.perf_counter()
        events = []
        for m in msgs:
            if m.error():
                st.bad += 1
                continue
            try:
                events.append(Event.from_bytes(m.value()))
            except (ValueError, KeyError):
                st.bad += 1
        live.ingest(events)
        consumer.commit(asynchronous=True)
        done_wall, done = time.time(), time.perf_counter()
        st.freshness_ms.extend((done_wall - e.sent_at) * 1000 for e in events if e.sent_at)
        st.batch_ms.append((done - t0) * 1000)
        st.events += len(events)
        st.batches += 1
        st.first_event = st.first_event or t0
        st.last_event = done
    return st


def main(argv: list[str] | None = None) -> None:
    from pathlib import Path

    import redis

    from recsys.config import ROOT, processed_dir
    from recsys.streaming.live import LiveRecommender

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--redis", default="redis://localhost:6379/0")
    p.add_argument("--topic", default=TOPIC)
    p.add_argument("--group", default="live-features")
    p.add_argument("--processed", type=Path, default=None)
    p.add_argument("--retrieval", type=Path, default=ROOT / "artifacts" / "retrieval")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--idle-timeout", type=float, default=float("inf"))
    a = p.parse_args(argv)
    live = LiveRecommender.from_artifacts(
        redis.Redis.from_url(a.redis), a.processed or processed_dir(), a.retrieval
    )
    st = run(make_consumer(a.bootstrap, a.group, a.topic), live, a.max_events, a.idle_timeout)
    for k, v in st.summary().items():
        print(f"{k:>20}: {v:,.2f}")


if __name__ == "__main__":
    main()
