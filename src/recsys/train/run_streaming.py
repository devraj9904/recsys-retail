"""Phase 3 benchmark: Kafka -> Redis freshness, throughput, and the value of fresh state.

    python -m recsys.train.run_streaming            # needs Kafka on :9092, Redis on :6379

1. Warm-start Redis with each replayed visitor's pre-test history.
2. Replay the test window into a fresh Kafka topic at several rates while a
   consumer thread updates online features and user vectors.
3. Report per-event freshness (produce -> new vector readable in Redis),
   sustained throughput, and one visitor's recommendations before/after.
4. Offline: next-event Hit@20 / MRR@20 of the live vector vs a stale one.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import ROOT, processed_dir
from recsys.streaming.consumer import make_consumer, run
from recsys.streaming.evaluate import next_event_eval
from recsys.streaming.live import LiveRecommender, warm_start
from recsys.streaming.producer import iter_events, make_producer, replay


def _create_topic(bootstrap: str, topic: str, partitions: int = 3) -> None:
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})  # keep alive until done
    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=partitions, replication_factor=1)]
    )
    for f in futures.values():
        f.result(timeout=30)


def _wait_for_assignment(consumer, timeout_s: float = 30.0) -> None:
    """Join the group before producing, so rebalance time isn't counted as lag."""
    end = time.perf_counter() + timeout_s
    while not consumer.assignment() and time.perf_counter() < end:
        consumer.poll(0.1)


def stream_run(live, client, history, test, bootstrap, rate, limit) -> dict:
    events = test.sort_values("ts", kind="stable").head(limit)
    client.flushdb()
    t0 = time.perf_counter()
    n_warm = warm_start(live, history, events["visitor_id"].unique())
    warm_s = time.perf_counter() - t0

    # Example visitor: has history and several test events.
    counts = events["visitor_id"].value_counts()
    cand = [v for v in counts.index[:200] if live.store.user_vector(int(v)) is not None]
    example = int(cand[len(cand) // 2]) if cand else int(counts.index[0])
    before = live.recommend(example, 10)

    topic = f"events-{uuid.uuid4().hex[:8]}"
    _create_topic(bootstrap, topic)
    consumer = make_consumer(bootstrap, f"bench-{topic}", topic)
    _wait_for_assignment(consumer)
    producer = make_producer(bootstrap)
    producer.list_topics(timeout=30)  # connect before the clock starts
    live.refresh_vectors([example])  # warm the model's first-call path
    box: dict = {}
    th = threading.Thread(
        target=lambda: box.setdefault("st", run(consumer, live, max_events=len(events))),
        daemon=True,
    )
    th.start()
    prod = replay(iter_events(events), producer, topic, rate=rate)
    th.join()
    consumer.close()
    after = live.recommend(example, 10)
    ex_events = events[events["visitor_id"] == example].sort_values("ts")
    return {
        "rate": rate or "max",
        "events": len(events),
        "warm_visitors": n_warm,
        "warm_start_s": warm_s,
        "producer": prod,
        "consumer": box["st"].summary(),
        "example": {
            "visitor_id": example,
            "stream_events": ex_events[["item_id", "event"]].astype(str).values.tolist()[:10],
            "before": before,
            "after": after,
        },
    }


def _rate(r) -> str:
    return "max (capacity)" if r in ("max", 0, 0.0) else f"{float(r):,.0f}/s"


def render(res: dict) -> str:
    L = ["# Phase 3: streaming features and live user vectors", ""]
    L += ["Kafka (KRaft, single broker) -> consumer -> Redis online store + two-tower",
          "user tower. Freshness is per event: producer timestamp to the moment the",
          "visitor's re-encoded vector is readable in Redis. One consumer process,",
          "CPU only.", ""]  # fmt: skip
    L += ["## Freshness and throughput", "",
          "| replay rate | events | consumer events/s | p50 ms | p95 ms | p99 ms | "
          "batch p50 ms | mean batch |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in res["runs"]:
        c = r["consumer"]
        L.append(
            f"| {_rate(r['rate'])} | {r['events']:,} | {c['events_per_s']:,.0f} | "
            f"{c['freshness_p50_ms']:.0f} | {c['freshness_p95_ms']:.0f} | "
            f"{c['freshness_p99_ms']:.0f} | {c['batch_p50_ms']:.1f} | {c['mean_batch_size']:.0f} |"
        )
    L += ["", "## Does fresh state help? Next-event prediction", ""]
    ev = res["value"]
    cnt = ev.pop("_counts")
    keys = list(next(iter(ev.values())).keys())
    L += ["| user state | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for name, m in ev.items():
        L.append(f"| {name} | " + " | ".join(f"{m[k]:.4f}" for k in keys) + " |")
    ev["_counts"] = cnt
    L += ["", f"{cnt['events']:,} test events from {cnt['visitors']:,} visitors with pre-test "
          f"history ({cnt['new item events']:,} on items not in the visitor's last 30 steps).",
          ""]  # fmt: skip
    ex = res["runs"][0]["example"]
    L += ["## Example: one visitor before and after the stream", "",
          f"Visitor {ex['visitor_id']}, first stream events: "
          + ", ".join(f"{e} {i}" for i, e in ex["stream_events"]), "",
          "| # | before (batch state) | after (live state) |", "|---:|---|---|"]  # fmt: skip
    for i, (b, a) in enumerate(zip(ex["before"], ex["after"], strict=False), 1):
        L.append(f"| {i} | {b} | {a} |")
    L += ["", "## Notes", "",
          "- Each replay run is the first N events of the 14-day test window, sent to a fresh",
          "  3-partition topic; Redis is flushed and warm-started with the replayed visitors'",
          "  last 30 pre-test events first (the batch to online hand-off).",
          "- The `max` run measures capacity: the producer outruns the single consumer, so",
          "  its freshness is queueing delay, not processing time. Per-batch cost (Redis",
          "  pipeline + Lua session/affinity script + user-tower forward pass) is in the",
          "  batch columns.",
          "- For scale: RetailRocket's 2.7M events over 4.5 months average about 0.25",
          "  events/s, so the paced runs are thousands of times the dataset's real traffic.",
          "- The prediction table is offline and deterministic (no Kafka timing involved):",
          "  every test event of a visitor with pre-test history, predicted from the",
          "  vector frozen at the cutoff vs the vector after all earlier events. Events on",
          "  items outside the trained catalogue are dropped from history and targets",
          "  (Phase 4 handles them), so these hit rates are per warm next event and not",
          "  comparable to Phase 1 recall, which counts cold targets as misses.",
          "- Single machine (container CPU), single broker and consumer, no replication;",
          "  numbers will differ on other hardware."]  # fmt: skip
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    import redis
    import torch

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, default=None)
    p.add_argument("--retrieval", type=Path, default=ROOT / "artifacts" / "retrieval")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "streaming")
    p.add_argument("--report", type=Path, default=ROOT / "reports" / "streaming.md")
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--redis", default="redis://localhost:6379/1")
    p.add_argument("--rates", default="0,2000,500", help="events/s per run, 0 = max")
    p.add_argument("--limit", type=int, default=60_000, help="events per run")
    p.add_argument("--eval-events", type=int, default=50_000)
    a = p.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    proc = a.processed or processed_dir()
    d = {k: pd.read_parquet(proc / f"{k}.parquet") for k in ("train", "tune", "test", "item_map")}
    history = pd.concat([d["train"], d["tune"]], ignore_index=True)
    client = redis.Redis.from_url(a.redis)
    live = LiveRecommender.from_artifacts(client, proc, a.retrieval)

    runs = []
    for rate in [float(x) for x in a.rates.split(",")]:
        r = stream_run(live, client, history, d["test"], a.bootstrap, rate, a.limit)
        print(json.dumps({k: r[k] for k in ("rate", "events")} | r["consumer"]), flush=True)
        runs.append(r)

    w = history.merge(d["item_map"], on="item_id")
    popular = w.groupby("item_idx")["weight"].sum().sort_values(ascending=False).index.to_numpy()
    value = next_event_eval(live.model, live.index, history, d["test"], d["item_map"],
                            popular, max_events=a.eval_events)  # fmt: skip
    print(json.dumps(value, indent=1), flush=True)

    res = {"runs": runs, "value": value}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "streaming_results.json").write_text(json.dumps(res, indent=1, default=str))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(render(res))
    print(f"wrote {a.report}")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
