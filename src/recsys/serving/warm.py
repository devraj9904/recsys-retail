"""Load the online store before serving (the nightly batch -> Redis hand-off).

    python -m recsys.serving.warm            # demo visitors + returning visitors of the test window

Writes each selected visitor's last ``--per-visitor`` pre-cutoff events and
their two-tower vector, resets the live trending set, and sets the data
clock to the cutoff.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def warm(rec, processed: Path, per_visitor: int = 90, max_visitors: int | None = None) -> int:
    from recsys.streaming.live import warm_start

    d = {k: pd.read_parquet(processed / f"{k}.parquet") for k in ("train", "tune", "test")}
    history = pd.concat([d["train"], d["tune"]], ignore_index=True)
    active = np.intersect1d(d["test"]["visitor_id"].unique(), history["visitor_id"].unique())
    if max_visitors and len(active) > max_visitors:
        active = np.random.default_rng(0).choice(active, max_visitors, replace=False)
    visitors = np.union1d(active, np.asarray(rec.b.demo_visitors, dtype=np.int64))
    n = warm_start(rec.live, history, visitors, per_visitor=per_visitor)
    rec.store.r.delete("trending")
    rec.store.r.set("clock", rec.b.cutoff)
    return n


def main(argv=None) -> None:  # pragma: no cover - wiring only
    import os

    import redis

    from recsys.config import ROOT, processed_dir
    from recsys.serving.recommender import Recommender

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, default=None)
    p.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    p.add_argument("--redis", default=os.environ.get("RECSYS_REDIS", "redis://localhost:6379/0"))
    p.add_argument("--per-visitor", type=int, default=90)
    p.add_argument("--max-visitors", type=int, default=None)
    p.add_argument("--flush", action="store_true", help="empty the Redis db first")
    a = p.parse_args(argv)
    client = redis.Redis.from_url(a.redis)
    if a.flush:
        client.flushdb()
    proc = a.processed or processed_dir()
    art = a.artifacts
    rec = Recommender.from_artifacts(
        client, art / "serving", proc, art / "retrieval", art / "ranking"
    )
    t = time.perf_counter()
    n = warm(rec, proc, a.per_visitor, a.max_visitors)
    print(f"warmed {n:,} visitors in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
