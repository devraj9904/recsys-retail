"""Closed-loop load test against a running API.

    python -m recsys.serving.loadtest --url http://localhost:8000 --concurrency 32 --duration 30

Each of ``concurrency`` clients loops over a weighted mix of requests:
recommendations for returning visitors (warm in Redis), recommendations for
unknown visitors (cold start), and click events, which invalidate that
visitor's cached response. Latency is measured client-side; the per-stage
server breakdown is read from /metrics before and after the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import numpy as np


def _stage_totals(text: str) -> dict[str, tuple[float, float]]:
    out = {}
    for m in re.finditer(r'recsys_stage_seconds_(sum|count)\{stage="([^"]+)"\} ([0-9.e+-]+)', text):
        kind, stage, val = m.group(1), m.group(2), float(m.group(3))
        s, c = out.get(stage, (0.0, 0.0))
        out[stage] = (s + val, c) if kind == "sum" else (s, c + val)
    return out


async def run(url, visitors, items, concurrency, duration, mix, nocache, seed=0) -> dict:
    import httpx

    lat: dict[str, list[float]] = {"recommend_warm": [], "recommend_cold": [], "event": []}
    errors = 0
    routes: dict[str, int] = {}
    cached = 0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=10, limits=limits) as client:
        before = _stage_totals((await client.get("/metrics")).text)
        end = time.perf_counter() + duration

        async def worker(wid: int):
            nonlocal errors, cached
            r = random.Random(seed * 1000 + wid)
            while time.perf_counter() < end:
                x = r.random()
                t = time.perf_counter()
                try:
                    if x < mix[0]:
                        kind = "recommend_warm"
                        resp = await client.get(f"/recommend/{r.choice(visitors)}",
                                                params={"k": 20, "nocache": nocache})  # fmt: skip
                    elif x < mix[0] + mix[1]:
                        kind = "recommend_cold"
                        resp = await client.get(f"/recommend/{2_000_000_000 + r.randrange(10**8)}",
                                                params={"k": 20, "nocache": nocache})  # fmt: skip
                    else:
                        kind = "event"
                        body = {"visitor_id": r.choice(visitors), "item_id": r.choice(items),
                                "event": "view"}  # fmt: skip
                        resp = await client.post("/events", json=body)
                    ok = resp.status_code == 200
                except httpx.HTTPError:
                    ok = False
                ms = (time.perf_counter() - t) * 1000
                if not ok:
                    errors += 1
                    continue
                lat[kind].append(ms)
                if kind.startswith("recommend"):
                    d = resp.json()
                    routes[d["route"]] = routes.get(d["route"], 0) + 1
                    cached += bool(d["cached"])

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(concurrency)))
        wall = time.perf_counter() - t0
        after = _stage_totals((await client.get("/metrics")).text)
    total = sum(len(v) for v in lat.values())
    allv = np.concatenate([np.asarray(v) for v in lat.values() if v]) if total else np.zeros(1)

    def pct(a):
        a = np.asarray(a) if len(a) else np.zeros(1)
        return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)}

    stages = {}
    for st, (s, c) in after.items():
        s0, c0 = before.get(st, (0.0, 0.0))
        if c - c0 > 0:
            stages[st] = round((s - s0) / (c - c0) * 1000, 3)
    n_rec = sum(routes.values())
    return {
        "concurrency": concurrency,
        "duration_s": round(wall, 1),
        "requests": total,
        "rps": round(total / wall, 1),
        "errors": errors,
        "nocache": nocache,
        "latency_ms_all": pct(allv),
        "latency_ms": {k: pct(v) | {"n": len(v)} for k, v in lat.items()},
        "routes": routes,
        "cache_hit_rate": round(cached / n_rec, 3) if n_rec else 0.0,
        "server_stage_mean_ms": stages,
    }


def sample_visitors(processed: Path, n: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Returning visitors of the test window (the ones the warm step loads) and items."""
    import pandas as pd

    tr = pd.read_parquet(processed / "train.parquet", columns=["visitor_id"])
    tu = pd.read_parquet(processed / "tune.parquet", columns=["visitor_id"])
    te = pd.read_parquet(processed / "test.parquet", columns=["visitor_id", "item_id"])
    hist = np.union1d(tr["visitor_id"].unique(), tu["visitor_id"].unique())
    active = np.intersect1d(te["visitor_id"].unique(), hist)
    rng = np.random.default_rng(seed)
    vis = rng.choice(active, min(n, len(active)), replace=False).tolist()
    items = te["item_id"].sample(min(5000, len(te)), random_state=seed).tolist()
    return [int(v) for v in vis], [int(i) for i in items]


def main(argv=None) -> None:
    from recsys.config import processed_dir

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--processed", type=Path, default=None)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--mix", default="0.7,0.1,0.2", help="warm recs, cold recs, events")
    p.add_argument("--nocache", action="store_true")
    p.add_argument("--visitors", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    vis, items = sample_visitors(a.processed or processed_dir(), a.visitors)
    mix = tuple(float(x) for x in a.mix.split(","))
    res = asyncio.run(run(a.url, vis, items, a.concurrency, a.duration, mix, a.nocache))
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
