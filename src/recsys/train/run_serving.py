"""Phase 5 benchmark: latency and throughput of the running API.

    python -m recsys.train.run_serving --url http://localhost:8000

Runs the closed-loop load test (recsys.serving.loadtest) at increasing
concurrency with the response cache bypassed, then once with the cache on,
and writes reports/serving.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from recsys.config import ROOT, processed_dir
from recsys.serving.loadtest import run, sample_visitors

STAGES = ("retrieval", "features", "ranking", "diversity", "total", "content_knn")


def render(res: dict) -> str:
    env = res["env"]
    L = ["# Phase 5: serving latency and throughput", "",
         f"API: {env['workers']} uvicorn workers, HNSW index, {env['cpus']} vCPUs shared with "
         "the load generator, Redis on the same machine. Mix per client loop: 70% "
         "recommendations for returning visitors, 10% for unknown visitors, 20% click events.",
         ""]  # fmt: skip
    L += ["## Latency vs load (cache bypassed: every request runs the full pipeline)", "",
          "| clients | requests/s | p50 ms | p95 ms | p99 ms | returning-visitor recs p95 ms | "
          "event p95 ms | errors |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in res["sweep"]:
        a, w, e = r["latency_ms_all"], r["latency_ms"]["recommend_warm"], r["latency_ms"]["event"]
        L.append(f"| {r['concurrency']} | {r['rps']:,.0f} | {a['p50']:.1f} | {a['p95']:.1f} | "
                 f"{a['p99']:.1f} | {w['p95']:.1f} | {e['p95']:.1f} | {r['errors']} |")  # fmt: skip
    c = res["cached"]
    a = c["latency_ms_all"]
    L += ["", "## With the Redis response cache", "",
          f"{c['concurrency']} clients: {c['rps']:,.0f} requests/s, p50 {a['p50']:.1f} ms, "
          f"p95 {a['p95']:.1f} ms, p99 {a['p99']:.1f} ms, cache hit rate "
          f"{c['cache_hit_rate']:.0%}. Hits skip the pipeline; every click event drops that "
          "visitor's cached response.", ""]  # fmt: skip
    one = res["sweep"][0]["server_stage_mean_ms"]
    L += ["## Where the time goes (server-side mean, one client)", "",
          "| stage | ms |", "|---|---:|"]  # fmt: skip
    names = {"retrieval": "retrieval: Redis reads + HNSW top-100 + candidate merge",
             "features": "24 ranking features", "ranking": "LambdaRank + conversion model",
             "diversity": "MMR over top 50", "total": "ranked route, total",
             "content_knn": "cold-start content kNN (TF-IDF over 233k items)"}  # fmt: skip
    for s in STAGES:
        if s in one:
            L.append(f"| {names[s]} | {one[s]:.2f} |")
    L += ["", "## Notes", "",
          "- Latency is client-side and includes HTTP, JSON and queueing. The gap between the",
          "  ~2-3 ms pipeline and the client p50 at higher load is queueing for CPU: the",
          "  load generator runs on the same 4 vCPUs as the API.",
          "- Throughput flattens past 8 clients because the machine is CPU-bound; more",
          "  workers or machines scale it horizontally (the API keeps no state in process).",
          "- Single-node Redis without persistence; no network hop between services."]  # fmt: skip
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--processed", type=Path, default=None)
    p.add_argument("--duration", type=float, default=20)
    p.add_argument("--clients", default="1,4,8,16")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 4)))
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "serving_bench.json")
    p.add_argument("--report", type=Path, default=ROOT / "reports" / "serving.md")
    a = p.parse_args(argv)
    vis, items = sample_visitors(a.processed or processed_dir(), 5000)
    mix = (0.7, 0.1, 0.2)
    sweep = []
    for c in [int(x) for x in a.clients.split(",")]:
        r = asyncio.run(run(a.url, vis, items, c, a.duration, mix, nocache=True))
        print(c, r["rps"], r["latency_ms_all"], flush=True)
        sweep.append(r)
    cached = asyncio.run(run(a.url, vis, items, 8, a.duration, mix, nocache=False))
    res = {"env": {"workers": a.workers, "cpus": os.cpu_count()}, "sweep": sweep,
           "cached": cached}  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    a.report.write_text(render(res))
    print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
