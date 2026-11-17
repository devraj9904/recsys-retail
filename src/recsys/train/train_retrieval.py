"""Train and compare retrieval models: popularity, ALS, two-tower + FAISS.

    python -m recsys.train.train_retrieval            # real data, defaults
    python -m recsys.train.train_retrieval --quick    # small smoke run

Protocol: models learn from the train split only. At evaluation each visitor
is represented by their latest pre-test history (train + tune), and the
targets are the distinct items they touch in the 14-day test window. Items
outside the warm catalogue count as misses, so CF numbers are not inflated
by dropping cold items (Phase 4 handles those).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.config import ROOT, processed_dir
from recsys.eval.metrics import evaluate
from recsys.index.ann import ItemIndex
from recsys.models.als import ALSRetriever, interaction_matrix
from recsys.models.popularity import popularity_scores, recommend_popular
from recsys.models.two_tower import TwoTower, build_item_features, train_two_tower
from recsys.retrieval.data import build_eval_set, build_streams, load_processed, training_examples

KS = (10, 20, 50, 100)


def _drop_seen(recs: np.ndarray, hist: np.ndarray, k: int) -> np.ndarray:
    """Remove each visitor's already-seen items, keep the next k (pad -2)."""
    out = np.full((len(recs), k), -2, dtype=np.int64)
    for u in range(len(recs)):
        fresh = recs[u][~np.isin(recs[u], hist[u])][:k]
        out[u, : len(fresh)] = fresh
    return out


def _eval_both(recs_long, es, n_items) -> dict:
    """Standard metrics plus *discovery* metrics on items the visitor never saw.

    Repeat visits to already-viewed items are a big share of e-commerce
    traffic, so a model can score well just by echoing history. The discovery
    numbers drop seen items from both the ranking and the targets.
    """
    recs = recs_long[:, : max(KS)]
    out = evaluate(recs, es.targets, KS, n_items)
    fresh = _drop_seen(recs_long, es.hist_items, max(KS))
    new_targets = [t[~np.isin(t, h)] for t, h in zip(es.targets, es.hist_items, strict=True)]
    disc = evaluate(fresh, new_targets, (20, 100))
    out["discovery_recall@20"] = disc["recall@20"]
    out["discovery_recall@100"] = disc["recall@100"]
    out["discovery_users"] = disc["users"]
    buyers = [i for i, t in enumerate(es.purchase_targets) if len(t)]
    if buyers:
        p = evaluate(recs[buyers], [es.purchase_targets[i] for i in buyers], (20, 100))
        out["purchase_recall@20"] = p["recall@20"]
        out["purchase_recall@100"] = p["recall@100"]
        out["purchase_users"] = p["users"]
    return out


def run(data: Path, out_dir: Path, args) -> dict:
    t0 = time.perf_counter()
    d = load_processed(data)
    item_map = d["item_map"]
    n_items = len(item_map)
    history = pd.concat([d["train"], d["tune"]], ignore_index=True)
    es = build_eval_set(history, d["test"], item_map, args.max_history, args.eval_users)
    print(f"eval users {len(es.visitors):,} · warm items {n_items:,}")
    results, timings = {}, {}
    fetch = max(KS) + args.max_history  # room to drop seen items for discovery metrics

    # Popularity: all-time (train) and most recent week (tune window).
    for name, events in (
        ("popularity (all-time)", d["train"]),
        ("popularity (last 7d)", d["tune"]),
    ):
        scores = popularity_scores(events, item_map)
        results[name] = _eval_both(recommend_popular(scores, len(es.visitors), fetch), es, n_items)

    # Echo baseline: the visitor's own recent items, newest first, then trending.
    recent = recommend_popular(popularity_scores(d["tune"], item_map), 1, fetch)[0]
    echo = np.empty((len(es.visitors), fetch), dtype=np.int64)
    for u, h in enumerate(es.hist_items):
        mine = list(dict.fromkeys(int(x) for x in h[::-1] if x >= 0))
        echo[u] = (mine + [x for x in recent if x not in set(mine)])[:fetch]
    results["recently viewed + trending"] = _eval_both(echo, es, n_items)

    # ALS on train, user factors re-solved from each visitor's pre-test history.
    t = time.perf_counter()
    train_mat, _ = interaction_matrix(d["train"], item_map, n_items)
    als = ALSRetriever(factors=args.dim, iterations=args.als_iters).fit(train_mat)
    hist_ev = history[history["visitor_id"].isin(es.visitors)]
    hist_mat, hist_users = interaction_matrix(hist_ev, item_map, n_items)
    order = np.searchsorted(hist_users, es.visitors)
    results["ALS"] = _eval_both(als.recommend(hist_mat[order], fetch), es, n_items)
    timings["ALS train+eval (s)"] = round(time.perf_counter() - t, 1)

    # Two-tower trained on next-item prediction within train streams.
    t = time.perf_counter()
    streams = build_streams(d["train"], item_map)
    hist, hist_w, target = training_examples(streams, args.max_history)
    print(f"two-tower examples {len(target):,}")
    feats = build_item_features(d["items"], item_map)
    model = TwoTower(n_items, feats, dim=args.dim, max_history=args.max_history)
    freq = np.bincount(target, minlength=n_items).astype(np.float64)
    losses = train_two_tower(
        model, hist, hist_w, target, freq, epochs=args.epochs, batch_size=args.batch_size
    )
    timings["two-tower train (s)"] = round(time.perf_counter() - t, 1)

    item_vecs = model.item_matrix()
    user_vecs = model.user_matrix(es.hist_items, es.hist_weights)
    for kind in ("flat", "hnsw"):
        index = ItemIndex(item_vecs, kind=kind)
        t = time.perf_counter()
        _, recs = index.search(user_vecs, fetch)
        elapsed = time.perf_counter() - t
        results[f"two-tower + FAISS {kind}"] = _eval_both(recs, es, n_items)
        timings[f"FAISS {kind} search, ms/query (batch)"] = round(
            1000 * elapsed / len(user_vecs), 4
        )
    # Single-query latency, which is what the online path pays.
    index = ItemIndex(item_vecs, kind="hnsw")
    lat = []
    for q in user_vecs[:2000]:
        t = time.perf_counter()
        index.search(q[None, :], 200)
        lat.append((time.perf_counter() - t) * 1000)
    timings["FAISS hnsw single query p50/p99 (ms)"] = [
        round(float(np.percentile(lat, 50)), 3),
        round(float(np.percentile(lat, 99)), 3),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "two_tower.pt")
    np.save(out_dir / "item_vectors.npy", item_vecs)
    summary = {
        "eval_users": int(len(es.visitors)),
        "warm_items": n_items,
        "two_tower_examples": int(len(target)),
        "two_tower_loss": [round(x, 4) for x in losses],
        "config": vars(args) | {"data": str(data), "out": str(out_dir)},
        "results": results,
        "timings": timings,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "retrieval_results.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def render(summary: dict) -> str:
    cols = [
        "recall@20",
        "recall@100",
        "ndcg@20",
        "discovery_recall@20",
        "discovery_recall@100",
        "purchase_recall@100",
        "coverage@100",
    ]
    lines = [
        "## Retrieval results",
        "",
        f"{summary['eval_users']:,} test visitors with pre-test history · "
        f"{summary['warm_items']:,} warm items · two-tower trained on "
        f"{summary['two_tower_examples']:,} next-item examples",
        "",
        "| model | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name, r in summary["results"].items():
        lines.append(
            f"| {name} | " + " | ".join(f"{r.get(c, float('nan')):.4f}" for c in cols) + " |"
        )
    lines += ["", "| timing | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["timings"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=processed_dir())
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "retrieval")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "retrieval.md")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--max-history", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--als-iters", type=int, default=15)
    ap.add_argument("--eval-users", type=int, default=20000)
    args = ap.parse_args()
    summary = run(args.data, args.out, args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(summary))
    print(render(summary))


if __name__ == "__main__":
    main()
