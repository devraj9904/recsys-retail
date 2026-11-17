"""Train and evaluate the ranking stage on top of Phase 1 retrieval.

    python -m recsys.train.train_ranker

1. Candidates (two-tower top-100 + own history + trending) are built twice:
   at the start of the tune window (training rows, history = train) and at
   the start of the test window (evaluation rows, history = train + tune).
2. Labels on tune come from the strongest event each visitor had on each
   candidate during the tune week.
3. LambdaRank (engagement) and a purchase classifier are trained on tune,
   blended, optionally diversified with MMR, and evaluated on the 14-day
   test window with the same metrics as Phase 1.
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
from recsys.models.popularity import popularity_scores
from recsys.models.two_tower import TwoTower, build_item_features, train_two_tower
from recsys.ranking.candidates import generate_candidates
from recsys.ranking.diversity import intra_list_diversity, mmr
from recsys.ranking.features import FEATURES, attach_labels, build_features
from recsys.ranking.ranker import blend, train_lambdarank, train_purchase
from recsys.retrieval.data import build_eval_set, build_streams, load_processed, training_examples

K = 20


def _load_or_train_two_tower(d, item_map, feats, artifacts: Path, max_history: int):
    model = TwoTower(len(item_map), feats, dim=64, max_history=max_history)
    weights = artifacts / "two_tower.pt"
    if weights.exists():
        model.load_state_dict(torch.load(weights))
    else:
        s = build_streams(d["train"], item_map)
        hist, w, tgt = training_examples(s, max_history)
        freq = np.bincount(tgt, minlength=len(item_map)).astype(float)
        train_two_tower(model, hist, w, tgt, freq, epochs=5)
    return model


def _ranked_lists(df: pd.DataFrame, score: str, n_users: int, k: int) -> np.ndarray:
    d = df[["u", "item_idx", score]].sort_values(["u", score], ascending=[True, False])
    d = d[d.groupby("u").cumcount() < k]
    out = np.full((n_users, k), -2, dtype=np.int64)
    pos = d.groupby("u").cumcount().to_numpy()
    out[d["u"].to_numpy(), pos] = d["item_idx"].to_numpy()
    return out


def _metrics(recs, es, item_vecs, feats) -> dict:
    m = evaluate(recs, es.targets, (10, 20))
    buyers = [i for i, t in enumerate(es.purchase_targets) if len(t)]
    p = evaluate(recs[buyers], [es.purchase_targets[i] for i in buyers], (20,))
    ild, cats = [], []
    for row in recs:
        row = row[row >= 0]
        ild.append(intra_list_diversity(item_vecs[row]))
        cats.append(len(np.unique(feats.leaf[row])))
    return {
        "ndcg@10": m["ndcg@10"],
        "ndcg@20": m["ndcg@20"],
        "recall@20": m["recall@20"],
        "purchase_recall@20": p["recall@20"],
        "ild@20": float(np.mean(ild)),
        "categories@20": float(np.mean(cats)),
    }


def _stage(d, history, future, cutoff, model, index, item_vecs, feats, item_map, args, max_users):
    es = build_eval_set(history, future, item_map, args.max_history, max_users, seed=1)
    user_vecs = model.user_matrix(es.hist_items, es.hist_weights)
    week = history[history["ts"] >= cutoff - pd.Timedelta(days=7)]
    trending = np.argsort(-popularity_scores(week, item_map), kind="stable")
    cands = generate_candidates(es, user_vecs, item_vecs, index, trending, n_tt=args.n_tt)
    df = build_features(cands, history, item_map, feats, cutoff)
    df = attach_labels(df, future, item_map)
    return es, df


def run(args) -> dict:
    t0 = time.perf_counter()
    d = load_processed(args.data)
    item_map = d["item_map"]
    feats = build_item_features(d["items"], item_map)
    model = _load_or_train_two_tower(d, item_map, feats, args.retrieval, args.max_history)
    item_vecs = model.item_matrix()
    index = ItemIndex(item_vecs, kind="hnsw")

    tune_cut, test_cut = d["tune"]["ts"].min(), d["test"]["ts"].min()
    pre_test = pd.concat([d["train"], d["tune"]], ignore_index=True)
    _, train_df = _stage(d, d["train"], d["tune"], tune_cut, model, index, item_vecs, feats,
                         item_map, args, None)  # fmt: skip
    es, test_df = _stage(d, pre_test, d["test"], test_cut, model, index, item_vecs, feats,
                         item_map, args, args.eval_users)  # fmt: skip
    print(f"train rows {len(train_df):,} ({train_df['visitor_id'].nunique():,} visitors), "
          f"test rows {len(test_df):,} ({len(es.visitors):,} visitors)")  # fmt: skip

    t = time.perf_counter()
    ranker = train_lambdarank(train_df)
    buyer = train_purchase(train_df)
    train_seconds = time.perf_counter() - t
    test_df["engagement"] = ranker.predict(test_df[FEATURES])
    test_df["p_buy"] = buyer.predict_proba(test_df[FEATURES])[:, 1]
    test_df["neg_tt_rank"] = -test_df["tt_rank"]
    test_df["echo"] = np.where(
        test_df["from_hist"] == 1, 1000 - test_df["hist_pos"], -test_df["trend_rank"]
    )

    n = len(es.visitors)
    results = {}
    # Candidate-set ceiling: how many targets any ranking of these candidates could reach.
    pos = test_df[test_df["label"] > 0].groupby("u").size()
    n_targets = np.array([len(t) for t in es.targets])
    results["_candidate_recall"] = float((pos.reindex(range(n), fill_value=0) / n_targets).mean())
    variants = {
        "recently viewed + trending": "echo",
        "two-tower order": "neg_tt_rank",
        "LambdaRank (engagement)": "engagement",
        "conversion model only": "p_buy",
    }
    for name, col in variants.items():
        results[name] = _metrics(_ranked_lists(test_df, col, n, K), es, item_vecs, feats)
    for w in args.blend_weights:
        test_df[f"blend_{w}"] = blend(test_df["engagement"], test_df["p_buy"], test_df["u"], w)
        results[f"LambdaRank + conversion (w={w})"] = _metrics(
            _ranked_lists(test_df, f"blend_{w}", n, K), es, item_vecs, feats
        )

    # Diversity: MMR over each visitor's top-50 by the chosen blend.
    base = f"blend_{args.mmr_base_weight}"
    top = test_df.sort_values(["u", base], ascending=[True, False])
    top = top[top.groupby("u").cumcount() < 50]
    groups = {u: g for u, g in top.groupby("u")}
    for lam in args.mmr_lambdas:
        recs = np.full((n, K), -2, dtype=np.int64)
        for u, g in groups.items():
            items = g["item_idx"].to_numpy()
            order = mmr(g[base].to_numpy(), item_vecs[items], K, lam)
            recs[u, : len(order)] = items[order]
        results[f"+ MMR λ={lam}"] = _metrics(recs, es, item_vecs, feats)

    imp = pd.Series(ranker.booster_.feature_importance("gain"), index=FEATURES)
    summary = {
        "train_rows": int(len(train_df)),
        "train_visitors": int(train_df["visitor_id"].nunique()),
        "train_positive_visitors": int((train_df.groupby("visitor_id")["label"].max() > 0).sum()),
        "test_rows": int(len(test_df)),
        "eval_users": int(n),
        "ranker_trees": int(ranker.best_iteration_ or ranker.n_estimators),
        "ranker_train_seconds": round(train_seconds, 1),
        "feature_importance": (imp / imp.sum()).sort_values(ascending=False).round(4).to_dict(),
        "results": results,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    ranker.booster_.save_model(str(args.out / "lambdarank.txt"))
    buyer.booster_.save_model(str(args.out / "purchase.txt"))
    (args.out / "ranking_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def render(s: dict) -> str:
    cols = ["ndcg@10", "ndcg@20", "recall@20", "purchase_recall@20", "ild@20", "categories@20"]
    lines = [
        "## Ranking results",
        "",
        f"Trained on {s['train_rows']:,} candidate rows from {s['train_positive_visitors']:,} "
        f"tune-week visitors with a positive; evaluated on {s['eval_users']:,} test visitors "
        f"({s['test_rows']:,} candidates). Candidate recall ceiling: "
        f"{s['results']['_candidate_recall']:.3f}.",
        "",
        "| ranking | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name, r in s["results"].items():
        if name.startswith("_"):
            continue
        lines.append(f"| {name} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " |")
    lines += ["", "Top features (LambdaRank gain share):", ""]
    lines += [f"- `{k}`: {v:.1%}" for k, v in list(s["feature_importance"].items())[:10]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=processed_dir())
    ap.add_argument("--retrieval", type=Path, default=ROOT / "artifacts" / "retrieval")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "ranking")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "ranking.md")
    ap.add_argument("--max-history", type=int, default=30)
    ap.add_argument("--n-tt", type=int, default=100)
    ap.add_argument("--eval-users", type=int, default=20000)
    ap.add_argument("--blend-weights", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    ap.add_argument("--mmr-base-weight", type=float, default=0.2)
    ap.add_argument("--mmr-lambdas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    args = ap.parse_args()
    s = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(s))
    print(render(s))


if __name__ == "__main__":
    main()
