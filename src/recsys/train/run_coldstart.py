"""Phase 4: cold start for new items and new visitors, content vs CF.

    python -m recsys.train.run_coldstart

A. Content vs CF on the same items: hold out 10% of warm items from the
   content encoder's training, then retrieve them for returning visitors with
   their real two-tower vectors vs their content-only vectors.
B. Genuinely new items (not in the trained catalogue, 11.6% of test events):
   add content vectors to the index and measure what Phase 1 had to count
   as misses.
C. New visitors (88% of test visitors): from their first 1 or 3 clicks,
   predict the rest of their test-window clicks with content, CF and hybrid
   methods.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.coldstart import methods as M
from recsys.coldstart.catalog import build_catalog, lookup
from recsys.coldstart.content import (
    ContentEncoder,
    build_content_features,
    tfidf_matrix,
    train_content_encoder,
)
from recsys.config import ROOT, processed_dir
from recsys.eval.metrics import evaluate
from recsys.index.ann import ItemIndex
from recsys.models.als import ALSRetriever, interaction_matrix
from recsys.models.two_tower import TwoTower, build_item_features
from recsys.retrieval.data import PAD, build_eval_set

KS = (10, 20)


# --------------------------------------------------------------------- helpers
def _pad_hist(context, n_warm: int, length: int = 30):
    items = np.full((len(context), length), PAD, dtype=np.int64)
    w = np.zeros((len(context), length), dtype=np.float32)
    for u, c in enumerate(context):
        c = c[c < n_warm][-length:]
        if len(c):
            items[u, length - len(c) :] = c
            w[u, length - len(c) :] = 1.0
    return items, w


def _two_tower_users(model, context, n_warm):
    h, w = _pad_hist(context, n_warm)
    valid = (w > 0).any(1)
    vecs = np.zeros((len(context), model.items.id_emb.embedding_dim), np.float32)
    if valid.any():
        vecs[valid] = model.user_matrix(h[valid], w[valid])
    return vecs, valid


def new_visitor_tasks(test: pd.DataFrame, history_visitors, catalog, n_ctx: int, max_users, seed):
    """First ``n_ctx`` distinct clicked items as context, later distinct items as targets."""
    t = test[~test["visitor_id"].isin(history_visitors)].sort_values(["visitor_id", "ts"])
    t = t.assign(cidx=lookup(catalog, t["item_id"].to_numpy()))
    t = t[t["cidx"] >= 0]
    ctx, tgt, vis = [], [], []
    for v, g in t.groupby("visitor_id", sort=False):
        seq = list(dict.fromkeys(g["cidx"].tolist()))
        if len(seq) <= n_ctx:
            continue
        ctx.append(np.array(seq[:n_ctx], dtype=np.int64))
        tgt.append(np.array(seq[n_ctx:], dtype=np.int64))
        vis.append(v)
    if max_users and len(ctx) > max_users:
        sel = np.sort(np.random.default_rng(seed).choice(len(ctx), max_users, replace=False))
        ctx, tgt, vis = [ctx[i] for i in sel], [tgt[i] for i in sel], [vis[i] for i in sel]
    return ctx, tgt, np.array(vis)


def _ev(recs, targets, n_items=None):
    return evaluate(recs, targets, KS, n_items)


# ------------------------------------------------------------------ experiments
def exp_content_vs_cf(model, item_vecs, cf, catalog, d, history, enc_epochs, max_users, seed):
    n_warm = len(item_vecs)
    rng = np.random.default_rng(seed)
    held = rng.random(n_warm) < 0.10
    content = {}
    for obj in ("cosine", "softmax"):
        enc = ContentEncoder(cf)
        tr = np.flatnonzero(~held)
        train_content_encoder(enc, tr, item_vecs[~held], epochs=enc_epochs, loss=obj)
        content[obj] = enc.encode(np.arange(n_warm))

    es = build_eval_set(history, d["test"], d["item_map"], 30, max_users, seed)
    user_vecs = model.user_matrix(es.hist_items, es.hist_weights)
    held_idx = np.flatnonzero(held)
    targets = [t[np.isin(t, held_idx)] for t in es.targets]
    keep = np.array([len(t) > 0 for t in targets])
    out = {"held-out items": int(held.sum()), "visitors": int(keep.sum())}
    variants = {
        "CF: two-tower item vectors (learned from clicks)": item_vecs,
        "content: metadata encoder (cosine objective)": np.where(
            held[:, None], content["cosine"], item_vecs
        ),
        "content: metadata encoder (in-batch softmax objective)": np.where(
            held[:, None], content["softmax"], item_vecs
        ),
    }
    # Category-only content vector: mean behavioural vector of the item's warm leaf peers.
    leaf = cf.leaf[:n_warm]
    sums = np.zeros((leaf.max() + 1, item_vecs.shape[1]), np.float32)
    np.add.at(sums, leaf[~held], item_vecs[~held])
    cat_vec = sums[leaf]
    cat_vec /= np.maximum(np.linalg.norm(cat_vec, axis=1, keepdims=True), 1e-9)
    variants["content: category centroid only"] = np.where(held[:, None], cat_vec, item_vecs)
    for name, vecs in variants.items():
        _, recs = ItemIndex(vecs).search(user_vecs[keep], 100)
        r = evaluate(recs, [t for t, k in zip(targets, keep, strict=True) if k], (20, 100))
        out[name] = {"recall@20": r["recall@20"], "recall@100": r["recall@100"],
                     "ndcg@20": r["ndcg@20"]}  # fmt: skip
    out["held-out cosine to CF vector"] = {
        "encoder (cosine)": float((content["cosine"][held] * item_vecs[held]).sum(1).mean()),
        "encoder (softmax)": float((content["softmax"][held] * item_vecs[held]).sum(1).mean()),
        "category centroid": float((cat_vec[held] * item_vecs[held]).sum(1).mean()),
    }
    return out


def exp_new_items(model, item_vecs, content_all, catalog, d, history, max_users, seed):
    n_warm = len(item_vecs)
    n_all = len(catalog)
    es = build_eval_set(history, d["test"], d["item_map"], 30, max_users, seed)
    user_vecs = model.user_matrix(es.hist_items, es.hist_weights)
    test = d["test"][d["test"]["visitor_id"].isin(es.visitors)]
    test = test.assign(cidx=lookup(catalog, test["item_id"].to_numpy()))
    by_user = test.groupby("visitor_id")["cidx"].apply(lambda s: np.unique(s.to_numpy()))
    targets = [by_user[v] for v in es.visitors]
    cold_t = [t[t >= n_warm] for t in targets]
    has_cold = np.array([len(t) > 0 for t in cold_t])

    warm_idx = ItemIndex(item_vecs)
    all_vecs = np.vstack([item_vecs, content_all[n_warm:]])
    all_idx = ItemIndex(all_vecs)
    _, r_warm = warm_idx.search(user_vecs, 20)
    _, r_all = all_idx.search(user_vecs, 20)
    _, r_cold = ItemIndex(content_all[n_warm:]).search(user_vecs, 20)
    r_cold = r_cold + n_warm
    quota = np.concatenate([r_warm[:, :17], r_cold[:, :3]], axis=1)

    res = {"visitors": len(es.visitors), "visitors with a new-item target": int(has_cold.sum()),
           "new items in catalogue": int(n_all - n_warm)}  # fmt: skip
    for name, recs in {
        "warm index only (Phase 1)": r_warm,
        "warm + new-item content vectors in one index": r_all,
        "warm top 17 + 3 new-item slots": quota,
    }.items():
        allm = evaluate(recs, targets, (20,), n_all)
        coldm = evaluate(recs[has_cold], [cold_t[i] for i in np.flatnonzero(has_cold)], (20,))
        res[name] = {"recall@20 (all targets)": allm["recall@20"],
                     "recall@20 (new-item targets)": coldm["recall@20"],
                     "share of slots on new items": float((recs >= n_warm).mean())}  # fmt: skip
    return res, cold_t, has_cold, r_cold, es


def exp_new_item_ranking(cf, tfidf, cold_t, has_cold, r_cold, es, n_warm, n_all, seed):
    """Rank only among new items: how well does each method find the right ones?"""
    idx = np.flatnonzero(has_cold)
    tg = [cold_t[i] for i in idx]
    rng = np.random.default_rng(seed)
    rand = rng.integers(n_warm, n_all, size=(len(idx), 20))
    # Category affinity: visitor's most frequent leaf among history, new items in it.
    leaf = cf.leaf
    cold_leaf = leaf[n_warm:]
    by_leaf: dict[int, np.ndarray] = {}
    order = np.argsort(cold_leaf, kind="stable")
    cuts = np.flatnonzero(np.diff(cold_leaf[order])) + 1
    for g in np.split(order, cuts):
        if len(g):
            by_leaf[int(cold_leaf[g[0]])] = g + n_warm
    cat = np.full((len(idx), 20), -2, np.int64)
    for j, u in enumerate(idx):
        h = es.hist_items[u]
        h = h[h >= 0][::-1]
        leaves = list(dict.fromkeys(int(leaf[i]) for i in h if leaf[i] > 0))
        pool = np.concatenate([by_leaf.get(c, np.empty(0, np.int64)) for c in leaves] or [[]])
        pool = rng.permutation(pool.astype(np.int64))[:20] if len(pool) > 20 else pool
        cat[j, : len(pool)] = pool[:20]
    recent = [h[h >= 0][-5:] for h in es.hist_items[idx]]
    knn = M.content_knn(tfidf, recent, 20, candidates=np.arange(n_warm, n_all))
    out = {}
    for name, recs in {
        "random new items": rand,
        "content: TF-IDF kNN of the visitor's last 5 items": knn,
        "content: new items in the visitor's categories": cat,
        "hybrid: user tower x content encoder": r_cold[idx],
    }.items():
        m = evaluate(recs, tg, KS)
        out[name] = {k: m[k] for k in ("recall@10", "recall@20", "ndcg@20")}
    out["visitors"] = len(idx)
    return out


def exp_new_visitors(model, als, coview, tfidf, content_all, all_index, pop, cf, catalog,
                     d, history, n_warm, max_users, seed):  # fmt: skip
    n_all = len(catalog)
    hist_vis = history["visitor_id"].unique()
    out = {}
    for n_ctx in (1, 3):
        ctx, tgt, _ = new_visitor_tasks(d["test"], hist_vis, catalog, n_ctx, max_users, seed)
        print(f"  new visitors, {n_ctx} click(s): {len(ctx):,}", flush=True)
        timings: dict[str, float] = {}

        def timed(name, fn, *fargs, _t=timings):
            t = time.perf_counter()
            r = fn(*fargs)
            _t[name] = round(time.perf_counter() - t, 1)
            return r

        tt_vecs, tt_valid = _two_tower_users(model, ctx, n_warm)
        cont_q = np.stack([content_all[c].mean(0) for c in ctx])
        cont_q /= np.maximum(np.linalg.norm(cont_q, axis=1, keepdims=True), 1e-9)
        popular = M.popular(pop, ctx, 20)
        hybrid_q = np.where(tt_valid[:, None], tt_vecs, cont_q)
        everyone = np.ones(len(ctx), bool)
        recs = {
            "CF: popularity (last 7 days)": popular,
            "content: popular in clicked categories": timed(
                "category", M.category_popular, pop, cf.leaf, ctx, 20
            ),
            "content: TF-IDF item kNN": timed("tfidf", M.content_knn, tfidf, ctx, 20),
            "CF: co-view item kNN": timed("coview", M.item_knn, coview, ctx, 20),
            "CF: ALS fold-in": timed("als", M.als_foldin, als, n_warm, ctx, 20),
            "CF: two-tower user tower (warm index)": timed(
                "two-tower", M.vector_search, all_index["warm"], tt_vecs, tt_valid, ctx, 20
            ),
            "hybrid: two-tower + content encoder (full index)": timed(
                "hybrid", M.vector_search, all_index["all"], hybrid_q, everyone, ctx, 20
            ),
        }
        recs["hybrid router: two-tower if a click is warm, else TF-IDF kNN"] = M.route(
            tt_valid,
            recs["CF: two-tower user tower (warm index)"],
            recs["content: TF-IDF item kNN"],
        )
        first_cold = np.array([c[0] >= n_warm for c in ctx])
        res = {
            "visitors": len(ctx),
            "share with a new item clicked first": float(first_cold.mean()),
        }
        for name, r in recs.items():
            r = M.fill(r, popular)
            m = _ev(r, tgt, n_all)
            row = {k: m[k] for k in ("recall@10", "recall@20", "ndcg@20", "coverage@20")}
            if n_ctx == 1:
                for flag, label in ((False, "first click warm"), (True, "first click new item")):
                    sel = np.flatnonzero(first_cold == flag)
                    if len(sel):
                        row[f"recall@20 ({label})"] = _ev(r[sel], [tgt[i] for i in sel])[
                            "recall@20"
                        ]
            res[name] = row
        res["_timings_s"] = timings
        out[f"{n_ctx} click"] = res
    return out


# ------------------------------------------------------------------------ main
def run(proc: Path, retrieval: Path, args) -> dict:
    t0 = time.perf_counter()
    d = {k: pd.read_parquet(proc / f"{k}.parquet")
         for k in ("train", "tune", "test", "items", "item_map")}  # fmt: skip
    history = pd.concat([d["train"], d["tune"]], ignore_index=True)
    item_map = d["item_map"]
    catalog = build_catalog(
        item_map,
        d["items"]["item_id"].to_numpy(),
        history["item_id"].to_numpy(),
        d["test"]["item_id"].to_numpy(),
    )
    n_warm, n_all = len(item_map), len(catalog)
    print(f"catalogue {n_all:,} items ({n_warm:,} warm)", flush=True)

    model = TwoTower(n_warm, build_item_features(d["items"], item_map), dim=64, max_history=30)
    model.load_state_dict(torch.load(retrieval / "two_tower.pt"))
    model.eval()
    item_vecs = np.load(retrieval / "item_vectors.npy")
    cf = build_content_features(d["items"], catalog)

    print("A. content vs CF on held-out warm items", flush=True)
    res = {"A": exp_content_vs_cf(model, item_vecs, cf, catalog, d, history, args.enc_epochs,
                                  args.max_users, args.seed)}  # fmt: skip
    print(json.dumps(res["A"], indent=1), flush=True)

    print("content encoder on all warm items", flush=True)
    enc = ContentEncoder(cf)
    losses = train_content_encoder(enc, np.arange(n_warm), item_vecs, epochs=args.enc_epochs)
    content_all = enc.encode()

    print("B. new items", flush=True)
    res["B"], cold_t, has_cold, r_cold, es = exp_new_items(
        model, item_vecs, content_all, catalog, d, history, args.max_users, args.seed
    )
    tfidf = tfidf_matrix(cf)
    res["B_rank"] = exp_new_item_ranking(cf, tfidf, cold_t, has_cold, r_cold, es, n_warm, n_all,
                                         args.seed)  # fmt: skip
    print(json.dumps({"B": res["B"], "B_rank": res["B_rank"]}, indent=1), flush=True)

    print("C. new visitors: fitting CF baselines", flush=True)
    recent = history[history["ts"] >= history["ts"].max() - pd.Timedelta(days=7)]
    pop = np.bincount(lookup(catalog, recent["item_id"].to_numpy()), weights=recent["weight"],
                      minlength=n_all + 1)[:n_all]  # fmt: skip
    h = history.assign(cidx=lookup(catalog, history["item_id"].to_numpy()))
    coview = M.coview_matrix(
        h["visitor_id"].astype(np.int64).to_numpy() * 10_000 + h["session_id"].to_numpy(),
        h["cidx"].to_numpy(), n_all,
    )  # fmt: skip
    mat, _ = interaction_matrix(history, item_map, n_warm)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1, "blas"):  # implicit is much slower with threaded BLAS
        als = ALSRetriever(factors=64, iterations=args.als_iters).fit(mat)
    indexes = {
        "warm": ItemIndex(item_vecs),
        "all": ItemIndex(np.vstack([item_vecs, content_all[n_warm:]])),
    }
    res["C"] = exp_new_visitors(model, als, coview, tfidf, content_all, indexes, pop, cf, catalog,
                                d, history, n_warm, args.max_users, args.seed)  # fmt: skip
    res["meta"] = {"catalogue items": n_all, "warm items": n_warm,
                   "encoder loss by epoch": [round(x, 3) for x in losses],
                   "runtime_s": round(time.perf_counter() - t0, 1)}  # fmt: skip
    return res


def _table(rows: dict, cols: list[str]) -> list[str]:
    L = ["| method | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, m in rows.items():
        if isinstance(m, dict):
            L.append(f"| {name} | " + " | ".join(
                f"{m[c]:.4f}" if c in m else "" for c in cols) + " |")  # fmt: skip
    return L


def render(res: dict) -> str:
    A, B, Br, C = res["A"], res["B"], res["B_rank"], res["C"]
    L = ["# Phase 4: cold start, content vs collaborative filtering", ""]
    L += ["## A. Same items, with and without click history", "",
          f"{A['held-out items']:,} warm items (10%) were hidden from the content encoder's "
          f"training. For {A['visitors']:,} returning visitors whose test clicks include them, "
          "each method retrieves over the full catalogue with those items represented by:",
          ""]  # fmt: skip
    L += _table({k: v for k, v in A.items() if isinstance(v, dict) and "recall@20" in v},
                ["recall@20", "recall@100", "ndcg@20"])  # fmt: skip
    cos = A["held-out cosine to CF vector"]
    L += ["", "Mean cosine between a held-out item's content vector and its CF vector: "
          + ", ".join(f"{k} {v:.3f}" for k, v in cos.items()) + ".", ""]  # fmt: skip
    L += ["## B. Genuinely new items", "",
          f"{B['new items in catalogue']:,} items are not in the trained catalogue. Returning "
          f"visitors: {B['visitors']:,}, of whom {B['visitors with a new-item target']:,} "
          "clicked at least one new item in the test window.", ""]  # fmt: skip
    L += _table({k: v for k, v in B.items() if isinstance(v, dict)},
                ["recall@20 (all targets)", "recall@20 (new-item targets)",
                 "share of slots on new items"])  # fmt: skip
    L += ["", f"Ranking only among new items, for the {Br['visitors']:,} visitors who clicked "
          "one:", ""]  # fmt: skip
    L += _table({k: v for k, v in Br.items() if isinstance(v, dict)},
                ["recall@10", "recall@20", "ndcg@20"])  # fmt: skip
    for key, title in (("1 click", "one click"), ("3 click", "three clicks")):
        c = C[key]
        L += ["", f"## C. New visitors after {title}", "",
              f"{c['visitors']:,} visitors with no history before the test window; predict "
              "the distinct items they click afterwards (context items excluded).", ""]  # fmt: skip
        cols = ["recall@10", "recall@20", "ndcg@20", "coverage@20"]
        if key == "1 click":
            cols += ["recall@20 (first click warm)", "recall@20 (first click new item)"]
        L += _table({k: v for k, v in c.items() if isinstance(v, dict) and k != "_timings_s"},
                    cols)  # fmt: skip
        if key == "1 click":
            L += ["", f"{c['share with a new item clicked first']:.1%} of these visitors' first "
                  "click is on a new item."]  # fmt: skip
    L += ["", "## Notes", "",
          "- Item metadata is RetailRocket's latest property snapshot. Category and price are",
          "  known when an item is listed, but the snapshot can include later edits, a mild",
          "  optimism for content methods. Property values are hashed: opaque tokens.",
          "- CF models (popularity, co-view, ALS, two-tower) are fitted on events before the",
          "  test window; the content encoder is trained on warm items only.",
          "- Gaps in a list are filled with popular items, so every method shows 20 items.",
          "- Missing targets count as misses (no target is dropped for being cold).",
          "- The router hybrid was added after seeing the first-click breakdown (content wins",
          "  when the first click is a new item, CF wins otherwise), so its gain is measured",
          "  on the same data that suggested it.",
          "- The encoder objective was chosen on held-out items' cosine to their CF vectors,",
          "  which uses no test-window clicks."]  # fmt: skip
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, default=None)
    p.add_argument("--retrieval", type=Path, default=ROOT / "artifacts" / "retrieval")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "coldstart")
    p.add_argument("--report", type=Path, default=ROOT / "reports" / "coldstart.md")
    p.add_argument("--max-users", type=int, default=30_000)
    p.add_argument("--enc-epochs", type=int, default=8)
    p.add_argument("--als-iters", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    res = run(a.processed or processed_dir(), a.retrieval, a)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "coldstart_results.json").write_text(json.dumps(res, indent=1, default=float))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(render(res))
    print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
