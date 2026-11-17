"""Top-K retrieval metrics with binary relevance.

``recs`` is an (n_users, K) array of item indices, best first. ``targets`` is a
list of per-user arrays of relevant item indices. A target of -1 marks an item
outside the model's catalogue (a cold item): it counts in the denominator but
can never be hit, so CF models are not flattered by silently dropping it.
"""

from __future__ import annotations

import numpy as np


def _hits(recs: np.ndarray, targets: list[np.ndarray]) -> np.ndarray:
    hits = np.zeros(recs.shape, dtype=bool)
    for u, t in enumerate(targets):
        if len(t):
            hits[u] = np.isin(recs[u], t[t >= 0])
    return hits


def evaluate(
    recs: np.ndarray,
    targets: list[np.ndarray],
    ks: tuple[int, ...] = (10, 20, 50, 100),
    n_items: int | None = None,
) -> dict[str, float]:
    if len(recs) != len(targets):
        raise ValueError("recs and targets must have one row per user")
    hits = _hits(recs, targets)
    n_rel = np.array([len(t) for t in targets], dtype=float)
    valid = n_rel > 0
    discounts = 1.0 / np.log2(np.arange(2, recs.shape[1] + 2))
    out: dict[str, float] = {"users": int(valid.sum())}
    for k in ks:
        if k > recs.shape[1]:
            continue
        h = hits[valid, :k]
        out[f"recall@{k}"] = float((h.sum(1) / n_rel[valid]).mean())
        out[f"hit@{k}"] = float(h.any(1).mean())
        dcg = (h * discounts[:k]).sum(1)
        ideal = np.array([discounts[: int(min(n, k))].sum() for n in n_rel[valid]])
        out[f"ndcg@{k}"] = float((dcg / ideal).mean())
        if n_items:
            out[f"coverage@{k}"] = float(len(np.unique(recs[valid, :k])) / n_items)
    return out
