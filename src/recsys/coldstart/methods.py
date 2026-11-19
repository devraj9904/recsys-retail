"""Recommenders for a visitor known only by a few clicks.

Every method takes ``context``: per visitor, the catalogue ids (cidx) of the
items clicked so far, oldest first, and returns (n_visitors, k) cidx arrays
with the context items removed (padding -2).

Content-based (metadata only): category popularity, TF-IDF item kNN.
Collaborative (behaviour only): popularity, co-view, ALS fold-in.
Hybrid: two-tower user tower over warm items, content encoder for cold ones.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

PADREC = -2


def _finish(scores_or_lists, context, k):
    out = np.full((len(context), k), PADREC, dtype=np.int64)
    for u, (cand, ctx) in enumerate(zip(scores_or_lists, context, strict=True)):
        cand = np.asarray(cand)
        cand = cand[~np.isin(cand, ctx)][:k]
        out[u, : len(cand)] = cand
    return out


def _topk_rows(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
    return np.take_along_axis(top, order, 1)


def popular(pop: np.ndarray, context, k: int) -> np.ndarray:
    ranked = np.argsort(-pop)[: k + 50]
    return _finish([ranked] * len(context), context, k)


def category_popular(pop: np.ndarray, leaf: np.ndarray, context, k: int) -> np.ndarray:
    """Most popular items in the leaf categories of the clicked items, then global."""
    order = np.argsort(-pop, kind="stable")
    order = order[pop[order] > 0]
    by_leaf: dict[int, np.ndarray] = {}
    lf = leaf[order]
    srt = np.argsort(lf, kind="stable")
    cuts = np.flatnonzero(np.diff(lf[srt])) + 1
    for grp in np.split(srt, cuts):
        if len(grp):
            by_leaf[int(lf[grp[0]])] = order[grp[: k + 20]]
    fallback = order[: k + 50]
    lists = []
    for ctx in context:
        leaves = list(dict.fromkeys(int(leaf[i]) for i in ctx[::-1] if leaf[i] > 0))
        parts = [by_leaf.get(c, np.empty(0, np.int64)) for c in leaves] + [fallback]
        lists.append(np.array(list(dict.fromkeys(np.concatenate(parts).tolist()))))
    return _finish(lists, context, k)


def _context_matrix(context, n_items: int, weights=None) -> sp.csr_matrix:
    rows = np.repeat(np.arange(len(context)), [len(c) for c in context])
    cols = np.concatenate(context) if len(context) else np.empty(0, np.int64)
    vals = np.ones(len(cols), np.float32) if weights is None else weights
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(context), n_items))


def item_knn(sim_rows: sp.csr_matrix, context, k: int, batch: int = 512) -> np.ndarray:
    """Sum of item-item similarities to the clicked items (rows = items).

    ``sim_rows`` is either a feature matrix X (then similarity is X X^T, the
    TF-IDF content kNN) or an item-item matrix C (co-view), see callers.
    """
    out = []
    for i in range(0, len(context), batch):
        ctx = context[i : i + batch]
        q = _context_matrix(ctx, sim_rows.shape[0]) @ sim_rows
        s = q.toarray() if sp.issparse(q) else q
        out.append(_finish(_topk_rows(s, k + 10), ctx, k))
    return np.concatenate(out) if out else np.empty((0, k), np.int64)


def content_knn(
    x: sp.csr_matrix, context, k: int, candidates: np.ndarray | None = None, batch: int = 256
) -> np.ndarray:
    """Items whose TF-IDF content is closest to the clicked items (optionally a subset)."""
    cand = np.arange(x.shape[0]) if candidates is None else np.asarray(candidates)
    xt = x[cand].T.tocsr()
    out = []
    for i in range(0, len(context), batch):
        ctx = context[i : i + batch]
        q = _context_matrix(ctx, x.shape[0]) @ x  # (b, features)
        s = (q @ xt).toarray()
        out.append(_finish(cand[_topk_rows(s, k + 10)], ctx, k))
    return np.concatenate(out) if out else np.empty((0, k), np.int64)


def coview_matrix(sessions: np.ndarray, items: np.ndarray, n_items: int) -> sp.csr_matrix:
    """Item-item co-occurrence within sessions, cosine-normalised, diagonal removed."""
    _, s = np.unique(sessions, return_inverse=True)
    m = sp.csr_matrix((np.ones(len(items), np.float32), (s, items)), shape=(s.max() + 1, n_items))
    m.data[:] = 1.0  # binary per session
    m.sum_duplicates()
    m.data[:] = 1.0
    c = (m.T @ m).tocsr()
    c = (c - sp.diags(c.diagonal())).tocsr()
    c.eliminate_zeros()
    n = np.sqrt(np.asarray(m.sum(0)).ravel())
    return (sp.diags(1 / np.maximum(n, 1)) @ c @ sp.diags(1 / np.maximum(n, 1))).tocsr()


def als_foldin(als, n_warm: int, context, k: int) -> np.ndarray:
    """Re-solve ALS user factors from the clicked warm items (cold ones unusable)."""
    warm_ctx = [c[c < n_warm] for c in context]
    has = np.array([len(c) > 0 for c in warm_ctx])
    lists = [np.empty(0, np.int64)] * len(context)
    if has.any():
        idx = np.flatnonzero(has)
        mat = _context_matrix([warm_ctx[i] for i in idx], n_warm)
        recs = als.recommend(mat, k + 10)
        for j, u in enumerate(idx):
            lists[u] = recs[j]
    return _finish(lists, context, k)


def vector_search(index, user_vecs: np.ndarray, valid: np.ndarray, context, k: int) -> np.ndarray:
    lists = [np.empty(0, np.int64)] * len(context)
    if valid.any():
        _, idx = index.search(user_vecs[valid], k + 10)
        for j, u in enumerate(np.flatnonzero(valid)):
            lists[u] = idx[j]
    return _finish(lists, context, k)


def route(use_primary: np.ndarray, primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Per visitor, take ``primary`` where ``use_primary`` else ``secondary``."""
    return np.where(use_primary[:, None], primary, secondary)


def fill(primary: np.ndarray, backup: np.ndarray) -> np.ndarray:
    """Fill padding in ``primary`` from ``backup`` without duplicates."""
    out = primary.copy()
    for u in range(len(out)):
        have = out[u][out[u] >= 0]
        if len(have) == out.shape[1]:
            continue
        extra = backup[u][~np.isin(backup[u], have) & (backup[u] >= 0)]
        merged = np.concatenate([have, extra])[: out.shape[1]]
        out[u, : len(merged)] = merged
    return out
