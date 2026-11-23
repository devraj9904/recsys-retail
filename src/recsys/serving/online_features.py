"""The 24 ranking features for one visitor, from the online store.

Mirrors ``ranking.features.build_features`` row for row, but reads the
visitor's events from Redis (the last ``recent_len`` events) and item
statistics from the serving bundle, with numpy instead of pandas joins so a
request costs well under a millisecond here. ``tests/test_serving.py``
checks that both paths give the same numbers on the same events.
"""

from __future__ import annotations

import numpy as np

from recsys.ranking.features import FEATURES
from recsys.serving.bundle import Bundle

_COL = {f: i for i, f in enumerate(FEATURES)}


def visitor_features(
    bundle: Bundle,
    cand: np.ndarray,
    retrieval: dict[str, np.ndarray],
    ev_items: np.ndarray,
    ev_weights: np.ndarray,
    ev_ts: np.ndarray,
    clock: float,
) -> np.ndarray:
    """(n_candidates, 24) float32 matrix in ``FEATURES`` order.

    ``ev_*`` are the visitor's events on warm items (item_idx, weight 1..3,
    unix seconds) strictly before ``clock``; ``retrieval`` holds the seven
    retrieval-signal columns aligned with ``cand``.
    """
    n = len(cand)
    X = np.full((n, len(FEATURES)), np.nan, dtype=np.float64)
    for name in ("tt_score", "tt_rank", "hist_pos", "trend_rank", "from_tt", "from_hist",
                 "from_trend"):  # fmt: skip
        X[:, _COL[name]] = retrieval[name]

    keep = ev_ts < clock
    it, w, ts = ev_items[keep], ev_weights[keep], ev_ts[keep]

    # user x item
    ui_n = np.zeros(n)
    ui_max = np.zeros(n)
    ui_last = np.full(n, np.nan)
    if len(it):
        pos = {int(c): j for j, c in enumerate(cand)}
        for i, wt, t in zip(it, w, ts, strict=True):
            j = pos.get(int(i))
            if j is None:
                continue
            ui_n[j] += 1
            ui_max[j] = max(ui_max[j], wt)
            ui_last[j] = t if np.isnan(ui_last[j]) else max(ui_last[j], t)
    X[:, _COL["ui_events"]] = ui_n
    X[:, _COL["ui_max_weight"]] = ui_max
    X[:, _COL["ui_hours_since"]] = np.where(np.isnan(ui_last), 1e5, (clock - ui_last) / 3600)

    # item
    for name in ("item_pop", "item_pop_7d", "item_views", "item_cart_rate", "item_buy_rate"):
        X[:, _COL[name]] = bundle.stats[name][cand]

    # user activity (NaN for a visitor with no warm events, as offline)
    f = bundle.feats
    if len(it):
        pb = f.price_bucket[it]
        u_price = pb[pb > 0].mean() if (pb > 0).any() else np.nan
        X[:, _COL["u_events"]] = len(it)
        X[:, _COL["u_items"]] = len(np.unique(it))
        X[:, _COL["u_buys"]] = float((w == 3).sum())
        X[:, _COL["u_hours_since"]] = (clock - ts.max()) / 3600
    else:
        u_price = np.nan
        X[:, _COL["u_buys"]] = 0.0

    # content and affinity
    c_leaf, c_top = f.leaf[cand], f.top[cand]
    denom = max(len(it), 1)
    for level, arr, cand_lv in (("leaf", f.leaf, c_leaf), ("top", f.top, c_top)):
        counts = np.bincount(arr[it], minlength=int(arr.max()) + 1) if len(it) else None
        share = counts[cand_lv] / denom if counts is not None else np.zeros(n)
        X[:, _COL[f"{level}_share"]] = share
    pbc = f.price_bucket[cand]
    X[:, _COL["price_bucket"]] = pbc
    X[:, _COL["price_gap"]] = np.where(pbc > 0, pbc - u_price, np.nan)
    X[:, _COL["available"]] = f.available[cand]
    return X
