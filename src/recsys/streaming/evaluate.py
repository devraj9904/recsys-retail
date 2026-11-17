"""Does a live user vector beat a stale one? Offline next-event benchmark.

For every test event of a visitor with pre-test history we ask both states
to predict the item: the *stale* vector, frozen at the start of the test
window (what a nightly batch job would serve), and the *live* vector, built
from everything up to the event before (what the streaming consumer keeps
in Redis). Hit@K and MRR@K are computed per event, split into all next
events and "new" next items the visitor had not touched in their last 30
steps. Popularity at the cutoff is the floor. Consecutive repeats are
collapsed (as in training), so the target is never the item just viewed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.index.ann import ItemIndex
from recsys.models.two_tower import TwoTower
from recsys.retrieval.data import build_streams, history_windows


def next_event_eval(
    model: TwoTower,
    index: ItemIndex,
    history: pd.DataFrame,
    test: pd.DataFrame,
    item_map: pd.DataFrame,
    popular: np.ndarray,
    k: int = 20,
    length: int = 30,
    max_events: int | None = 50_000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    cutoff = test["ts"].min()
    allev = pd.concat([history, test], ignore_index=True)
    s = build_streams(allev, item_map)
    # Timestamps survive build_streams only implicitly; recover is_test per row.
    ev = allev.merge(item_map, on="item_id", how="inner").sort_values(
        ["visitor_id", "ts"], kind="stable"
    )
    v, it = ev["visitor_id"].to_numpy(), ev["item_idx"].to_numpy()
    new_run = np.ones(len(v), dtype=bool)
    new_run[1:] = (v[1:] != v[:-1]) | (it[1:] != it[:-1])
    is_test = (ev["ts"] >= cutoff).to_numpy()[new_run]

    rows = np.arange(len(s.item))
    first_test = np.where(is_test, rows, len(rows))
    # Per visitor: index of first test row (= end of the stale window).
    vis_first = pd.Series(first_test).groupby(s.start).transform("min").to_numpy()
    eligible = is_test & (vis_first > s.start)  # visitor has pre-test history
    pos = np.flatnonzero(eligible)
    if max_events and len(pos) > max_events:
        pos = np.sort(np.random.default_rng(seed).choice(pos, max_events, replace=False))

    live_h, live_w = history_windows(s, pos, length)
    stale_h, stale_w = history_windows(s, vis_first[pos], length)
    target = s.item[pos]
    novel = ~(live_h == target[:, None]).any(axis=1)

    def topk(h, w):
        out = []
        for i in range(0, len(h), 8192):
            u = model.user_matrix(h[i : i + 8192], w[i : i + 8192])
            out.append(index.search(u, k)[1])
        return np.concatenate(out)

    lists = {
        "popularity": np.broadcast_to(popular[:k], (len(pos), k)),
        "stale vector (batch, frozen at cutoff)": topk(stale_h, stale_w),
        "live vector (streaming)": topk(live_h, live_w),
    }
    out = {}
    for name, recs in lists.items():
        hit = recs == target[:, None]
        rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
        rr = np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0.0)
        out[name] = {
            f"hit@{k}": float(hit.any(1).mean()),
            f"mrr@{k}": float(rr.mean()),
            f"hit@{k} new item": float(hit.any(1)[novel].mean()),
            f"mrr@{k} new item": float(rr[novel].mean()),
        }
    out["_counts"] = {
        "events": int(len(pos)),
        "new item events": int(novel.sum()),
        "visitors": int(len(np.unique(s.visitor[pos]))),
    }
    return out
