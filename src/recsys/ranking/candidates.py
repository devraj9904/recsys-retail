"""Candidate generation: union of three retrieval sources per visitor.

* two-tower + FAISS top-N (discovery)
* the visitor's own recent items (re-engagement; Phase 1 showed this is
  where most purchases come from)
* trending items from the week before the cutoff (fallback)

Every candidate gets the two-tower score, whatever source produced it, so
the ranker can compare them on one scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.index.ann import ItemIndex
from recsys.retrieval.data import EvalSet


def generate_candidates(
    es: EvalSet,
    user_vecs: np.ndarray,
    item_vecs: np.ndarray,
    index: ItemIndex,
    trending: np.ndarray,
    n_tt: int = 100,
    n_trend: int = 20,
) -> pd.DataFrame:
    n_users = len(es.visitors)
    _, tt = index.search(user_vecs, n_tt)
    frames = [
        pd.DataFrame(
            {
                "u": np.repeat(np.arange(n_users), n_tt),
                "item_idx": tt.ravel(),
                "tt_rank": np.tile(np.arange(1, n_tt + 1), n_users),
            }
        )
    ]
    # Own history, newest first; hist_pos 1 = most recent item.
    rows_u, rows_i, rows_p = [], [], []
    for u, h in enumerate(es.hist_items):
        seen = list(dict.fromkeys(int(x) for x in h[::-1] if x >= 0))
        rows_u += [u] * len(seen)
        rows_i += seen
        rows_p += list(range(1, len(seen) + 1))
    frames.append(pd.DataFrame({"u": rows_u, "item_idx": rows_i, "hist_pos": rows_p}))
    top = trending[:n_trend]
    frames.append(
        pd.DataFrame(
            {
                "u": np.repeat(np.arange(n_users), len(top)),
                "item_idx": np.tile(top, n_users),
                "trend_rank": np.tile(np.arange(1, len(top) + 1), n_users),
            }
        )
    )
    c = pd.concat(frames, ignore_index=True)
    c = c[c["item_idx"] >= 0]
    c = c.groupby(["u", "item_idx"], as_index=False).agg(
        tt_rank=("tt_rank", "min"), hist_pos=("hist_pos", "min"), trend_rank=("trend_rank", "min")
    )
    c["from_tt"] = c["tt_rank"].notna().astype(np.int8)
    c["from_hist"] = c["hist_pos"].notna().astype(np.int8)
    c["from_trend"] = c["trend_rank"].notna().astype(np.int8)
    c["tt_rank"] = c["tt_rank"].fillna(n_tt + 1)
    c["hist_pos"] = c["hist_pos"].fillna(99)
    c["trend_rank"] = c["trend_rank"].fillna(n_trend + 1)
    u, i = c["u"].to_numpy(), c["item_idx"].to_numpy()
    c["tt_score"] = np.einsum("ij,ij->i", user_vecs[u], item_vecs[i])
    c["visitor_id"] = es.visitors[u]
    return c.reset_index(drop=True)
