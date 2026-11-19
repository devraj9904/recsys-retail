"""One index space for every item, warm or cold.

Warm items keep their Phase 1 ``item_idx`` (0..n_warm-1); every other item
with metadata or an event gets the next ids. Content models score the whole
catalogue; CF models only ever see the warm prefix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_catalog(item_map: pd.DataFrame, *extra_ids: np.ndarray) -> pd.DataFrame:
    warm = item_map.sort_values("item_idx")[["item_id", "item_idx"]]
    warm = warm.rename(columns={"item_idx": "cidx"})
    others = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in extra_ids]))
    others = np.setdiff1d(others, warm["item_id"].to_numpy())
    cold = pd.DataFrame({"item_id": others, "cidx": np.arange(len(warm), len(warm) + len(others))})
    out = pd.concat([warm, cold], ignore_index=True)
    out["warm"] = out["cidx"] < len(warm)
    return out


def lookup(catalog: pd.DataFrame, item_ids: np.ndarray) -> np.ndarray:
    """item_id -> cidx (-1 if unknown)."""
    s = pd.Series(catalog["cidx"].to_numpy(), index=catalog["item_id"].to_numpy())
    return s.reindex(np.asarray(item_ids)).fillna(-1).astype(np.int64).to_numpy()
