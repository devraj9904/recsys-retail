"""Popularity baselines: everyone gets the same ranked list."""

from __future__ import annotations

import numpy as np
import pandas as pd


def popularity_scores(events: pd.DataFrame, item_map: pd.DataFrame) -> np.ndarray:
    """Event-weighted counts per warm item_idx (views 1, carts 2, purchases 3)."""
    w = events.merge(item_map, on="item_id").groupby("item_idx")["weight"].sum()
    scores = np.zeros(len(item_map), dtype=np.float64)
    scores[w.index.to_numpy()] = w.to_numpy()
    return scores


def recommend_popular(scores: np.ndarray, n_users: int, k: int) -> np.ndarray:
    top = np.argsort(-scores, kind="stable")[:k]
    return np.broadcast_to(top, (n_users, k))
