"""Implicit-feedback ALS baseline (Hu, Koren & Volinsky 2008) via ``implicit``.

Confidence = 1 + alpha * (summed event weights) per (user, item).
At evaluation time user factors are re-solved from each visitor's latest
history (``recalculate_user``), so visitors first seen in the tune window get
a fair shot, exactly like the two-tower user encoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def interaction_matrix(
    events: pd.DataFrame, item_map: pd.DataFrame, n_items: int
) -> tuple[sp.csr_matrix, np.ndarray]:
    ev = events.merge(item_map, on="item_id")
    agg = ev.groupby(["visitor_id", "item_idx"], as_index=False)["weight"].sum()
    users, rows = np.unique(agg["visitor_id"].to_numpy(), return_inverse=True)
    mat = sp.csr_matrix(
        (agg["weight"].to_numpy(np.float32), (rows, agg["item_idx"].to_numpy())),
        shape=(len(users), n_items),
    )
    return mat, users


class ALSRetriever:
    def __init__(
        self,
        factors: int = 64,
        iterations: int = 15,
        regularization: float = 0.05,
        alpha: float = 10.0,
        seed: int = 0,
    ):
        from implicit.als import AlternatingLeastSquares

        self.alpha = alpha
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=seed,
            calculate_training_loss=False,
        )

    def fit(self, user_items: sp.csr_matrix) -> ALSRetriever:
        self.model.fit((user_items * self.alpha).tocsr(), show_progress=False)
        return self

    def recommend(self, user_items: sp.csr_matrix, k: int) -> np.ndarray:
        ids = np.arange(user_items.shape[0])
        items, _ = self.model.recommend(
            ids,
            (user_items * self.alpha).tocsr(),
            N=k,
            recalculate_user=True,
            filter_already_liked_items=False,
        )
        return np.asarray(items)

    @property
    def item_factors(self) -> np.ndarray:
        f = self.model.item_factors
        return f.to_numpy() if hasattr(f, "to_numpy") else np.asarray(f)
