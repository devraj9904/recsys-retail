"""LightGBM rankers: LambdaRank on graded engagement, plus a purchase model.

The engagement ranker optimises NDCG directly with graded labels
(view=1 < add-to-cart=2 < purchase=3). The conversion model is a binary
classifier on add-to-cart or purchase (purchases weighted 3x). ``blend``
combines them per visitor after standardising each score within the visitor's candidate list:

    score = (1 - w) * z(engagement) + w * z(logit p_purchase)

so w in [0, 1] trades engagement relevance against purchase likelihood on
a common scale.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from recsys.ranking.features import FEATURES


def _split_by_group(df: pd.DataFrame, valid_frac: float, seed: int):
    users = df["visitor_id"].unique()
    rng = np.random.default_rng(seed)
    valid_users = set(rng.choice(users, max(1, int(len(users) * valid_frac)), replace=False))
    is_valid = df["visitor_id"].isin(valid_users).to_numpy()
    return df[~is_valid], df[is_valid]


def _groups(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("visitor_id", sort=False).size().to_numpy()


def train_lambdarank(df: pd.DataFrame, seed: int = 0, n_estimators: int = 500):
    # Groups with no positive carry no ranking signal.
    has_pos = df.groupby("visitor_id")["label"].transform("max") > 0
    df = df[has_pos].sort_values("visitor_id", kind="stable")
    tr, va = _split_by_group(df, 0.15, seed)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        label_gain=[0, 1, 3, 7],
        lambdarank_truncation_level=30,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        tr[FEATURES], tr["label"], group=_groups(tr),
        eval_set=[(va[FEATURES], va["label"])], eval_group=[_groups(va)],
        eval_at=[10, 20], callbacks=[lgb.early_stopping(50, verbose=False)],
    )  # fmt: skip
    return model


def train_purchase(df: pd.DataFrame, seed: int = 0, n_estimators: int = 500):
    """Conversion-intent model: add-to-cart or purchase, purchases weighted 3x.

    Purchases alone are too rare in one tune week (a few hundred positives
    among candidates) for a stable classifier; add-to-cart is the closest,
    10x more frequent proxy, and the 3x weight keeps purchases on top.
    """
    y = (df["label"] >= 2).astype(int)
    tr, va = _split_by_group(df.assign(y=y), 0.15, seed)
    sw = np.where(tr["label"] == 3, 3.0, 1.0)
    pos = max(int(tr["y"].sum()), 1)
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=min(20.0, (len(tr) - pos) / pos),
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        tr[FEATURES], tr["y"], sample_weight=sw, eval_set=[(va[FEATURES], va["y"])],
        eval_metric="average_precision", callbacks=[lgb.early_stopping(50, verbose=False)],
    )  # fmt: skip
    return model


def _z(x: pd.Series, groups: pd.Series) -> pd.Series:
    g = x.groupby(groups)
    return (x - g.transform("mean")) / (g.transform("std").fillna(0) + 1e-9)


def blend(engagement: pd.Series, p_buy: pd.Series, groups: pd.Series, w: float) -> pd.Series:
    p = np.clip(p_buy, 1e-6, 1 - 1e-6)
    return (1 - w) * _z(engagement, groups) + w * _z(np.log(p / (1 - p)), groups)
