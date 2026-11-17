"""Ranking features for (visitor, candidate item) pairs at a cutoff time.

Everything is computed from events strictly before the cutoff, so the same
code builds training rows (cutoff = start of tune window) and evaluation
rows (cutoff = start of test window) without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.models.two_tower import ItemFeatures

FEATURES = [
    # retrieval signals
    "tt_score", "tt_rank", "hist_pos", "trend_rank", "from_tt", "from_hist", "from_trend",
    # user x item history
    "ui_events", "ui_max_weight", "ui_hours_since",
    # item popularity and funnel
    "item_pop", "item_pop_7d", "item_views", "item_cart_rate", "item_buy_rate",
    # user activity
    "u_events", "u_items", "u_buys", "u_hours_since",
    # content and affinity
    "leaf_share", "top_share", "price_bucket", "price_gap", "available",
]  # fmt: skip


def _hours(cutoff: pd.Timestamp, ts: pd.Series) -> np.ndarray:
    return ((cutoff - ts).dt.total_seconds() / 3600.0).to_numpy()


def build_features(
    cands: pd.DataFrame,
    history: pd.DataFrame,
    item_map: pd.DataFrame,
    feats: ItemFeatures,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    h = history[history["ts"] < cutoff].merge(item_map, on="item_id")
    ev = h["event"].astype(str)
    h = h.assign(
        is_view=(ev == "view").astype(np.int32),
        is_cart=(ev == "addtocart").astype(np.int32),
        is_buy=(ev == "transaction").astype(np.int32),
        leaf=feats.leaf[h["item_idx"].to_numpy()],
        top=feats.top[h["item_idx"].to_numpy()],
        price_bucket=feats.price_bucket[h["item_idx"].to_numpy()],
    )
    df = cands

    ui = h.groupby(["visitor_id", "item_idx"]).agg(
        ui_events=("weight", "size"), ui_max_weight=("weight", "max"), ui_last=("ts", "max")
    )
    ui["ui_hours_since"] = _hours(cutoff, ui.pop("ui_last"))
    df = df.merge(ui, left_on=["visitor_id", "item_idx"], right_index=True, how="left")

    it = h.groupby("item_idx").agg(
        item_pop=("weight", "sum"), item_views=("is_view", "sum"),
        item_carts=("is_cart", "sum"), item_buys=("is_buy", "sum"),
    )  # fmt: skip
    it["item_cart_rate"] = it.pop("item_carts") / (it["item_views"] + 5)
    it["item_buy_rate"] = it.pop("item_buys") / (it["item_views"] + 5)
    recent = h[h["ts"] >= cutoff - pd.Timedelta(days=7)]
    it["item_pop_7d"] = recent.groupby("item_idx")["weight"].sum()
    df = df.merge(it, left_on="item_idx", right_index=True, how="left")

    us = h.groupby("visitor_id").agg(
        u_events=("weight", "size"), u_items=("item_idx", "nunique"),
        u_buys=("is_buy", "sum"), u_last=("ts", "max"),
        u_price=("price_bucket", lambda s: s[s > 0].mean()),
    )  # fmt: skip
    us["u_hours_since"] = _hours(cutoff, us.pop("u_last"))
    df = df.merge(us, left_on="visitor_id", right_index=True, how="left")

    i = df["item_idx"].to_numpy()
    df["leaf"], df["top"] = feats.leaf[i], feats.top[i]
    df["price_bucket"], df["available"] = feats.price_bucket[i], feats.available[i]
    for level in ("leaf", "top"):
        share = h.groupby(["visitor_id", level]).size().rename(f"{level}_n").reset_index()
        df = df.merge(share, on=["visitor_id", level], how="left")
        df[f"{level}_share"] = df.pop(f"{level}_n").fillna(0) / df["u_events"].clip(lower=1)
    df["price_gap"] = np.where(df["price_bucket"] > 0, df["price_bucket"] - df["u_price"], np.nan)

    fill0 = ["ui_events", "ui_max_weight", "item_pop", "item_pop_7d", "item_views",
             "item_cart_rate", "item_buy_rate", "u_buys"]  # fmt: skip
    df[fill0] = df[fill0].fillna(0)
    df["ui_hours_since"] = df["ui_hours_since"].fillna(1e5)
    return df


def attach_labels(df: pd.DataFrame, future: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """label = strongest future event on the item (0 none, 1 view, 2 cart, 3 buy)."""
    f = future.merge(item_map, on="item_id")
    lab = f.groupby(["visitor_id", "item_idx"])["weight"].max().rename("label")
    df = df.merge(lab, left_on=["visitor_id", "item_idx"], right_index=True, how="left")
    df["label"] = df["label"].fillna(0).astype(np.int8)
    return df
