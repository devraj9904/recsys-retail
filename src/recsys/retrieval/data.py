"""Turn prepared events into retrieval training examples and an eval set.

Histories are per-visitor streams of warm items ordered by time, with
consecutive repeats of the same item collapsed (a view followed by an
add-to-cart of that item becomes one step carrying the stronger signal).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PAD = -1


@dataclass
class Streams:
    """Flat, user-contiguous event stream (sorted by visitor then time)."""

    visitor: np.ndarray  # int64
    item: np.ndarray  # item_idx, int64
    weight: np.ndarray  # event weight 1..3, int8
    start: np.ndarray  # per-row index of the visitor's first row


def build_streams(events: pd.DataFrame, item_map: pd.DataFrame) -> Streams:
    ev = events.merge(item_map, on="item_id", how="inner")
    ev = ev.sort_values(["visitor_id", "ts"], kind="stable")
    v = ev["visitor_id"].to_numpy()
    it = ev["item_idx"].to_numpy().astype(np.int64)
    w = ev["weight"].to_numpy().astype(np.int8)

    # Collapse consecutive repeats of the same item; keep the strongest weight.
    new_run = np.ones(len(v), dtype=bool)
    new_run[1:] = (v[1:] != v[:-1]) | (it[1:] != it[:-1])
    run_id = np.cumsum(new_run) - 1
    w = np.maximum.reduceat(w, np.flatnonzero(new_run)) if len(w) else w
    v, it = v[new_run], it[new_run]
    del run_id

    first = np.ones(len(v), dtype=bool)
    first[1:] = v[1:] != v[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(len(v)), 0))
    return Streams(visitor=v, item=it, weight=w.astype(np.int8), start=start)


def history_windows(s: Streams, pos: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Items/weights of the ``length`` rows before each position (PAD-filled)."""
    offs = pos[:, None] - length + np.arange(length)[None, :]
    valid = offs >= s.start[pos][:, None]
    offs = np.clip(offs, 0, None)
    items = np.where(valid, s.item[offs], PAD)
    weights = np.where(valid, s.weight[offs], 0).astype(np.float32)
    return items, weights


def training_examples(s: Streams, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(history items, history weights, target) for every step with history."""
    pos = np.flatnonzero(np.arange(len(s.item)) > s.start)
    hist, w = history_windows(s, pos, length)
    return hist, w, s.item[pos]


def last_history(s: Streams, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Most recent ``length`` steps per visitor -> (visitors, items, weights)."""
    last = np.ones(len(s.visitor), dtype=bool)
    last[:-1] = s.visitor[1:] != s.visitor[:-1]
    pos = np.flatnonzero(last)
    # Window ends *after* the last row, so shift by one.
    offs = pos[:, None] - length + 1 + np.arange(length)[None, :]
    valid = offs >= s.start[pos][:, None]
    offs = np.clip(offs, 0, None)
    items = np.where(valid, s.item[offs], PAD)
    weights = np.where(valid, s.weight[offs], 0).astype(np.float32)
    return s.visitor[pos], items, weights


@dataclass
class EvalSet:
    visitors: np.ndarray
    hist_items: np.ndarray
    hist_weights: np.ndarray
    targets: list[np.ndarray]  # item_idx, -1 for cold items
    purchase_targets: list[np.ndarray]


def build_eval_set(
    history_events: pd.DataFrame,
    test_events: pd.DataFrame,
    item_map: pd.DataFrame,
    length: int,
    max_users: int | None = None,
    seed: int = 0,
) -> EvalSet:
    """Visitors with pre-test history on warm items and at least one test event."""
    s = build_streams(history_events, item_map)
    visitors, items, weights = last_history(s, length)
    test = test_events[test_events["visitor_id"].isin(visitors)]
    test = test.merge(item_map, on="item_id", how="left")
    test["item_idx"] = test["item_idx"].fillna(-1).astype(np.int64)
    by_user = _group_unique(test["visitor_id"].to_numpy(), test["item_idx"].to_numpy())
    is_buy = (test["event"].astype(str) == "transaction").to_numpy()
    buys = _group_unique(test["visitor_id"].to_numpy()[is_buy], test["item_idx"].to_numpy()[is_buy])
    keep = np.isin(visitors, np.fromiter(by_user.keys(), dtype=np.int64, count=len(by_user)))
    visitors, items, weights = visitors[keep], items[keep], weights[keep]
    if max_users and len(visitors) > max_users:
        sel = np.sort(np.random.default_rng(seed).choice(len(visitors), max_users, replace=False))
        visitors, items, weights = visitors[sel], items[sel], weights[sel]
    empty = np.array([], dtype=np.int64)
    return EvalSet(
        visitors=visitors,
        hist_items=items,
        hist_weights=weights,
        targets=[by_user[v] for v in visitors],
        purchase_targets=[buys.get(v, empty) for v in visitors],
    )


def _group_unique(keys: np.ndarray, values: np.ndarray) -> dict[int, np.ndarray]:
    """{key: sorted unique values}, vectorised."""
    if len(keys) == 0:
        return {}
    order = np.lexsort((values, keys))
    k, v = keys[order], values[order]
    cuts = np.flatnonzero(k[1:] != k[:-1]) + 1
    return {
        int(kk[0]): np.unique(vv)
        for kk, vv in zip(np.split(k, cuts), np.split(v, cuts), strict=True)
    }


def load_processed(path: Path) -> dict[str, pd.DataFrame]:
    names = ["train", "tune", "test", "items", "item_map", "popularity"]
    return {n: pd.read_parquet(path / f"{n}.parquet") for n in names}
