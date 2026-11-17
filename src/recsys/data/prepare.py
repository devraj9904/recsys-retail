"""Turn raw RetailRocket files into model-ready Parquet tables.

Steps
    1. clean       drop duplicate rows and bot-like visitors
    2. sessionize  split each visitor's stream on gaps > ``session_gap_minutes``
    3. split       global temporal split: train | tune | test (last N days)
    4. items       latest metadata per item: category path, availability,
                   price-like numeric property, hashed description tokens
    5. index       contiguous ids for warm users/items (seen in train)

The split is by wall-clock time, not leave-last-out per user, so the test
period contains genuinely new visitors and new items: exactly the cold-start
population the later phases have to serve. ``tune`` holds the ranker's
labels, so the ranker never learns from interactions the retrievers saw.

Outputs under ``processed_dir()``:
    events.parquet       all cleaned events with session_id, split, weight
    train/tune/test.parquet   the same rows, one file per split
    items.parquet        item metadata (every item that appears anywhere)
    item_map.parquet     warm item_id -> item_idx
    user_map.parquet     train visitor_id -> user_idx
    popularity.parquet   warm items by weighted train popularity
    category_tree.parquet
    stats.json           dataset summary (also rendered to reports/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import EVENT_WEIGHTS, PrepareConfig, processed_dir, raw_dir
from recsys.data.loaders import RawData, load_retailrocket

PRICE_PROPERTY = "790"  # numeric property widely used as a price proxy
MAX_TOKENS = 64
META_PROPERTIES = {"categoryid", "available", PRICE_PROPERTY}


# --------------------------------------------------------------------- events
def clean_events(events: pd.DataFrame, cfg: PrepareConfig) -> pd.DataFrame:
    df = events.drop_duplicates().sort_values(["visitor_id", "ts"], kind="stable")

    # Collapse bursts of the same (visitor, item, event) a page reload produces.
    key = ["visitor_id", "item_id", "event"]
    gap = df.groupby(key, observed=True)["ts"].diff().dt.total_seconds()
    df = df[gap.isna() | (gap > cfg.dedupe_seconds)]

    per_visitor = df.groupby("visitor_id").size()
    per_day = df.groupby(["visitor_id", df["ts"].dt.floor("D")]).size()
    max_day = per_day.groupby(level=0).max()
    bots = per_visitor.index[
        (per_visitor > cfg.max_events_per_visitor)
        | (max_day.reindex(per_visitor.index) > cfg.max_events_per_day)
    ]
    df = df[~df["visitor_id"].isin(bots)]
    return df.reset_index(drop=True)


def sessionize(events: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    df = events.sort_values(["visitor_id", "ts"], kind="stable").reset_index(drop=True)
    new_visitor = df["visitor_id"].ne(df["visitor_id"].shift())
    long_gap = df["ts"].diff() > pd.Timedelta(minutes=gap_minutes)
    df["session_id"] = (new_visitor | long_gap).cumsum().astype("int64") - 1
    return df


def temporal_split(events: pd.DataFrame, cfg: PrepareConfig) -> pd.DataFrame:
    end = events["ts"].max()
    test_start = (end - pd.Timedelta(days=cfg.test_days)).floor("D")
    tune_start = test_start - pd.Timedelta(days=cfg.tune_days)
    if tune_start <= events["ts"].min():
        raise ValueError("log is shorter than tune_days + test_days")
    split = np.where(
        events["ts"] >= test_start, "test", np.where(events["ts"] >= tune_start, "tune", "train")
    )
    out = events.copy()
    out["split"] = pd.Categorical(split, categories=["train", "tune", "test"])
    return out


# ---------------------------------------------------------------------- items
def _parse_numeric(value: str | float) -> float:
    if not isinstance(value, str):
        return np.nan
    first = value.split(" ", 1)[0]
    if first.startswith("n"):
        try:
            return float(first[1:])
        except ValueError:
            return np.nan
    return np.nan


def category_paths(tree: pd.DataFrame, max_depth: int = 10) -> dict[int, list[int]]:
    """Map each category to its root-first ancestor path (cycle-safe)."""
    parent = dict(zip(tree["category_id"], tree["parent_id"], strict=True))
    paths: dict[int, list[int]] = {}
    for cat in parent:
        path = [int(cat)]
        node = parent.get(cat)
        while node is not None and not pd.isna(node) and len(path) < max_depth:
            if int(node) in path:
                break
            path.append(int(node))
            node = parent.get(int(node))
        paths[int(cat)] = path[::-1]
    return paths


def build_items(raw: RawData, item_ids: pd.Index) -> pd.DataFrame:
    """Latest value of each property per item, flattened into model features.

    Properties are weekly snapshots; taking the latest one gives the catalogue
    state at the end of the log. This is a mild look-ahead for metadata only,
    never for behaviour, and matches what a live catalogue would serve.
    """
    props = raw.item_properties
    props = props[props["item_id"].isin(item_ids)]
    latest = props.sort_values("ts").drop_duplicates(["item_id", "property"], keep="last")
    prop = latest["property"].astype(str)

    def single(name: str) -> pd.Series:
        rows = latest[prop == name]
        return rows.set_index("item_id")["value"]

    items = pd.DataFrame(index=pd.Index(item_ids.unique(), name="item_id"))
    items["category_id"] = pd.to_numeric(single("categoryid"), errors="coerce").astype("Int64")
    items["available"] = pd.to_numeric(single("available"), errors="coerce").astype("Int8")
    items["price"] = single(PRICE_PROPERTY).map(_parse_numeric).astype("float64")

    other = latest[~prop.isin(META_PROPERTIES)]
    tokens = other["property"].astype(str) + ":" + other["value"].fillna("").str.split(" ").str[0]
    hashed = other["value"].fillna("").str.split(" ").str[0]
    tokens = tokens[~hashed.str.startswith("n")]  # keep categorical/hashed values only
    items["tokens"] = (
        tokens.groupby(other.loc[tokens.index, "item_id"])
        .agg(lambda s: sorted(s)[:MAX_TOKENS])
        .reindex(items.index)
    )
    items["tokens"] = items["tokens"].apply(lambda t: t if isinstance(t, list) else [])

    paths = category_paths(raw.category_tree)
    items["category_path"] = items["category_id"].map(
        lambda c: paths.get(int(c), [int(c)]) if not pd.isna(c) else []
    )
    items["top_category"] = items["category_path"].map(lambda p: p[0] if p else pd.NA)
    items["top_category"] = items["top_category"].astype("Int64")
    return items.reset_index()


# ---------------------------------------------------------------------- index
def build_maps(train: pd.DataFrame, cfg: PrepareConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = train["item_id"].value_counts()
    warm_items = np.sort(counts.index[counts >= cfg.min_item_events].to_numpy())
    item_map = pd.DataFrame({"item_id": warm_items, "item_idx": np.arange(len(warm_items))})
    users = np.sort(train["visitor_id"].unique())
    user_map = pd.DataFrame({"visitor_id": users, "user_idx": np.arange(len(users))})
    return user_map, item_map


def popularity(train: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    w = train.groupby("item_id")["weight"].sum().rename("score")
    pop = item_map.merge(w, left_on="item_id", right_index=True, how="left")
    pop["score"] = pop["score"].fillna(0)
    return pop.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def summarize(events: pd.DataFrame, items: pd.DataFrame, user_map, item_map) -> dict:
    train_users = set(user_map["visitor_id"])
    warm_items = set(item_map["item_id"])
    test = events[events["split"] == "test"]
    by_split = {}
    for name, part in events.groupby("split", observed=True):
        by_split[str(name)] = {
            "events": int(len(part)),
            "visitors": int(part["visitor_id"].nunique()),
            "items": int(part["item_id"].nunique()),
            "sessions": int(part["session_id"].nunique()),
            "start": str(part["ts"].min()),
            "end": str(part["ts"].max()),
            **{f"n_{e}": int(n) for e, n in part["event"].value_counts().items()},
        }
    events_per_visitor = events.groupby("visitor_id").size()
    return {
        "events": int(len(events)),
        "visitors": int(events["visitor_id"].nunique()),
        "items_with_events": int(events["item_id"].nunique()),
        "items_with_metadata": int(items["category_id"].notna().sum()),
        "sessions": int(events["session_id"].nunique()),
        "event_mix": {str(k): int(v) for k, v in events["event"].value_counts().items()},
        "single_event_visitor_share": round(float((events_per_visitor == 1).mean()), 4),
        "warm_users": len(train_users),
        "warm_items": len(warm_items),
        "test_new_visitor_share": round(float((~test["visitor_id"].isin(train_users)).mean()), 4)
        if len(test)
        else 0.0,
        "test_cold_item_share": round(float((~test["item_id"].isin(warm_items)).mean()), 4)
        if len(test)
        else 0.0,
        "items_with_price": int(items["price"].notna().sum()),
        "splits": by_split,
    }


def prepare(raw: RawData, cfg: PrepareConfig | None = None) -> dict[str, object]:
    cfg = cfg or PrepareConfig()
    ev = clean_events(raw.events, cfg)
    ev = sessionize(ev, cfg.session_gap_minutes)
    ev = temporal_split(ev, cfg)
    ev["weight"] = ev["event"].astype(str).map(EVENT_WEIGHTS).astype("int8")

    train = ev[ev["split"] == "train"]
    user_map, item_map = build_maps(train, cfg)
    items = build_items(raw, pd.Index(ev["item_id"].unique()))
    pop = popularity(train, item_map)
    return {
        "events": ev,
        "train": train.reset_index(drop=True),
        "tune": ev[ev["split"] == "tune"].reset_index(drop=True),
        "test": ev[ev["split"] == "test"].reset_index(drop=True),
        "items": items,
        "user_map": user_map,
        "item_map": item_map,
        "popularity": pop,
        "category_tree": raw.category_tree,
        "stats": summarize(ev, items, user_map, item_map),
    }


def write(outputs: dict[str, object], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in outputs.items():
        if name == "stats":
            (out / "stats.json").write_text(json.dumps(obj, indent=2))
        else:
            obj.to_parquet(out / f"{name}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare RetailRocket for training")
    ap.add_argument("--raw", type=Path, default=raw_dir())
    ap.add_argument("--out", type=Path, default=processed_dir())
    args = ap.parse_args()
    outputs = prepare(load_retailrocket(args.raw))
    write(outputs, args.out)
    print(json.dumps(outputs["stats"], indent=2))


if __name__ == "__main__":
    main()
