"""Readers for the raw RetailRocket files.

Kaggle layout (retailrocket/ecommerce-dataset):
    events.csv                 timestamp(ms), visitorid, event, itemid, transactionid
    item_properties_part1.csv  timestamp(ms), itemid, property, value
    item_properties_part2.csv  (same columns, second half)
    category_tree.csv          categoryid, parentid

Everything downstream works on the normalised column names returned here
(``ts``, ``visitor_id``, ``item_id``, ``event``, ...), so another dataset
(e.g. Coveo SIGIR-21) only needs a loader that produces the same frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EVENT_TYPES = ("view", "addtocart", "transaction")


@dataclass
class RawData:
    events: pd.DataFrame  # ts, visitor_id, item_id, event, transaction_id
    item_properties: pd.DataFrame  # ts, item_id, property, value
    category_tree: pd.DataFrame  # category_id, parent_id


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"visitorid": "int64", "itemid": "int64", "event": "category"},
    )
    df = df.rename(
        columns={
            "visitorid": "visitor_id",
            "itemid": "item_id",
            "transactionid": "transaction_id",
        }
    )
    df["ts"] = pd.to_datetime(df.pop("timestamp"), unit="ms")
    unknown = set(df["event"].astype(str).unique()) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"unexpected event types: {sorted(unknown)}")
    df["event"] = pd.Categorical(df["event"].astype(str), categories=EVENT_TYPES)
    return df[["ts", "visitor_id", "item_id", "event", "transaction_id"]]


def load_item_properties(paths: list[Path]) -> pd.DataFrame:
    frames = [
        pd.read_csv(p, dtype={"itemid": "int64", "property": "category", "value": "string"})
        for p in paths
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"itemid": "item_id"})
    df["ts"] = pd.to_datetime(df.pop("timestamp"), unit="ms")
    df["property"] = df["property"].astype(str).astype("category")
    return df[["ts", "item_id", "property", "value"]]


def load_category_tree(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"categoryid": "int64", "parentid": "Int64"})
    return df.rename(columns={"categoryid": "category_id", "parentid": "parent_id"})


def load_retailrocket(raw: Path) -> RawData:
    prop_paths = sorted(raw.glob("item_properties_part*.csv"))
    if not prop_paths:
        raise FileNotFoundError(f"no item_properties_part*.csv under {raw}")
    return RawData(
        events=load_events(raw / "events.csv"),
        item_properties=load_item_properties(prop_paths),
        category_tree=load_category_tree(raw / "category_tree.csv"),
    )
