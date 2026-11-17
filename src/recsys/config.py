"""Central configuration: paths and data-preparation knobs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("RECSYS_DATA", ROOT / "data"))


def raw_dir(dataset: str = "retailrocket") -> Path:
    return data_dir() / dataset / "raw"


def processed_dir(dataset: str = "retailrocket") -> Path:
    return data_dir() / dataset / "processed"


# Graded relevance per event type: used as ALS confidence weights and as
# LambdaRank labels later on (view < addtocart < transaction).
EVENT_WEIGHTS: dict[str, int] = {"view": 1, "addtocart": 2, "transaction": 3}


@dataclass(frozen=True)
class PrepareConfig:
    # A new session starts after this much inactivity.
    session_gap_minutes: int = 30
    # Visitors above either threshold are treated as bots/crawlers and dropped.
    max_events_per_visitor: int = 1000
    max_events_per_day: int = 300
    # Collapse repeated identical (visitor, item, event) rows within this window.
    dedupe_seconds: int = 1
    # Temporal split, as day counts taken from the end of the log:
    # [ ... train ... | tune_days | test_days ]
    test_days: int = 14
    tune_days: int = 7
    # Items need this many train events to enter the warm (CF) catalogue;
    # rarer items are kept as cold items with metadata only.
    min_item_events: int = 2
