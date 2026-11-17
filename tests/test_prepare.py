import pandas as pd
import pytest

from recsys.config import PrepareConfig
from recsys.data.prepare import (
    _parse_numeric,
    category_paths,
    clean_events,
    prepare,
    sessionize,
    temporal_split,
    write,
)


def _events(rows):
    df = pd.DataFrame(rows, columns=["ts", "visitor_id", "item_id", "event"])
    df["ts"] = pd.to_datetime(df["ts"], format="mixed")
    df["transaction_id"] = pd.NA
    return df


def test_sessionize_splits_on_gap():
    ev = _events(
        [
            ("2015-06-01 10:00", 1, 1, "view"),
            ("2015-06-01 10:10", 1, 2, "view"),
            ("2015-06-01 11:00", 1, 3, "view"),  # 50 min gap -> new session
            ("2015-06-01 11:00", 2, 3, "view"),  # new visitor -> new session
        ]
    )
    out = sessionize(ev, gap_minutes=30)
    assert out["session_id"].tolist() == [0, 0, 1, 2]


def test_clean_drops_reload_bursts_and_bots():
    rows = [("2015-06-01 10:00:00", 1, 1, "view"), ("2015-06-01 10:00:00.5", 1, 1, "view")]
    rows += [(f"2015-06-01 10:{m:02d}", 9, m, "view") for m in range(50)]  # bot
    cfg = PrepareConfig(max_events_per_day=20)
    out = clean_events(_events(rows), cfg)
    assert len(out) == 1
    assert 9 not in set(out["visitor_id"])


def test_temporal_split_is_ordered():
    days = pd.date_range("2015-05-01", periods=40, freq="D")
    ev = _events([(d, i, i, "view") for i, d in enumerate(days)])
    out = temporal_split(ev, PrepareConfig(test_days=14, tune_days=7))
    order = {"train": 0, "tune": 1, "test": 2}
    ranks = out["split"].astype(str).map(order)
    assert ranks.is_monotonic_increasing
    assert set(out["split"].astype(str)) == {"train", "tune", "test"}


def test_temporal_split_rejects_short_log():
    ev = _events([("2015-05-01", 1, 1, "view"), ("2015-05-05", 1, 1, "view")])
    with pytest.raises(ValueError):
        temporal_split(ev, PrepareConfig())


def test_category_paths_root_first_and_cycle_safe():
    tree = pd.DataFrame({"category_id": [1, 2, 3, 7, 8], "parent_id": [None, 1, 2, 8, 7]})
    paths = category_paths(tree)
    assert paths[3] == [1, 2, 3]
    assert len(paths[7]) == 2  # 7 <-> 8 cycle terminates


def test_parse_numeric():
    assert _parse_numeric("n552.000") == 552.0
    assert pd.isna(_parse_numeric("1215254 n24.000"))
    assert pd.isna(_parse_numeric(None))


def test_prepare_end_to_end(raw, tmp_path):
    out = prepare(raw)
    train, tune, test = out["train"], out["tune"], out["test"]
    assert train["ts"].max() < tune["ts"].min() <= tune["ts"].max() < test["ts"].min()
    assert len(train) + len(tune) + len(test) == len(out["events"])

    # Sessions never span visitors.
    assert (out["events"].groupby("session_id")["visitor_id"].nunique() == 1).all()

    # Warm indices are contiguous and only cover train entities.
    im, um = out["item_map"], out["user_map"]
    assert im["item_idx"].tolist() == list(range(len(im)))
    assert set(um["visitor_id"]) <= set(train["visitor_id"])
    assert set(out["popularity"]["item_id"]) == set(im["item_id"])

    items = out["items"]
    assert items["category_id"].notna().all()
    assert items["price"].notna().all()
    assert items["category_path"].map(len).ge(1).all()
    assert items["tokens"].map(len).gt(0).all()
    assert not items["tokens"].map(lambda t: any(":n" in x for x in t)).any()

    stats = out["stats"]
    assert stats["test_cold_item_share"] > 0  # late-released items exist
    assert 0 < stats["test_new_visitor_share"] < 1

    write(out, tmp_path)
    assert (tmp_path / "stats.json").exists()
    assert len(pd.read_parquet(tmp_path / "items.parquet")) == len(items)


def test_report_renders(raw):
    from recsys.data.report import render

    md = render(prepare(raw)["stats"])
    assert "| train |" in md and "| test |" in md
    assert "view events" in md
