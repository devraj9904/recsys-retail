from recsys.data.loaders import EVENT_TYPES


def test_events_normalised(raw):
    ev = raw.events
    assert list(ev.columns) == ["ts", "visitor_id", "item_id", "event", "transaction_id"]
    assert set(ev["event"].astype(str)) <= set(EVENT_TYPES)
    assert str(ev["ts"].dtype).startswith("datetime64")


def test_properties_and_tree(raw):
    assert {"categoryid", "available", "790"} <= set(raw.item_properties["property"].astype(str))
    assert raw.category_tree["parent_id"].isna().any()  # roots exist
