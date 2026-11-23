import threading

import fakeredis
import numpy as np
import pandas as pd
import pytest

from recsys.data.prepare import prepare
from recsys.features.online_store import OnlineStore
from recsys.index.ann import ItemIndex
from recsys.models.popularity import popularity_scores
from recsys.models.two_tower import TwoTower, build_item_features
from recsys.retrieval.data import PAD
from recsys.streaming.consumer import run
from recsys.streaming.evaluate import next_event_eval
from recsys.streaming.events import Event
from recsys.streaming.live import LiveRecommender, warm_start
from recsys.streaming.producer import iter_events, replay

DAY = 86400.0


@pytest.fixture()
def store():
    leaf = np.array([5, 5, 7, 0])  # item_idx -> leaf category (0 = unknown)
    idx = {100: 0, 101: 1, 102: 2, 103: 3}
    return OnlineStore(
        fakeredis.FakeRedis(), idx, leaf, history_len=3, recent_len=4, session_gap_s=1800
    )


def test_event_roundtrip_and_validation():
    e = Event(1, 2, "addtocart", 10.5, sent_at=11.0)
    assert Event.from_bytes(e.to_bytes()) == e
    with pytest.raises(ValueError):
        Event(1, 2, "like", 0.0)


def test_store_counts_trims_and_trending(store):
    evs = [Event(1, i, "view", t) for t, i in [(0, 100), (10, 101), (20, 102), (30, 103)]]
    evs.append(Event(1, 102, "transaction", 40))
    assert store.apply(evs) == {1}
    assert [i for i, _, _ in store.recent(1)] == [102, 103, 102, 101]  # newest first, cap 4
    st = store.stats(1)
    assert st["events"] == 5 and st["view"] == 4 and st["transaction"] == 1
    assert st["last_ts"] == 40
    assert store.trending(1) == [102]  # 1 view + 3 for the purchase


def test_session_splits_after_gap(store):
    store.apply([Event(1, 100, "view", 0), Event(1, 101, "view", 600)])
    s = store.session(1)
    assert s["session_id"] == 1 and s["session_events"] == 2
    store.apply([Event(1, 102, "view", 600 + 1801)])
    s = store.session(1)
    assert s["session_id"] == 2 and s["session_events"] == 1 and s["session_start"] == 2401


def test_category_affinity_decays(store):
    store.apply([Event(1, 100, "view", 0)])  # leaf 5
    hl = 7 * DAY
    store.apply([Event(1, 102, "addtocart", hl)])  # leaf 7, one half-life later
    aff = store.category_affinity(1)
    assert aff[5] == pytest.approx(0.5, rel=1e-3)
    assert aff[7] == pytest.approx(2.0)
    assert store.category_affinity(1, now=2 * hl)[7] == pytest.approx(1.0, rel=1e-3)
    store.apply([Event(2, 103, "view", 0)])  # unknown leaf is not stored
    assert store.category_affinity(2) == {}


def test_history_arrays_collapse_and_pad(store):
    store.apply(
        [
            Event(1, 100, "view", 0),
            Event(1, 999, "view", 1),  # unknown item: skipped
            Event(1, 101, "view", 2),
            Event(1, 101, "addtocart", 3),  # repeat collapses, keeps weight 2
        ]
    )
    items, w = store.history_arrays([1, 2])
    assert items[0].tolist() == [PAD, 0, 1] and w[0].tolist() == [0, 1, 2]
    assert (items[1] == PAD).all() and (w[1] == 0).all()


@pytest.fixture(scope="module")
def small(raw):
    p = prepare(raw)
    item_map = p["item_map"].sort_values("item_idx")
    feats = build_item_features(p["items"], item_map)
    model = TwoTower(len(item_map), feats, dim=16, max_history=5)
    vecs = model.item_matrix()
    idx = dict(zip(item_map["item_id"], item_map["item_idx"], strict=True))
    return p, model, vecs, idx, feats


def _live(small):
    p, model, vecs, idx, feats = small
    store = OnlineStore(fakeredis.FakeRedis(), idx, feats.leaf, history_len=5)
    return LiveRecommender(store, model, ItemIndex(vecs), p["item_map"]["item_id"].to_numpy())


def test_live_ingest_updates_vector_and_recs(small):
    p, model, vecs, idx, _ = small
    live = _live(small)
    item_ids = list(idx)
    assert live.recommend(42, 5) == []  # cold visitor, empty trending
    live.ingest([Event(42, item_ids[0], "view", 0.0)])
    v1 = live.store.user_vector(42)
    assert v1 is not None and np.isclose(np.linalg.norm(v1), 1.0, atol=1e-4)
    live.ingest([Event(42, item_ids[1], "transaction", 60.0)])
    v2 = live.store.user_vector(42)
    assert not np.allclose(v1, v2)
    # Stored vector equals the user tower on the same history.
    items, w = live.store.history_arrays([42])
    assert np.allclose(v2, model.user_matrix(items, w)[0], atol=1e-5)
    recs = live.recommend(42, 5, exclude_recent=True)
    assert len(recs) == 5 and not {item_ids[0], item_ids[1]} & set(recs)


def test_warm_start_matches_offline_history(small):
    p = small[0]
    live = _live(small)
    visitors = p["train"]["visitor_id"].unique()[:20]
    n = warm_start(live, p["train"], visitors, per_visitor=5)  # store keeps 15 raw
    assert 0 < n <= 20
    some = int(visitors[0])
    last = p["train"][p["train"]["visitor_id"] == some].sort_values("ts").tail(5)
    assert [i for i, _, _ in live.store.recent(some)] == last["item_id"].tolist()[::-1]


class _Msg:
    def __init__(self, value):
        self._v = value

    def error(self):
        return None

    def value(self):
        return self._v


class _FakeBroker:
    """In-memory stand-in for a Kafka producer/consumer pair."""

    def __init__(self):
        self.q, self.lock = [], threading.Lock()

    def produce(self, topic, key, value):
        with self.lock:
            self.q.append(_Msg(value))

    def poll(self, t):
        return None

    def flush(self):
        return 0

    def consume(self, num_messages, timeout):
        with self.lock:
            out, self.q = self.q[:num_messages], self.q[num_messages:]
        return out

    def commit(self, asynchronous=True):
        return None


def test_replay_through_consumer_matches_direct_ingest(small):
    p = small[0]
    events = p["test"].sort_values("ts").head(300)
    broker = _FakeBroker()
    broker.q.append(_Msg(b"not json"))
    sent = replay(iter_events(events), broker)
    assert sent["sent"] == 300

    live = _live(small)
    st = run(broker, live, max_events=300, batch=64)
    s = st.summary()
    assert s["events"] == 300 and s["bad_messages"] == 1
    assert s["freshness_p50_ms"] >= 0

    direct = _live(small)
    direct.ingest(list(iter_events(events)))
    for v in events["visitor_id"].unique()[:10]:
        assert live.store.recent(int(v)) == direct.store.recent(int(v))


def test_next_event_eval_shapes(small):
    p, model, vecs, _, _ = small
    pop = np.argsort(-popularity_scores(p["train"], p["item_map"]))
    hist = pd.concat([p["train"], p["tune"]])
    out = next_event_eval(model, ItemIndex(vecs), hist, p["test"], p["item_map"], pop, k=10,
                          length=5)  # fmt: skip
    cnt = out.pop("_counts")
    assert cnt["events"] > 0 and cnt["new item events"] <= cnt["events"]
    for m in out.values():
        assert 0 <= m["hit@10"] <= 1 and m["mrr@10"] <= m["hit@10"]


@pytest.mark.parametrize("unit", ["ns", "ms", "s"])
def test_iter_events_epoch_seconds_any_unit(unit):
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2015-09-04 00:00:01"]).astype(f"datetime64[{unit}]"),
            "visitor_id": [1],
            "item_id": [2],
            "event": ["view"],
        }
    )
    (e,) = list(iter_events(df))
    assert e.ts == pytest.approx(1441324801.0)
