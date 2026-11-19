import fakeredis
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from recsys.data.prepare import prepare, write
from recsys.features.online_store import OnlineStore
from recsys.index.ann import ItemIndex
from recsys.models.popularity import popularity_scores
from recsys.models.two_tower import TwoTower, build_item_features
from recsys.ranking.candidates import generate_candidates
from recsys.ranking.features import FEATURES, attach_labels, build_features
from recsys.ranking.ranker import train_lambdarank, train_purchase
from recsys.retrieval.data import build_eval_set
from recsys.serving.app import create_app
from recsys.serving.bundle import Bundle, build_bundle
from recsys.serving.online_features import visitor_features
from recsys.serving.recommender import Recommender
from recsys.serving.warm import warm
from recsys.streaming.live import LiveRecommender

L = 30  # history length, as in production


@pytest.fixture(scope="module")
def world(raw, tmp_path_factory):
    torch.manual_seed(0)
    p = prepare(raw)
    proc = tmp_path_factory.mktemp("proc")
    write(p, proc)
    bundle = build_bundle(proc, n_demo=5)
    bundle.save(proc / "serving")
    bundle = Bundle.load(proc / "serving")  # round trip

    item_map = p["item_map"].sort_values("item_idx")
    feats = build_item_features(p["items"], item_map)
    model = TwoTower(len(item_map), feats, dim=16, max_history=L).eval()
    vecs = model.item_matrix()

    history = pd.concat([p["train"], p["tune"]], ignore_index=True)
    cutoff = p["test"]["ts"].min()
    es = build_eval_set(history, p["test"], item_map, L)
    user_vecs = model.user_matrix(es.hist_items, es.hist_weights)
    week = history[history["ts"] >= cutoff - pd.Timedelta(days=7)]
    trending = np.argsort(-popularity_scores(week, item_map), kind="stable")
    cands = generate_candidates(es, user_vecs, vecs, ItemIndex(vecs), trending, n_tt=20)
    offline = build_features(cands, history, item_map, feats, cutoff)
    labelled = attach_labels(offline, p["test"], item_map)
    ranker = train_lambdarank(labelled, n_estimators=20).booster_
    buyer = train_purchase(labelled, n_estimators=20).booster_

    idx = dict(zip(item_map["item_id"], item_map["item_idx"], strict=True))
    store = OnlineStore(fakeredis.FakeRedis(), idx, feats.leaf, history_len=L)
    live = LiveRecommender(store, model, ItemIndex(vecs), item_map["item_id"].to_numpy())
    rec = Recommender(bundle, live, ranker, buyer)
    return dict(p=p, proc=proc, rec=rec, history=history, cutoff=cutoff, offline=offline,
                es=es)  # fmt: skip


def test_bundle_item_stats_match_offline(world):
    off, b = world["offline"], world["rec"].b
    for name in ("item_pop", "item_pop_7d", "item_views", "item_cart_rate", "item_buy_rate"):
        got = b.stats[name][off["item_idx"].to_numpy()]
        assert np.allclose(got, off[name].to_numpy()), name


def test_online_features_match_offline(world):
    rec, off, history = world["rec"], world["offline"], world["history"]
    counts = history.groupby("visitor_id").size()
    visitors = [v for v in off["visitor_id"].unique() if counts.get(v, 0) <= 90][:25]
    assert visitors
    from recsys.streaming.live import warm_start

    warm_start(rec.live, history, np.array(visitors), per_visitor=90)
    clock = world["cutoff"].value / 1e9
    retrieval_cols = FEATURES[:7]
    for v in visitors:
        rows = off[off["visitor_id"] == v]
        cand = rows["item_idx"].to_numpy()
        ev = [(rec.cidx[i], w, ts) for i, w, ts in rec.store.recent(int(v))
              if rec.cidx.get(i, rec.b.n_warm) < rec.b.n_warm]  # fmt: skip
        X = visitor_features(
            rec.b, cand, {c: rows[c].to_numpy(float) for c in retrieval_cols},
            np.array([e[0] for e in ev], np.int64), np.array([e[1] for e in ev], float),
            np.array([e[2] for e in ev], float), clock,
        )  # fmt: skip
        expected = rows[FEATURES].to_numpy(float)
        assert np.allclose(X, expected, equal_nan=True, atol=1e-3), (
            v, [f for j, f in enumerate(FEATURES)
                if not np.allclose(X[:, j], expected[:, j], equal_nan=True, atol=1e-3)],
        )  # fmt: skip


def test_routes_cache_and_invalidation(world):
    rec, p = world["rec"], world["p"]
    rec.store.r.flushdb()
    n = warm(rec, world["proc"], per_visitor=90)
    assert n > 0
    v = int(rec.b.demo_visitors[0]) if rec.b.demo_visitors else int(world["es"].visitors[0])
    r1 = rec.recommend(v, 10)
    assert r1.route == "ranked" and len(r1.items) == 10 and not r1.cached
    assert {"retrieval", "features", "ranking", "diversity", "total"} <= set(r1.timings_ms)
    assert len({i["item_id"] for i in r1.items}) == 10
    r2 = rec.recommend(v, 10)
    assert r2.cached and [i["item_id"] for i in r2.items] == [i["item_id"] for i in r1.items]

    # An event drops the cache.
    from recsys.streaming.events import Event

    rec.live.ingest([Event(v, int(r1.items[0]["item_id"]), "addtocart", rec.clock() + 1)])
    assert not rec.recommend(v, 10).cached

    # Brand-new visitor: trending; after a click on a new item: content route.
    assert rec.recommend(10**9, 5).route == "trending"
    cold_item = int(rec.b.item_ids[rec.b.n_warm]) if len(rec.b.item_ids) > rec.b.n_warm else None
    if cold_item is not None:
        rec.live.ingest([Event(10**9 + 1, cold_item, "view", rec.clock() + 1)])
        r = rec.recommend(10**9 + 1, 5)
        assert r.route == "content" and cold_item not in [i["item_id"] for i in r.items]
    # After a click on a warm item: ranked.
    warm_item = int(rec.b.item_ids[0])
    rec.live.ingest([Event(10**9 + 2, warm_item, "view", rec.clock() + 1)])
    assert rec.recommend(10**9 + 2, 5).route == "ranked"
    del p


def test_api_end_to_end(world):
    rec = world["rec"]
    client = TestClient(create_app(rec))
    assert client.get("/health").json()["status"] == "ok"
    assert "recsys-retail" in client.get("/").text
    demo = client.get("/demo/visitors").json()["visitors"]
    v = demo[0] if demo else 10**9 + 5
    r = client.get(f"/recommend/{v}", params={"k": 8}).json()
    assert len(r["items"]) == 8
    item = r["items"][0]["item_id"]
    assert client.get(f"/items/{item}").json()["item_id"] == item
    assert client.get("/items/123456789").status_code == 404
    ok = client.post("/events", json={"visitor_id": v, "item_id": item, "event": "view"})
    assert ok.json()["accepted"]
    bad = client.post("/events", json={"visitor_id": v, "item_id": item, "event": "like"})
    assert bad.status_code == 422
    hist = client.get(f"/visitors/{v}").json()
    assert hist["recent"][0]["item_id"] == item
    assert not client.get(f"/recommend/{v}", params={"k": 8}).json()["cached"]
    m = client.get("/metrics").text
    assert "recsys_request_seconds" in m and 'recsys_events_total{event="view"' in m
