import numpy as np
import pandas as pd
import pytest

from recsys.data.prepare import prepare
from recsys.index.ann import ItemIndex
from recsys.models.popularity import popularity_scores
from recsys.models.two_tower import TwoTower, build_item_features
from recsys.ranking.candidates import generate_candidates
from recsys.ranking.diversity import intra_list_diversity, mmr
from recsys.ranking.features import FEATURES, attach_labels, build_features
from recsys.ranking.ranker import blend, train_lambdarank, train_purchase
from recsys.retrieval.data import build_eval_set


@pytest.fixture(scope="module")
def ranking_rows(raw):
    p = prepare(raw)
    item_map = p["item_map"]
    feats = build_item_features(p["items"], item_map)
    model = TwoTower(len(item_map), feats, dim=16, max_history=5)
    item_vecs = model.item_matrix()
    es = build_eval_set(p["train"], p["tune"], item_map, length=5)
    user_vecs = model.user_matrix(es.hist_items, es.hist_weights)
    trending = np.argsort(-popularity_scores(p["train"], item_map))
    cands = generate_candidates(es, user_vecs, item_vecs, ItemIndex(item_vecs), trending, n_tt=20)
    cutoff = p["tune"]["ts"].min()
    df = build_features(cands, p["train"], item_map, feats, cutoff)
    return p, es, cands, attach_labels(df, p["tune"], item_map)


def test_candidates_union_sources(ranking_rows):
    _, es, cands, _ = ranking_rows
    assert not cands.duplicated(["u", "item_idx"]).any()
    assert cands["u"].nunique() == len(es.visitors)
    # Every visitor's most recent item is a candidate.
    for u, h in enumerate(es.hist_items[:20]):
        assert ((cands["u"] == u) & (cands["item_idx"] == h[-1])).any()


def test_features_complete_and_leak_free(ranking_rows):
    p, _, _, df = ranking_rows
    assert set(FEATURES) <= set(df.columns)
    assert df[["ui_events", "item_pop", "leaf_share"]].notna().all().all()
    assert df["label"].isin([0, 1, 2, 3]).all()
    # History features only see pre-cutoff events: no item counts beyond train totals.
    train_counts = p["train"].merge(p["item_map"], on="item_id").groupby("item_idx").size()
    got = df.groupby("item_idx")["item_views"].first()
    assert (got <= train_counts.reindex(got.index, fill_value=0)).all()


def test_rankers_train_and_score(ranking_rows):
    _, _, _, df = ranking_rows
    ranker = train_lambdarank(df, n_estimators=20)
    buyer = train_purchase(df, n_estimators=20)
    e = pd.Series(ranker.predict(df[FEATURES]))
    b = pd.Series(buyer.predict_proba(df[FEATURES])[:, 1])
    s = blend(e, b, df["u"].reset_index(drop=True), 0.3)
    assert len(s) == len(df) and np.isfinite(s).all()


def test_blend_extremes_follow_single_objective():
    groups = pd.Series([0, 0, 0])
    e, b = pd.Series([3.0, 2.0, 1.0]), pd.Series([0.1, 0.5, 0.9])
    assert blend(e, b, groups, 0.0).idxmax() == 0
    assert blend(e, b, groups, 1.0).idxmax() == 2


def test_mmr_trades_relevance_for_diversity():
    vecs = np.array([[1, 0], [0.99, 0.141], [0, 1.0]])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    scores = np.array([1.0, 0.9, 0.5])
    assert mmr(scores, vecs, 2, 0.0).tolist() == [0, 1]
    picked = mmr(scores, vecs, 2, 0.5)
    assert picked.tolist() == [0, 2]
    assert intra_list_diversity(vecs[picked]) > intra_list_diversity(vecs[[0, 1]])
