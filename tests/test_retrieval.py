import numpy as np
import pandas as pd
import pytest
import torch

from recsys.data.prepare import prepare
from recsys.index.ann import ItemIndex
from recsys.models.popularity import popularity_scores, recommend_popular
from recsys.models.two_tower import TwoTower, build_item_features, train_two_tower
from recsys.retrieval.data import (
    PAD,
    build_eval_set,
    build_streams,
    last_history,
    training_examples,
)


@pytest.fixture(scope="module")
def prepared(raw):
    return prepare(raw)


def _events(rows):
    df = pd.DataFrame(rows, columns=["visitor_id", "item_id", "weight"])
    df["ts"] = pd.date_range("2015-06-01", periods=len(df), freq="min")
    return df


def test_streams_collapse_repeats_and_keep_strongest_signal():
    item_map = pd.DataFrame({"item_id": [10, 11, 12], "item_idx": [0, 1, 2]})
    ev = _events([(1, 10, 1), (1, 10, 2), (1, 11, 1), (2, 12, 1), (2, 99, 1)])
    s = build_streams(ev, item_map)
    assert s.item.tolist() == [0, 1, 2]  # repeat collapsed, unknown item 99 dropped
    assert s.weight.tolist() == [2, 1, 1]
    assert s.start.tolist() == [0, 0, 2]


def test_training_examples_only_use_past_events():
    item_map = pd.DataFrame({"item_id": [10, 11, 12], "item_idx": [0, 1, 2]})
    ev = _events([(1, 10, 1), (1, 11, 1), (1, 12, 3), (2, 12, 1)])
    hist, w, tgt = training_examples(build_streams(ev, item_map), length=3)
    assert tgt.tolist() == [1, 2]
    assert hist.tolist() == [[PAD, PAD, 0], [PAD, 0, 1]]
    assert w[0].tolist() == [0, 0, 1]
    v, items, _ = last_history(build_streams(ev, item_map), length=2)
    assert v.tolist() == [1, 2] and items.tolist() == [[1, 2], [PAD, 2]]


def test_eval_set_targets_include_cold_items(prepared):
    history = pd.concat([prepared["train"], prepared["tune"]])
    es = build_eval_set(history, prepared["test"], prepared["item_map"], length=5)
    assert len(es.visitors) == len(es.targets) > 0
    assert all(len(t) > 0 for t in es.targets)
    assert (es.hist_weights[:, -1] > 0).all()  # every visitor has history


def test_popularity_ranks_heaviest_item_first(prepared):
    scores = popularity_scores(prepared["train"], prepared["item_map"])
    recs = recommend_popular(scores, 3, 5)
    assert recs.shape == (3, 5) and scores[recs[0, 0]] == scores.max()


def test_two_tower_learns_and_index_agrees(prepared):
    item_map = prepared["item_map"]
    s = build_streams(prepared["train"], item_map)
    hist, w, tgt = training_examples(s, length=5)
    feats = build_item_features(prepared["items"], item_map)
    model = TwoTower(len(item_map), feats, dim=16, max_history=5)
    freq = np.bincount(tgt, minlength=len(item_map)).astype(float)
    losses = train_two_tower(model, hist, w, tgt, freq, epochs=4, batch_size=64, log=lambda _: None)
    assert losses[-1] < losses[0]

    items = model.item_matrix()
    users = model.user_matrix(hist[:20], w[:20])
    assert np.allclose(np.linalg.norm(items, axis=1), 1, atol=1e-5)
    exact = np.argsort(-(users @ items.T), axis=1)[:, :10]
    for kind in ("flat", "numpy"):
        idx = ItemIndex(items, kind=kind) if kind == "flat" else ItemIndex.__new__(ItemIndex)
        if kind == "numpy":
            idx.vectors, idx.kind = items, "numpy"
        _, got = idx.search(users, 10)
        assert (got[:, 0] == exact[:, 0]).all()


def test_user_tower_ignores_padding():
    feats = build_item_features(
        pd.DataFrame(
            {
                "item_id": [1, 2],
                "category_id": [5, 6],
                "top_category": [5, 5],
                "price": [10.0, 20.0],
                "available": [1, 0],
            }
        ),
        pd.DataFrame({"item_id": [1, 2], "item_idx": [0, 1]}),
    )
    model = TwoTower(2, feats, dim=8, max_history=3)
    a = model.encode_users(torch.tensor([[PAD, PAD, 0]]), torch.tensor([[0.0, 0.0, 1.0]]))
    b = model.encode_users(torch.tensor([[1, PAD, 0]]), torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(a, b)
