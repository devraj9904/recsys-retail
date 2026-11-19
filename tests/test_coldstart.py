import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.coldstart import methods as M
from recsys.coldstart.catalog import build_catalog, lookup
from recsys.coldstart.content import (
    ContentEncoder,
    build_content_features,
    tfidf_matrix,
    train_content_encoder,
)
from recsys.data.prepare import prepare
from recsys.train.run_coldstart import new_visitor_tasks


@pytest.fixture(scope="module")
def prep(raw):
    p = prepare(raw)
    cat = build_catalog(p["item_map"], p["items"]["item_id"].to_numpy(),
                        p["test"]["item_id"].to_numpy())  # fmt: skip
    return p, cat, build_content_features(p["items"], cat, min_df=1)


def test_catalog_keeps_warm_ids_and_appends_cold(prep):
    p, cat, _ = prep
    n_warm = len(p["item_map"])
    warm = cat[cat["warm"]]
    assert (warm["cidx"].to_numpy() == np.arange(n_warm)).all()
    m = p["item_map"].sort_values("item_idx")
    assert (lookup(cat, m["item_id"].to_numpy()) == m["item_idx"].to_numpy()).all()
    assert cat["cidx"].is_unique and lookup(cat, np.array([-12345]))[0] == -1
    assert (~cat["warm"]).sum() > 0


def test_content_features_and_tfidf(prep):
    _, cat, f = prep
    assert len(f) == len(cat) and f.tok_off[-1] == len(f.tok_idx)
    x = tfidf_matrix(f)
    norms = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_content_encoder_learns_to_align(prep):
    p, cat, f = prep
    n_warm = len(p["item_map"])
    rng = np.random.default_rng(0)
    # Targets that depend on the leaf category only: learnable from content.
    centers = rng.normal(size=(f.leaf.max() + 1, 16)).astype(np.float32)
    tgt = centers[f.leaf[:n_warm]]
    tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
    enc = ContentEncoder(f, dim=16)
    before = (enc.encode(np.arange(n_warm)) * tgt).sum(1).mean()
    losses = train_content_encoder(enc, np.arange(n_warm), tgt, epochs=15, batch_size=128)
    after = (enc.encode(np.arange(n_warm)) * tgt).sum(1).mean()
    assert losses[-1] < losses[0] and after > before + 0.2
    allv = enc.encode()
    assert allv.shape == (len(cat), 16) and np.allclose(np.linalg.norm(allv, axis=1), 1, atol=1e-4)


def test_methods_exclude_context_and_pad():
    ctx = [np.array([0]), np.array([1, 2])]
    pop = np.array([5.0, 4, 3, 2, 1, 0])
    r = M.popular(pop, ctx, 3)
    assert r[0].tolist() == [1, 2, 3] and r[1].tolist() == [0, 3, 4]
    leaf = np.array([1, 1, 2, 2, 1, 0])
    r = M.category_popular(pop, leaf, ctx, 3)
    assert r[0].tolist() == [1, 4, 2]  # leaf-1 items by popularity, then global
    filled = M.fill(np.array([[7, -2, -2]]), np.array([[7, 8, 9]]))
    assert filled.tolist() == [[7, 8, 9]]


def test_coview_and_knn():
    # sessions: (0,1), (0,1), (1,2)
    c = M.coview_matrix(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 0, 1, 1, 2]), 4)
    assert c[0, 0] == 0 and c[0, 1] > c[1, 2] > 0 and c[0, 3] == 0
    r = M.item_knn(c, [np.array([0])], 2)
    assert r[0, 0] == 1
    x = sp.csr_matrix(np.array([[1, 0], [1, 0.1], [0, 1]], dtype=np.float32))
    r = M.content_knn(x, [np.array([0])], 2)
    assert r[0].tolist() == [1, 2]


def test_new_visitor_tasks_split_context_and_targets():
    test = pd.DataFrame(
        {
            "visitor_id": [1, 1, 1, 1, 2, 3, 3],
            "item_id": [10, 10, 11, 12, 10, 11, 99],
            "ts": pd.to_datetime([1, 2, 3, 4, 1, 1, 2], unit="s"),
        }
    )
    cat = pd.DataFrame({"item_id": [10, 11, 12], "cidx": [0, 1, 2], "warm": True})
    ctx, tgt, vis = new_visitor_tasks(test, np.array([3]), cat, 1, None, 0)
    assert vis.tolist() == [1]  # 2 has one item, 3 is returning
    assert ctx[0].tolist() == [0] and tgt[0].tolist() == [1, 2]
