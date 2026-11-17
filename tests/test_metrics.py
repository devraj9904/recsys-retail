import numpy as np
import pytest

from recsys.eval.metrics import evaluate


def test_perfect_and_empty_rankings():
    recs = np.array([[1, 2, 3], [7, 8, 9]])
    targets = [np.array([1]), np.array([4])]
    m = evaluate(recs, targets, ks=(3,), n_items=10)
    assert m["recall@3"] == 0.5
    assert m["hit@3"] == 0.5
    assert m["ndcg@3"] == 0.5
    assert m["coverage@3"] == 0.6


def test_cold_targets_count_as_misses():
    recs = np.array([[1, 2]])
    m = evaluate(recs, [np.array([1, -1])], ks=(2,))
    assert m["recall@2"] == 0.5


def test_ndcg_rewards_earlier_hits():
    targets = [np.array([5])]
    early = evaluate(np.array([[5, 0, 1]]), targets, ks=(3,))["ndcg@3"]
    late = evaluate(np.array([[0, 1, 5]]), targets, ks=(3,))["ndcg@3"]
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(0.5)


def test_users_without_targets_are_skipped():
    m = evaluate(np.array([[1], [2]]), [np.array([1]), np.array([], dtype=int)], ks=(1,))
    assert m["users"] == 1 and m["recall@1"] == 1.0
