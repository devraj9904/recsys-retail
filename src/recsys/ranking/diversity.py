"""Diversity-aware reranking (Maximal Marginal Relevance) and list diversity.

MMR picks items greedily by (1 - lam) * relevance - lam * max similarity to
the items already picked, using the two-tower item embeddings as the
similarity space. lam = 0 reproduces the relevance ranking.
"""

from __future__ import annotations

import numpy as np


def mmr(scores: np.ndarray, vecs: np.ndarray, k: int, lam: float) -> np.ndarray:
    """Return indices into ``scores`` (one user's candidates) in MMR order."""
    n = len(scores)
    k = min(k, n)
    if lam <= 0 or n <= 1:
        return np.argsort(-scores, kind="stable")[:k]
    rel = (scores - scores.min()) / (np.ptp(scores) + 1e-9)
    sim = vecs @ vecs.T
    chosen = [int(np.argmax(rel))]
    max_sim = sim[chosen[0]].copy()
    available = np.ones(n, dtype=bool)
    available[chosen[0]] = False
    for _ in range(k - 1):
        gain = np.where(available, (1 - lam) * rel - lam * max_sim, -np.inf)
        j = int(np.argmax(gain))
        chosen.append(j)
        available[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    return np.array(chosen)


def intra_list_diversity(vecs: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity of a list (vectors L2-normalised)."""
    n = len(vecs)
    if n < 2:
        return 0.0
    sim = vecs @ vecs.T
    return float(1 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))
