"""Vector index over item embeddings: FAISS when available, numpy otherwise.

``flat`` is exact inner-product search; ``hnsw`` is the graph index a
production catalogue of millions of items would use. Both take
L2-normalised vectors, so inner product equals cosine similarity.
"""

from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only without faiss installed
    faiss = None


class ItemIndex:
    def __init__(
        self, vectors: np.ndarray, kind: str = "flat", hnsw_m: int = 32, ef_search: int = 128
    ):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.kind = kind if faiss is not None else "numpy"
        d = self.vectors.shape[1]
        if self.kind == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif self.kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
        elif self.kind != "numpy":
            raise ValueError(f"unknown index kind {kind!r}")
        if self.kind != "numpy":
            self.index.add(self.vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if self.kind == "numpy":
            scores = q @ self.vectors.T
            top = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
            part = np.take_along_axis(scores, top, 1)
            order = np.argsort(-part, axis=1)
            return np.take_along_axis(part, order, 1), np.take_along_axis(top, order, 1)
        return self.index.search(q, k)
