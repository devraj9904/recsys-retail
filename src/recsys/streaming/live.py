"""Live retrieval state: online features in Redis + the two-tower user tower.

``LiveRecommender`` is what the consumer and (in Phase 5) the API share:

* ``ingest(events)`` writes the online features and re-encodes the user
  vector of every visitor the batch touched, so the next request already
  sees the click.
* ``recommend(visitor, k)`` reads the stored vector and searches the item
  index, skipping what the visitor just interacted with when asked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recsys.features.online_store import OnlineStore
from recsys.index.ann import ItemIndex
from recsys.models.two_tower import TwoTower, build_item_features
from recsys.streaming.events import Event


class LiveRecommender:
    def __init__(self, store: OnlineStore, model: TwoTower, index: ItemIndex, item_ids: np.ndarray):
        self.store = store
        self.model = model.eval()
        self.index = index
        self.item_ids = item_ids  # item_idx -> item_id

    @classmethod
    def from_artifacts(cls, client, processed: Path, retrieval: Path, history_len: int = 30):
        item_map = pd.read_parquet(processed / "item_map.parquet")
        items = pd.read_parquet(processed / "items.parquet")
        feats = build_item_features(items, item_map)
        model = TwoTower(len(item_map), feats, dim=64, max_history=history_len)
        model.load_state_dict(torch.load(retrieval / "two_tower.pt"))
        vecs = np.load(retrieval / "item_vectors.npy")
        item_map = item_map.sort_values("item_idx")
        idx = dict(zip(item_map["item_id"].to_numpy().tolist(),
                       item_map["item_idx"].to_numpy().tolist(), strict=True))  # fmt: skip
        store = OnlineStore(client, idx, feats.leaf, history_len=history_len)
        return cls(store, model, ItemIndex(vecs, kind="flat"), item_map["item_id"].to_numpy())

    @torch.no_grad()
    def refresh_vectors(self, visitors: list[int]) -> np.ndarray:
        items, weights = self.store.history_arrays(visitors)
        has = (weights > 0).any(axis=1)
        vecs = np.zeros((len(visitors), self.index.vectors.shape[1]), dtype=np.float32)
        if has.any():
            vecs[has] = self.model.user_matrix(items[has], weights[has])
            self.store.set_user_vectors([v for v, h in zip(visitors, has, strict=True) if h],
                                        vecs[has])  # fmt: skip
        return vecs

    def ingest(self, events: list[Event]) -> set[int]:
        touched = self.store.apply(events)
        if touched:
            self.refresh_vectors(sorted(touched))
        return touched

    def recommend(self, visitor: int, k: int = 20, exclude_recent: bool = False) -> list[int]:
        vec = self.store.user_vector(visitor)
        if vec is None:  # cold visitor: trending until Phase 4's cold-start model
            return self.store.trending(k)
        seen = {i for i, _, _ in self.store.recent(visitor)} if exclude_recent else set()
        _, idx = self.index.search(vec[None, :], k + len(seen))
        out = [int(self.item_ids[i]) for i in idx[0] if i >= 0]
        return [i for i in out if i not in seen][:k]


def warm_start(
    live: LiveRecommender,
    history: pd.DataFrame,
    visitors: np.ndarray,
    per_visitor: int = 30,
    chunk: int = 5000,
) -> int:
    """Load each visitor's last ``per_visitor`` pre-stream events and encode them.

    This is the batch -> online hand-off: in production a nightly job would
    materialise the same state from the warehouse before the stream resumes.
    """
    from recsys.streaming.producer import iter_events

    h = history[history["visitor_id"].isin(visitors)].sort_values(["visitor_id", "ts"])
    h = h[h.groupby("visitor_id").cumcount(ascending=False) < per_visitor]
    batch: list[Event] = []
    touched: set[int] = set()
    for e in iter_events(h):
        batch.append(e)
        if len(batch) >= chunk:
            touched |= live.store.apply(batch)
            batch = []
    if batch:
        touched |= live.store.apply(batch)
    ordered = sorted(touched)
    for i in range(0, len(ordered), 8192):
        live.refresh_vectors(ordered[i : i + 8192])
    return len(ordered)
