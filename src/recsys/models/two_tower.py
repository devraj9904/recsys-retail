"""Two-tower retrieval model.

Item tower: item-id embedding + leaf category + top-level category + price
bucket + availability, fused by an MLP into an L2-normalised vector.

User tower: the visitor's recent history (up to ``max_history`` warm items)
encoded with the *same* item tower, pooled with event-type and recency
weights, then passed through an MLP. Because the user is represented by
their history rather than a learned user-id embedding, the model can encode
any visitor with at least one event, including ones first seen after
training. Phase 3 relies on this to refresh preferences from the event
stream without retraining.

Loss: in-batch sampled softmax with logQ correction (Yi et al., 2019), which
removes the bias against popular items that in-batch negatives introduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

N_PRICE_BUCKETS = 20


@dataclass
class ItemFeatures:
    """Dense per-item_idx feature arrays (index 0..n_items-1)."""

    leaf: np.ndarray  # int64, 0 = unknown
    top: np.ndarray  # int64, 0 = unknown
    price_bucket: np.ndarray  # int64, 0 = unknown
    available: np.ndarray  # int64: 0 unknown, 1 no, 2 yes

    @property
    def cardinalities(self) -> dict[str, int]:
        return {
            "leaf": int(self.leaf.max()) + 1,
            "top": int(self.top.max()) + 1,
            "price_bucket": N_PRICE_BUCKETS + 1,
            "available": 3,
        }


def build_item_features(items: pd.DataFrame, item_map: pd.DataFrame) -> ItemFeatures:
    df = item_map.merge(items, on="item_id", how="left").sort_values("item_idx")

    def codes(col: pd.Series) -> np.ndarray:
        c = pd.Categorical(col.astype("Int64").astype("string"))
        return (c.codes.astype(np.int64) + 1).clip(min=0)  # NaN (-1) -> 0

    price = df["price"].to_numpy(dtype=float)
    ok = np.isfinite(price) & (price > 0)
    bucket = np.zeros(len(df), dtype=np.int64)
    if ok.any():
        edges = np.quantile(np.log1p(price[ok]), np.linspace(0, 1, N_PRICE_BUCKETS + 1)[1:-1])
        bucket[ok] = np.searchsorted(edges, np.log1p(price[ok])) + 1
    avail = df["available"].astype("Float64").fillna(-1).to_numpy()
    return ItemFeatures(
        leaf=codes(df["category_id"]),
        top=codes(df["top_category"]),
        price_bucket=bucket,
        available=np.where(avail < 0, 0, avail + 1).astype(np.int64),
    )


class ItemTower(nn.Module):
    def __init__(self, n_items: int, feats: ItemFeatures, dim: int = 64):
        super().__init__()
        card = feats.cardinalities
        self.id_emb = nn.Embedding(n_items, dim)
        self.leaf_emb = nn.Embedding(card["leaf"], dim // 2)
        self.top_emb = nn.Embedding(card["top"], dim // 4)
        self.price_emb = nn.Embedding(card["price_bucket"], dim // 4)
        self.avail_emb = nn.Embedding(card["available"], dim // 8)
        in_dim = dim + dim // 2 + dim // 4 + dim // 4 + dim // 8
        self.mlp = nn.Sequential(nn.Linear(in_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        for name in ("leaf", "top", "price_bucket", "available"):
            self.register_buffer(f"f_{name}", torch.as_tensor(getattr(feats, name)))
        nn.init.normal_(self.id_emb.weight, std=0.05)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                self.id_emb(idx),
                self.leaf_emb(self.f_leaf[idx]),
                self.top_emb(self.f_top[idx]),
                self.price_emb(self.f_price_bucket[idx]),
                self.avail_emb(self.f_available[idx]),
            ],
            dim=-1,
        )
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self, dim: int = 64, max_history: int = 30):
        super().__init__()
        self.type_w = nn.Parameter(torch.tensor([0.0, 1.0, 1.5, 2.0]))  # pad, view, cart, buy
        self.recency = nn.Parameter(torch.linspace(-1.0, 0.0, max_history))
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, hist_vecs: torch.Tensor, hist_w: torch.Tensor) -> torch.Tensor:
        # hist_vecs (B, L, D) from the item tower; hist_w (B, L) event weights, 0 = pad.
        mask = hist_w > 0
        logits = self.type_w[hist_w.long()] + self.recency[-hist_w.shape[1] :]
        logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=1).nan_to_num(0.0).unsqueeze(-1)
        pooled = (attn * hist_vecs).sum(1)
        last = hist_vecs[:, -1] * mask[:, -1:].float()
        return F.normalize(self.mlp(torch.cat([pooled, last], dim=-1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        feats: ItemFeatures,
        dim: int = 64,
        max_history: int = 30,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.items = ItemTower(n_items, feats, dim)
        self.users = UserTower(dim, max_history)
        self.temperature = temperature

    def encode_users(self, hist: torch.Tensor, hist_w: torch.Tensor) -> torch.Tensor:
        vecs = self.items(hist.clamp(min=0))
        return self.users(vecs, hist_w)

    def loss(self, hist, hist_w, target, log_q):
        u = self.encode_users(hist, hist_w)
        v = self.items(target)
        logits = u @ v.T / self.temperature - log_q[target][None, :]
        # The same item can appear twice in a batch; don't punish it as a negative.
        dup = (target[:, None] == target[None, :]) & ~torch.eye(
            len(target), dtype=torch.bool, device=target.device
        )
        logits = logits.masked_fill(dup, float("-inf"))
        return F.cross_entropy(logits, torch.arange(len(target), device=target.device))

    @torch.no_grad()
    def item_matrix(self, batch: int = 65536) -> np.ndarray:
        self.eval()
        n = self.items.id_emb.num_embeddings
        out = [self.items(torch.arange(i, min(i + batch, n))) for i in range(0, n, batch)]
        return torch.cat(out).numpy().astype(np.float32)

    @torch.no_grad()
    def user_matrix(self, hist: np.ndarray, hist_w: np.ndarray, batch: int = 8192) -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(hist), batch):
            h = torch.as_tensor(hist[i : i + batch])
            w = torch.as_tensor(hist_w[i : i + batch])
            out.append(self.encode_users(h, w))
        return torch.cat(out).numpy().astype(np.float32)


def train_two_tower(
    model: TwoTower,
    hist: np.ndarray,
    hist_w: np.ndarray,
    target: np.ndarray,
    item_freq: np.ndarray,
    epochs: int = 3,
    batch_size: int = 2048,
    lr: float = 2e-3,
    seed: int = 0,
    log=print,
) -> list[float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = (item_freq + 1) / (item_freq + 1).sum()
    log_q = torch.as_tensor(np.log(q), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    hist_t, w_t, tgt_t = map(torch.as_tensor, (hist, hist_w, target))
    losses = []
    for epoch in range(epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(target)))
        total, steps = 0.0, 0
        for i in range(0, len(order) - batch_size + 1, batch_size):
            b = order[i : i + batch_size]
            loss = model.loss(hist_t[b], w_t[b], tgt_t[b], log_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        losses.append(total / max(steps, 1))
        log(f"epoch {epoch + 1}/{epochs} loss {losses[-1]:.4f}")
    return losses
