"""Content representations of items: what we know before anyone clicks.

Two views of the same metadata (leaf and top category, price, availability
and RetailRocket's hashed property values):

* ``tfidf_matrix``: a sparse, purely content-based vector (no behaviour).
* ``ContentEncoder``: a small network trained to put an item's metadata
  where the two-tower model put its *behavioural* vector, using warm items
  as supervision. A new item then lands next to the warm items people treat
  the same way, and the Phase 1 user tower can score it immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

N_PRICE_BUCKETS = 20


@dataclass
class ContentFeatures:
    leaf: np.ndarray  # int64 per cidx, 0 = unknown
    top: np.ndarray
    price_bucket: np.ndarray
    available: np.ndarray  # 0 unknown, 1 no, 2 yes
    tok_idx: np.ndarray  # flat token ids (1-based, 0 unused)
    tok_off: np.ndarray  # offsets into tok_idx, len n_items + 1
    n_tokens: int

    def __len__(self) -> int:
        return len(self.leaf)


def build_content_features(
    items: pd.DataFrame, catalog: pd.DataFrame, min_df: int = 5
) -> ContentFeatures:
    df = catalog[["item_id", "cidx"]].merge(items, on="item_id", how="left").sort_values("cidx")

    def codes(col: pd.Series) -> np.ndarray:
        c = pd.Categorical(col.astype("Int64").astype("string"))
        return (c.codes.astype(np.int64) + 1).clip(min=0)

    price = df["price"].to_numpy(dtype=float)
    ok = np.isfinite(price) & (price > 0)
    bucket = np.zeros(len(df), dtype=np.int64)
    if ok.any():
        edges = np.quantile(np.log1p(price[ok]), np.linspace(0, 1, N_PRICE_BUCKETS + 1)[1:-1])
        bucket[ok] = np.searchsorted(edges, np.log1p(price[ok])) + 1
    avail = df["available"].astype("Float64").fillna(-1).to_numpy()

    toks = df["tokens"].apply(lambda t: list(t) if isinstance(t, list | np.ndarray) else [])
    flat = pd.Series([t for ts in toks for t in ts], dtype="string")
    counts = flat.value_counts()
    vocab = {t: i + 1 for i, t in enumerate(counts.index[counts >= min_df])}
    ids = [[vocab[t] for t in ts if t in vocab] for ts in toks]
    off = np.zeros(len(ids) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(x) for x in ids])
    flat_ids = np.fromiter((i for x in ids for i in x), dtype=np.int64, count=int(off[-1]))
    return ContentFeatures(
        leaf=codes(df["category_id"]),
        top=codes(df["top_category"]),
        price_bucket=bucket,
        available=np.where(avail < 0, 0, avail + 1).astype(np.int64),
        tok_idx=flat_ids,
        tok_off=off,
        n_tokens=len(vocab) + 1,
    )


def tfidf_matrix(
    f: ContentFeatures, category_weight: float = 3.0, max_df: float = 0.05
) -> sp.csr_matrix:
    """L2-normalised TF-IDF over property tokens + one-hot leaf/top/price.

    Tokens on more than ``max_df`` of items (shared boilerplate values) are
    dropped: they say little and make the similarity product dense.
    """
    n = len(f)
    rows = np.repeat(np.arange(n), np.diff(f.tok_off))
    tok = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, f.tok_idx)), shape=(n, f.n_tokens))
    df = np.asarray((tok > 0).sum(0)).ravel()
    idf = np.log((1 + n) / (1 + df)).astype(np.float32)
    idf[df > max_df * n] = 0
    tok = tok @ sp.diags(idf)

    def onehot(codes: np.ndarray, w: float) -> sp.csr_matrix:
        known = np.flatnonzero(codes > 0)  # unknown (0) carries no signal
        vals = np.full(len(known), w, np.float32)
        return sp.csr_matrix((vals, (known, codes[known])), shape=(n, int(codes.max()) + 1))

    x = sp.hstack(
        [
            tok,
            onehot(f.leaf, category_weight * 2),
            onehot(f.top, category_weight),
            onehot(f.price_bucket, category_weight),
        ]
    ).tocsr()
    x.eliminate_zeros()
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    return (sp.diags(1 / np.maximum(norm, 1e-9)) @ x).tocsr().astype(np.float32)


class ContentEncoder(nn.Module):
    def __init__(self, f: ContentFeatures, dim: int = 64, tok_dim: int = 64):
        super().__init__()
        self.f = f
        self.leaf = nn.Embedding(int(f.leaf.max()) + 1, 32)
        self.top = nn.Embedding(int(f.top.max()) + 1, 8)
        self.price = nn.Embedding(N_PRICE_BUCKETS + 1, 8)
        self.avail = nn.Embedding(3, 4)
        self.tokens = nn.EmbeddingBag(f.n_tokens, tok_dim, mode="mean")
        self.mlp = nn.Sequential(
            nn.Linear(32 + 8 + 8 + 4 + tok_dim, 256), nn.GELU(), nn.Linear(256, dim)
        )

    def _bags(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        starts, ends = self.f.tok_off[idx], self.f.tok_off[idx + 1]
        lens = ends - starts
        flat = np.concatenate([self.f.tok_idx[s:e] for s, e in zip(starts, ends, strict=True)])
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        return torch.as_tensor(flat, dtype=torch.long), torch.as_tensor(offsets)

    def forward(self, idx: np.ndarray) -> torch.Tensor:
        t = lambda a: torch.as_tensor(a[idx])  # noqa: E731
        toks, offs = self._bags(idx)
        x = torch.cat(
            [
                self.leaf(t(self.f.leaf)),
                self.top(t(self.f.top)),
                self.price(t(self.f.price_bucket)),
                self.avail(t(self.f.available)),
                self.tokens(toks, offs),
            ],
            dim=-1,
        )
        return F.normalize(self.mlp(x), dim=-1)

    @torch.no_grad()
    def encode(self, idx: np.ndarray | None = None, batch: int = 65536) -> np.ndarray:
        self.eval()
        idx = np.arange(len(self.f)) if idx is None else idx
        out = [self(idx[i : i + batch]) for i in range(0, len(idx), batch)]
        return torch.cat(out).numpy().astype(np.float32)


def train_content_encoder(
    enc: ContentEncoder,
    idx: np.ndarray,
    targets: np.ndarray,
    epochs: int = 8,
    batch_size: int = 2048,
    lr: float = 2e-3,
    temperature: float = 0.05,
    loss: str = "cosine",
    seed: int = 0,
) -> list[float]:
    """Regress content vectors onto behavioural (two-tower) vectors.

    ``cosine`` maximises cosine similarity to the item's own two-tower
    vector. ``softmax`` instead asks each content vector to pick out its own
    item's vector from the batch (in-batch softmax); on RetailRocket it
    overfits the hashed property tokens and generalises worse to unseen items
    (held-out cosine 0.48 vs 0.69), so cosine is the default.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    tgt = torch.as_tensor(targets, dtype=torch.float32)
    losses = []
    for _ in range(epochs):
        enc.train()
        order = rng.permutation(len(idx))
        total = 0.0
        for i in range(0, len(order), batch_size):
            b = order[i : i + batch_size]
            c = enc(idx[b])
            if loss == "softmax":
                out = F.cross_entropy(c @ tgt[b].T / temperature, torch.arange(len(b)))
            else:
                out = (1 - (c * tgt[b]).sum(1)).mean()
            opt.zero_grad()
            out.backward()
            opt.step()
            total += out.item() * len(b)
        losses.append(total / len(order))
    return losses
