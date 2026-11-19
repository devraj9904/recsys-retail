"""Everything the API needs, precomputed once from the batch side.

    python -m recsys.serving.bundle     # -> artifacts/serving/

The bundle is the batch -> online hand-off of a nightly job: item tables
(features, popularity and funnel stats at the cutoff), the trending list,
content vectors for the whole catalogue, display metadata, and a list of
demo visitors. Models are loaded from the retrieval and ranking artifacts.
Item statistics are computed exactly as ``ranking.features.build_features``
computes them, so online ranking features match the offline ones.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.coldstart.catalog import build_catalog
from recsys.coldstart.content import build_content_features, tfidf_matrix
from recsys.config import ROOT, processed_dir
from recsys.models.popularity import popularity_scores
from recsys.models.two_tower import ItemFeatures, build_item_features

ITEM_STATS = ("item_pop", "item_pop_7d", "item_views", "item_cart_rate", "item_buy_rate")


def item_stats(history: pd.DataFrame, item_map: pd.DataFrame, cutoff) -> dict[str, np.ndarray]:
    """Per-item popularity and funnel features at ``cutoff`` (dense over item_idx)."""
    n = len(item_map)
    h = history[history["ts"] < cutoff].merge(item_map, on="item_id")
    ev = h["event"].astype(str).to_numpy()
    idx = h["item_idx"].to_numpy()
    w = h["weight"].to_numpy(dtype=float)
    views = np.bincount(idx, weights=(ev == "view"), minlength=n)
    carts = np.bincount(idx, weights=(ev == "addtocart"), minlength=n)
    buys = np.bincount(idx, weights=(ev == "transaction"), minlength=n)
    recent = (h["ts"] >= cutoff - pd.Timedelta(days=7)).to_numpy()
    return {
        "item_pop": np.bincount(idx, weights=w, minlength=n),
        "item_pop_7d": np.bincount(idx[recent], weights=w[recent], minlength=n),
        "item_views": views,
        "item_cart_rate": carts / (views + 5),
        "item_buy_rate": buys / (views + 5),
    }


@dataclass
class Bundle:
    item_ids: np.ndarray  # cidx -> item_id (warm prefix = item_idx)
    n_warm: int
    feats: ItemFeatures  # warm items, as the two-tower and ranker saw them
    stats: dict[str, np.ndarray]  # warm items
    trending: np.ndarray  # warm item_idx, best first
    category: np.ndarray  # cidx -> category id (-1 unknown), display only
    price: np.ndarray  # cidx -> price (nan unknown), display only
    tfidf: sp.csr_matrix  # cidx -> content vector
    demo_visitors: list[int]
    cutoff: float  # unix seconds

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "bundle.npz",
            item_ids=self.item_ids,
            leaf=self.feats.leaf, top=self.feats.top,
            price_bucket=self.feats.price_bucket, available=self.feats.available,
            trending=self.trending, category=self.category, price=self.price,
            **{f"stat_{k}": v for k, v in self.stats.items()},
        )  # fmt: skip
        sp.save_npz(out / "tfidf.npz", self.tfidf)
        meta = {"n_warm": self.n_warm, "cutoff": self.cutoff, "demo_visitors": self.demo_visitors}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, path: Path) -> Bundle:
        z = np.load(path / "bundle.npz")
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            item_ids=z["item_ids"],
            n_warm=meta["n_warm"],
            feats=ItemFeatures(z["leaf"], z["top"], z["price_bucket"], z["available"]),
            stats={k: z[f"stat_{k}"] for k in ITEM_STATS},
            trending=z["trending"],
            category=z["category"],
            price=z["price"],
            tfidf=sp.load_npz(path / "tfidf.npz").tocsr(),
            demo_visitors=meta["demo_visitors"],
            cutoff=meta["cutoff"],
        )


def build_bundle(proc: Path, n_demo: int = 40, seed: int = 0) -> Bundle:
    d = {k: pd.read_parquet(proc / f"{k}.parquet")
         for k in ("train", "tune", "test", "items", "item_map")}  # fmt: skip
    history = pd.concat([d["train"], d["tune"]], ignore_index=True)
    item_map = d["item_map"]
    cutoff = d["test"]["ts"].min()
    ids = [d["items"]["item_id"], history["item_id"], d["test"]["item_id"]]
    catalog = build_catalog(item_map, *[x.to_numpy() for x in ids])
    week = history[history["ts"] >= cutoff - pd.Timedelta(days=7)]
    trending = np.argsort(-popularity_scores(week, item_map), kind="stable")[:200]
    meta = catalog.merge(d["items"], on="item_id", how="left").sort_values("cidx")

    # Demo visitors: returning visitors with a few test-window clicks.
    hist_n = history.groupby("visitor_id").size()
    test_n = d["test"].groupby("visitor_id").size()
    both = pd.concat([hist_n.rename("h"), test_n.rename("t")], axis=1, join="inner")
    good = both[(both["h"].between(5, 60)) & (both["t"].between(3, 30))].index.to_numpy()
    rng = np.random.default_rng(seed)
    demo = (
        sorted(rng.choice(good, min(n_demo, len(good)), replace=False).tolist())
        if len(good)
        else []
    )

    return Bundle(
        item_ids=catalog["item_id"].to_numpy(),
        n_warm=len(item_map),
        feats=build_item_features(d["items"], item_map),
        stats=item_stats(history, item_map, cutoff),
        trending=trending,
        category=meta["category_id"].astype("Int64").fillna(-1).to_numpy(dtype=np.int64),
        price=meta["price"].to_numpy(dtype=float),
        tfidf=tfidf_matrix(build_content_features(d["items"], catalog)),
        demo_visitors=[int(v) for v in demo],
        cutoff=float(cutoff.value / 1e9),
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--processed", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "serving")
    a = p.parse_args(argv)
    b = build_bundle(a.processed or processed_dir())
    b.save(a.out)
    print(f"wrote {a.out}: {len(b.item_ids):,} items ({b.n_warm:,} warm), "
          f"{len(b.demo_visitors)} demo visitors")  # fmt: skip


if __name__ == "__main__":
    main()
