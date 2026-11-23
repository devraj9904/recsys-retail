"""The online recommendation path, end to end.

    warm visitor:  two-tower vector from Redis -> FAISS top-100
                   + own recent items + trending  (candidates, ~120)
                   -> 24 online features -> LambdaRank + conversion blend
                   -> MMR diversity -> top k
    clicked only new items:  TF-IDF content kNN over the whole catalogue
    no events at all:        trending

Responses are cached in Redis per visitor and dropped by the online store
on that visitor's next event, so a cache hit is never staler than the last
click.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from recsys.ranking.diversity import mmr
from recsys.ranking.features import FEATURES
from recsys.serving.bundle import Bundle
from recsys.serving.online_features import visitor_features
from recsys.streaming.live import LiveRecommender

_UI_EVENTS = FEATURES.index("ui_events")
_UI_MAX = FEATURES.index("ui_max_weight")


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return (x - x.mean()) / ((0.0 if np.isnan(sd) else sd) + 1e-9)


@dataclass
class Config:
    n_tt: int = 100
    n_trend: int = 20
    blend_w: float = 0.2
    mmr_lambda: float = 0.2
    mmr_pool: int = 50
    cache_ttl_s: int = 300


@dataclass
class Result:
    visitor_id: int
    route: str
    items: list[dict]
    timings_ms: dict[str, float] = field(default_factory=dict)
    cached: bool = False
    pipeline: dict = field(default_factory=dict)  # candidate counts, for the demo / debugging

    def to_dict(self) -> dict:
        return {
            "visitor_id": self.visitor_id,
            "route": self.route,
            "cached": self.cached,
            "timings_ms": self.timings_ms,
            "pipeline": self.pipeline,
            "items": self.items,
        }


class Recommender:
    def __init__(self, bundle: Bundle, live: LiveRecommender, ranker, buyer, cfg: Config = None):
        self.b = bundle
        self.live = live
        self.store = live.store
        self.ranker = ranker
        self.buyer = buyer
        self.cfg = cfg or Config()
        self.cidx = {int(i): c for c, i in enumerate(bundle.item_ids)}
        self.item_vecs = live.index.vectors
        self.tfidf_t = bundle.tfidf.T.tocsr()

    @classmethod
    def from_artifacts(
        cls,
        client,
        serving: Path,
        processed: Path,
        retrieval: Path,
        ranking: Path,
        cfg: Config | None = None,
        index: str = "flat",
    ) -> Recommender:
        live = LiveRecommender.from_artifacts(client, processed, retrieval, index=index)
        ranker = lgb.Booster(model_file=str(ranking / "lambdarank.txt"))
        buyer = lgb.Booster(model_file=str(ranking / "purchase.txt"))
        return cls(Bundle.load(serving), live, ranker, buyer, cfg)

    # ----------------------------------------------------------------- public
    def recommend(
        self, visitor: int, k: int = 20, use_cache: bool = True, hide_purchased: bool = False
    ) -> Result:
        """Top-k items; ``hide_purchased`` drops items the visitor already bought."""
        field_ = f"{k}:{int(hide_purchased)}"
        if use_cache:
            raw = self.store.r.hget(f"rec:{visitor}", field_)
            if raw is not None:
                r = Result(**json.loads(raw))
                r.cached = True
                return r
        t0 = time.perf_counter()
        recent = self.store.recent(visitor)  # newest first: (item_id, weight, ts)
        vec = self.store.user_vector(visitor)
        if vec is not None:
            res = self._ranked(visitor, vec, recent, k, hide_purchased)
        elif recent:
            res = self._content(visitor, recent, k, hide_purchased)
        else:
            res = self._trending(visitor, k)
        res.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 3)
        if use_cache:
            key = f"rec:{visitor}"
            pipe = self.store.r.pipeline(transaction=False)
            body = res.to_dict()
            body.pop("cached")
            pipe.hset(key, field_, json.dumps(body))
            pipe.expire(key, self.cfg.cache_ttl_s)
            pipe.execute()
        return res

    def item_info(self, item_id: int) -> dict:
        c = self.cidx.get(int(item_id))
        if c is None:
            return {"item_id": int(item_id), "known": False}
        cat, price = int(self.b.category[c]), float(self.b.price[c])
        return {"item_id": int(item_id), "known": True, "warm": c < self.b.n_warm,
                "category": None if cat < 0 else cat,
                "price": None if np.isnan(price) else price}  # fmt: skip

    def clock(self) -> float:
        return max(self.store.clock() or 0.0, self.b.cutoff)

    # ----------------------------------------------------------------- routes
    def _ranked(self, visitor, vec, recent, k, hide_purchased=False) -> Result:
        cfg, T = self.cfg, {}
        t = time.perf_counter()
        _, tt = self.live.index.search(vec[None, :], cfg.n_tt)
        tt = tt[0][tt[0] >= 0]
        hist_items, _ = self.store.history_arrays([visitor])
        hist = list(dict.fromkeys(int(x) for x in hist_items[0][::-1] if x >= 0))
        trend = self.b.trending[: cfg.n_trend]
        cand = np.array(list(dict.fromkeys([*tt.tolist(), *hist, *trend.tolist()])), np.int64)
        n = len(cand)
        tt_rank = np.full(n, cfg.n_tt + 1.0)
        hist_pos = np.full(n, 99.0)
        trend_rank = np.full(n, cfg.n_trend + 1.0)
        pos = {int(c): j for j, c in enumerate(cand)}
        for r, i in enumerate(tt.tolist(), 1):
            tt_rank[pos[i]] = min(tt_rank[pos[i]], r)
        for r, i in enumerate(hist, 1):
            hist_pos[pos[i]] = min(hist_pos[pos[i]], r)
        for r, i in enumerate(trend.tolist(), 1):
            trend_rank[pos[i]] = min(trend_rank[pos[i]], r)
        retrieval = {
            "tt_score": self.item_vecs[cand] @ vec,
            "tt_rank": tt_rank, "hist_pos": hist_pos, "trend_rank": trend_rank,
            "from_tt": (tt_rank <= cfg.n_tt).astype(float),
            "from_hist": (hist_pos < 99).astype(float),
            "from_trend": (trend_rank <= cfg.n_trend).astype(float),
        }  # fmt: skip
        T["retrieval"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        ev = [(self.cidx[i], w, ts) for i, w, ts in recent
              if self.cidx.get(i, self.b.n_warm) < self.b.n_warm]  # fmt: skip
        ev_i = np.array([e[0] for e in ev], np.int64)
        ev_w = np.array([e[1] for e in ev], np.float64)
        ev_t = np.array([e[2] for e in ev], np.float64)
        clock = max(self.clock(), (ev_t.max() + 1) if len(ev_t) else 0)
        X = visitor_features(self.b, cand, retrieval, ev_i, ev_w, ev_t, clock)
        T["features"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        # One request is ~120 rows: threading costs more than it saves, and the
        # server already runs requests in parallel.
        eng = self.ranker.predict(X, num_threads=1)
        p = np.clip(self.buyer.predict(X, num_threads=1), 1e-6, 1 - 1e-6)
        score = (1 - cfg.blend_w) * _z(eng) + cfg.blend_w * _z(np.log(p / (1 - p)))
        ui_n, ui_max = X[:, _UI_EVENTS], X[:, _UI_MAX]
        bought = ui_max >= 3
        if hide_purchased:
            score = np.where(bought, -np.inf, score)
        T["ranking"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        top = np.argsort(-score, kind="stable")[: cfg.mmr_pool]
        top = top[np.isfinite(score[top])]
        order = top[mmr(score[top], self.item_vecs[cand[top]], k, cfg.mmr_lambda)]
        T["diversity"] = (time.perf_counter() - t) * 1000

        hist_arr = np.array(hist, np.int64)
        items = []
        for j in order:
            c = int(cand[j])
            if hist_pos[j] < 99:
                src = "history"
                verb = {3: "You bought this", 2: "You added this to your cart"}.get(
                    int(ui_max[j]), "You viewed this"
                )
                times = int(ui_n[j])
                reason = verb + (f" ({times}×)" if times > 1 and ui_max[j] < 3 else "")
            elif tt_rank[j] <= cfg.n_tt:
                src = "two-tower"
                reason = "Matches your recent taste"
                if len(hist_arr):
                    near = hist_arr[np.argmax(self.item_vecs[hist_arr] @ self.item_vecs[c])]
                    reason = f"Similar to item {int(self.b.item_ids[near])}"
            else:
                src = "trending"
                reason = f"Trending this week (#{int(trend_rank[j])})"
            items.append(self._item(c, float(score[j]), src, p_buy=float(p[j]), reason=reason))
        pipeline = {
            "candidates": {
                "two_tower": int(len(tt)),
                "history": int(len(hist)),
                "trending": int(len(trend)),
                "unique": int(n),
            },
            "hidden_purchased": int(bought.sum()) if hide_purchased else 0,
            "mmr_pool": int(len(top)),
            "returned": len(items),
        }
        return Result(int(visitor), "ranked", items,
                      {kk: round(v, 3) for kk, v in T.items()}, pipeline=pipeline)  # fmt: skip

    def _content(self, visitor, recent, k, hide_purchased=False) -> Result:
        t = time.perf_counter()
        ctx = list(dict.fromkeys(self.cidx[i] for i, _, _ in recent if i in self.cidx))
        if not ctx:
            return self._trending(visitor, k)
        q = self.b.tfidf[ctx].sum(0)
        scores = np.asarray(self.b.tfidf @ np.asarray(q).ravel()).ravel()
        scores[ctx] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        top = [int(c) for c in top if scores[c] > 0]
        near = {}
        if top:  # which clicked item each result is most similar to
            sims = (self.b.tfidf[ctx] @ self.b.tfidf[top].T).toarray()
            near = {c: ctx[int(np.argmax(sims[:, j]))] for j, c in enumerate(top)}
        items = [
            self._item(c, float(scores[c]), "content",
                       reason=f"Similar details to item {int(self.b.item_ids[near[c]])}")
            for c in top
        ]  # fmt: skip
        ms = round((time.perf_counter() - t) * 1000, 3)
        if len(items) < k:
            seen = {it["item_id"] for it in items} | {i for i, _, _ in recent}
            items += [x for x in self._trending(visitor, 2 * k).items
                      if x["item_id"] not in seen][: k - len(items)]  # fmt: skip
        return Result(int(visitor), "content", items, {"content_knn": ms},
                      pipeline={"context_items": len(ctx), "returned": len(items)})  # fmt: skip

    def _trending(self, visitor, k) -> Result:
        live = self.store.trending(k)
        ids = live if len(live) >= k else [int(self.b.item_ids[i]) for i in self.b.trending[:k]]
        items = [self._item(self.cidx.get(i, -1), None, "trending", item_id=i,
                            reason="Popular right now") for i in ids]  # fmt: skip
        return Result(int(visitor), "trending", items,
                      pipeline={"source": "live" if len(live) >= k else "batch"})  # fmt: skip

    def _item(self, c, score, source, p_buy=None, item_id=None, reason=None) -> dict:
        out = self.item_info(int(self.b.item_ids[c]) if item_id is None else item_id)
        out.update(score=None if score is None else round(score, 4), source=source, reason=reason)
        if p_buy is not None:
            out["p_conversion"] = round(p_buy, 4)
        return out
