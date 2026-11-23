"""Redis online feature store, updated event by event.

Keys (per visitor ``v`` and item ``i``)::

    u:{v}:recent   LIST  newest-first "item_id:weight:ts", capped at recent_len
    u:{v}:stats    HASH  events, view, addtocart, transaction, last_ts
    u:{v}:sess     HASH  session_id, session_start, session_events, last_ts
    u:{v}:cat      HASH  leaf category -> time-decayed affinity, plus "_ts"
    u:{v}:vec      STRING float32 user embedding (two-tower user tower)
    i:{i}:stats    HASH  view, addtocart, transaction
    trending       ZSET  item_id -> summed event weight
    clock          STRING latest event time seen (data clock for serving)
    rec:{v}        HASH  cached recommendation responses, dropped on every event

Session and affinity updates depend on the previous state, so they run as
one Lua script per event: atomic, one round trip, and safe with several
consumers. Everything else is batched into a single pipeline per poll.

The same reads back the features the ranker uses offline (recent items,
counts, session, category shares), so serving sees the live state.
"""

from __future__ import annotations

import math

import numpy as np

from recsys.config import EVENT_WEIGHTS
from recsys.retrieval.data import PAD
from recsys.streaming.events import Event

_SESSION_AND_AFFINITY = """
local now = tonumber(ARGV[1])
local gap = tonumber(ARGV[2])
local leaf = ARGV[3]
local weight = tonumber(ARGV[4])
local tau = tonumber(ARGV[5])

local last = tonumber(redis.call('HGET', KEYS[1], 'last_ts') or '-1')
if last < 0 or now - last > gap then
  redis.call('HINCRBY', KEYS[1], 'session_id', 1)
  redis.call('HSET', KEYS[1], 'session_start', now, 'session_events', 0)
end
redis.call('HINCRBY', KEYS[1], 'session_events', 1)
if now > last then redis.call('HSET', KEYS[1], 'last_ts', now) end

if leaf ~= '0' then
  local cat_ts = tonumber(redis.call('HGET', KEYS[2], '_ts') or tostring(now))
  local decay = math.exp(-math.max(now - cat_ts, 0) / tau)
  local all = redis.call('HGETALL', KEYS[2])
  for j = 1, #all, 2 do
    if all[j] ~= '_ts' then
      redis.call('HSET', KEYS[2], all[j], tonumber(all[j + 1]) * decay)
    end
  end
  redis.call('HINCRBYFLOAT', KEYS[2], leaf, weight)
  redis.call('HSET', KEYS[2], '_ts', math.max(now, cat_ts))
end
return 1
"""

_MAX_CLOCK = """
local cur = tonumber(redis.call('GET', KEYS[1]) or '-1')
if tonumber(ARGV[1]) > cur then redis.call('SET', KEYS[1], ARGV[1]) end
return 1
"""


class OnlineStore:
    def __init__(
        self,
        client,
        item_idx: dict[int, int],
        leaf: np.ndarray,
        history_len: int = 30,
        recent_len: int | None = None,
        session_gap_s: int = 1800,
        affinity_half_life_days: float = 7.0,
    ):
        self.r = client
        self.item_idx = item_idx
        self.leaf = leaf
        self.history_len = history_len
        # Raw events include repeats and unknown items that the user tower drops,
        # so keep more than one window's worth.
        self.recent_len = recent_len or 3 * history_len
        self.session_gap_s = session_gap_s
        self.tau = affinity_half_life_days * 86400 / math.log(2)
        self._script = self.r.register_script(_SESSION_AND_AFFINITY)
        self._clock = self.r.register_script(_MAX_CLOCK)

    def _leaf(self, item_id: int) -> int:
        idx = self.item_idx.get(item_id)
        return int(self.leaf[idx]) if idx is not None else 0

    def apply(self, events: list[Event]) -> set[int]:
        """Write a batch of events; returns the visitors whose state changed."""
        pipe = self.r.pipeline(transaction=False)
        touched = set()
        for e in events:
            v, w = e.visitor_id, EVENT_WEIGHTS[e.event]
            touched.add(v)
            pipe.lpush(f"u:{v}:recent", f"{e.item_id}:{w}:{e.ts:.0f}")
            pipe.ltrim(f"u:{v}:recent", 0, self.recent_len - 1)
            pipe.hincrby(f"u:{v}:stats", "events", 1)
            pipe.hincrby(f"u:{v}:stats", e.event, 1)
            pipe.hset(f"u:{v}:stats", "last_ts", e.ts)
            pipe.hincrby(f"i:{e.item_id}:stats", e.event, 1)
            pipe.zincrby("trending", w, e.item_id)
            pipe.delete(f"rec:{v}")  # cached recommendations are stale now
            self._script(
                keys=[f"u:{v}:sess", f"u:{v}:cat"],
                args=[e.ts, self.session_gap_s, self._leaf(e.item_id), w, self.tau],
                client=pipe,
            )
        if events:
            self._clock(keys=["clock"], args=[max(e.ts for e in events)], client=pipe)
        pipe.execute()
        return touched

    def clock(self) -> float | None:
        raw = self.r.get("clock")
        return None if raw is None else float(raw)

    # ------------------------------------------------------------------ reads
    def recent(self, visitor: int) -> list[tuple[int, int, float]]:
        out = []
        for raw in self.r.lrange(f"u:{visitor}:recent", 0, -1):
            item, w, ts = (raw.decode() if isinstance(raw, bytes) else raw).split(":")
            out.append((int(item), int(w), float(ts)))
        return out

    def history_arrays(self, visitors: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Oldest-to-newest warm-item history, PAD-filled, for the user tower."""
        L = self.history_len
        items = np.full((len(visitors), L), PAD, dtype=np.int64)
        weights = np.zeros((len(visitors), L), dtype=np.float32)
        pipe = self.r.pipeline(transaction=False)
        for v in visitors:
            pipe.lrange(f"u:{v}:recent", 0, -1)
        for row, raws in enumerate(pipe.execute()):
            steps = []
            for raw in raws:  # newest first
                item, w, _ = (raw.decode() if isinstance(raw, bytes) else raw).split(":")
                idx = self.item_idx.get(int(item))
                if idx is None:
                    continue
                if steps and steps[-1][0] == idx:  # collapse repeats, keep strongest
                    steps[-1] = (idx, max(steps[-1][1], int(w)))
                    continue
                steps.append((idx, int(w)))
            steps = steps[:L][::-1]
            if steps:
                items[row, L - len(steps) :] = [s[0] for s in steps]
                weights[row, L - len(steps) :] = [s[1] for s in steps]
        return items, weights

    def stats(self, visitor: int) -> dict[str, float]:
        return {k.decode() if isinstance(k, bytes) else k: float(v)
                for k, v in self.r.hgetall(f"u:{visitor}:stats").items()}  # fmt: skip

    def session(self, visitor: int) -> dict[str, float]:
        return {k.decode() if isinstance(k, bytes) else k: float(v)
                for k, v in self.r.hgetall(f"u:{visitor}:sess").items()}  # fmt: skip

    def category_affinity(self, visitor: int, now: float | None = None) -> dict[int, float]:
        raw = {
            (k.decode() if isinstance(k, bytes) else k): float(v)
            for k, v in self.r.hgetall(f"u:{visitor}:cat").items()
        }
        ts = raw.pop("_ts", None)
        decay = 1.0
        if now is not None and ts is not None:
            decay = math.exp(-max(now - ts, 0) / self.tau)
        return {int(k): v * decay for k, v in raw.items()}

    def set_user_vectors(self, visitors: list[int], vecs: np.ndarray) -> None:
        pipe = self.r.pipeline(transaction=False)
        for v, vec in zip(visitors, vecs.astype(np.float32), strict=True):
            pipe.set(f"u:{v}:vec", vec.tobytes())
        pipe.execute()

    def user_vector(self, visitor: int) -> np.ndarray | None:
        raw = self.r.get(f"u:{visitor}:vec")
        return None if raw is None else np.frombuffer(raw, dtype=np.float32)

    def trending(self, k: int) -> list[int]:
        return [int(x) for x in self.r.zrevrange("trending", 0, k - 1)]
