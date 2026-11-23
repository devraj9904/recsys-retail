"""FastAPI service.

    uvicorn recsys.serving.app:app --port 8000          # reads RECSYS_* env vars

Endpoints
    GET  /recommend/{visitor_id}?k=20&nocache=0   ranked recommendations
    POST /events                                  record a view / addtocart / transaction
    GET  /visitors/{visitor_id}                   recent events, session, top categories
    GET  /items/{item_id}                         display metadata
    GET  /demo/visitors                           sample visitors for the demo page
    GET  /health, /metrics, /                      health, Prometheus, demo page

Handlers are ``async`` on purpose: each request is a few milliseconds of
CPU work, and running it inline on the event loop avoids the threadpool hop
and GIL contention between threads. Scale out with uvicorn workers.

Events are written straight to Redis by default (``RECSYS_EVENTS=direct``),
or published to Kafka for the streaming consumer (``RECSYS_EVENTS=kafka``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from recsys.serving.metrics import Metrics
from recsys.serving.recommender import Recommender
from recsys.streaming.events import EVENT_TYPES, TOPIC, Event

STATIC = Path(__file__).parent / "static"


class EventIn(BaseModel):
    visitor_id: int = Field(ge=0)
    item_id: int = Field(ge=0)
    event: str = Field(pattern="^(view|addtocart|transaction)$")
    ts: float | None = None  # unix seconds; default: the service's data clock + 1s


def create_app(rec: Recommender, events_mode: str = "direct", producer=None) -> FastAPI:
    app = FastAPI(title="recsys-retail", version="0.5.0")
    m = Metrics()
    app.state.rec, app.state.metrics = rec, m

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t = time.perf_counter()
        resp = await call_next(request)
        route = request.scope.get("route")
        name = getattr(route, "path", "other")
        m.request.labels(name, str(resp.status_code)).observe(time.perf_counter() - t)
        return resp

    @app.get("/health")
    async def health():
        rec.store.r.ping()
        return {"status": "ok", "items": len(rec.b.item_ids), "warm_items": rec.b.n_warm,
                "events_mode": events_mode, "clock": rec.clock()}  # fmt: skip

    @app.get("/recommend/{visitor_id}")
    async def recommend(visitor_id: int, k: int = Query(20, ge=1, le=100), nocache: bool = False):
        r = rec.recommend(visitor_id, k, use_cache=not nocache)
        m.recs.labels(r.route, "hit" if r.cached else "miss").inc()
        if not r.cached:
            for stage, ms in r.timings_ms.items():
                m.stage.labels(stage).observe(ms / 1000)
        return r.to_dict()

    @app.post("/events")
    async def events(e: EventIn):
        if e.event not in EVENT_TYPES:
            raise HTTPException(422, "unknown event type")
        ts = e.ts if e.ts is not None else rec.clock() + 1
        ev = Event(e.visitor_id, e.item_id, e.event, ts, sent_at=time.time())
        if events_mode == "kafka" and producer is not None:
            producer.produce(TOPIC, key=str(ev.visitor_id), value=ev.to_bytes())
            producer.poll(0)
        else:
            rec.live.ingest([ev])
        m.events.labels(e.event, events_mode).inc()
        return {"accepted": True, "mode": events_mode, "ts": ts}

    @app.get("/visitors/{visitor_id}")
    async def visitor(visitor_id: int, n: int = Query(15, ge=1, le=90)):
        recent = rec.store.recent(visitor_id)[:n]
        cats = rec.store.category_affinity(visitor_id, now=rec.clock())
        top = sorted(cats.items(), key=lambda kv: -kv[1])[:5]
        return {
            "visitor_id": visitor_id,
            "has_vector": rec.store.user_vector(visitor_id) is not None,
            "stats": rec.store.stats(visitor_id),
            "session": rec.store.session(visitor_id),
            "top_categories": [{"leaf_code": c, "affinity": round(a, 3)} for c, a in top],
            "recent": [rec.item_info(i) | {"weight": w, "ts": ts} for i, w, ts in recent],
        }

    @app.get("/items/{item_id}")
    async def item(item_id: int):
        info = rec.item_info(item_id)
        if not info["known"]:
            raise HTTPException(404, "unknown item")
        return info

    @app.get("/demo/visitors")
    async def demo_visitors():
        return {"visitors": rec.b.demo_visitors}

    @app.get("/metrics")
    def metrics():
        registry = m.registry
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):  # several uvicorn workers
            from prometheus_client import CollectorRegistry, multiprocess

            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", response_class=HTMLResponse)
    def demo():
        return (STATIC / "demo.html").read_text()

    return app


def app_from_env() -> FastAPI:  # pragma: no cover - wiring only
    import redis
    import torch

    from recsys.config import ROOT, processed_dir

    torch.set_num_threads(int(os.environ.get("RECSYS_TORCH_THREADS", "1")))
    art = Path(os.environ.get("RECSYS_ARTIFACTS", ROOT / "artifacts"))
    client = redis.Redis.from_url(os.environ.get("RECSYS_REDIS", "redis://localhost:6379/0"))
    rec = Recommender.from_artifacts(
        client,
        serving=art / "serving",
        processed=Path(os.environ.get("RECSYS_PROCESSED", processed_dir())),
        retrieval=art / "retrieval",
        ranking=art / "ranking",
        index=os.environ.get("RECSYS_INDEX", "hnsw"),
    )
    mode = os.environ.get("RECSYS_EVENTS", "direct")
    producer = None
    if mode == "kafka":
        from recsys.streaming.producer import make_producer

        producer = make_producer(os.environ.get("RECSYS_KAFKA", "localhost:9092"))
    return create_app(rec, mode, producer)


def __getattr__(name):  # lazy: importing this module for tests must not load artifacts
    if name == "app":
        globals()["app"] = app_from_env()
        return globals()["app"]
    raise AttributeError(name)
