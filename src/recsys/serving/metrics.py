"""Prometheus metrics exposed at /metrics."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

BUCKETS = (0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1)


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.request = Histogram("recsys_request_seconds", "HTTP request latency",
                                 ["endpoint", "status"], buckets=BUCKETS, registry=r)  # fmt: skip
        self.stage = Histogram("recsys_stage_seconds", "Recommendation pipeline stage latency",
                               ["stage"], buckets=BUCKETS, registry=r)  # fmt: skip
        self.recs = Counter("recsys_recommendations_total", "Recommendation responses",
                            ["route", "cache"], registry=r)  # fmt: skip
        self.events = Counter("recsys_events_total", "Events received", ["event", "mode"],
                              registry=r)  # fmt: skip
