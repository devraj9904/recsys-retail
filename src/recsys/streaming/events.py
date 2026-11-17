"""Event schema shared by the producer, the consumer and the API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

EVENT_TYPES = ("view", "addtocart", "transaction")
TOPIC = "events"


@dataclass(frozen=True)
class Event:
    visitor_id: int
    item_id: int
    event: str  # view | addtocart | transaction
    ts: float  # event time, unix seconds
    sent_at: float = 0.0  # wall clock when produced, for freshness measurement

    def __post_init__(self):
        if self.event not in EVENT_TYPES:
            raise ValueError(f"unknown event type {self.event!r}")

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> Event:
        d = json.loads(raw)
        return cls(
            visitor_id=int(d["visitor_id"]),
            item_id=int(d["item_id"]),
            event=d["event"],
            ts=float(d["ts"]),
            sent_at=float(d.get("sent_at", 0.0)),
        )
