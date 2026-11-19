"""Locust load test (interactive UI on :8089).

    pip install -e ".[load]"
    locust -f loadtest/locustfile.py --host http://localhost:8000

Visitor ids come from RECSYS_LOAD_VISITORS (comma-separated) or /demo/visitors.
The scripted benchmark used for reports/serving.md is recsys.serving.loadtest.
"""

import os
import random

from locust import HttpUser, between, task


class Shopper(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self):
        env = os.environ.get("RECSYS_LOAD_VISITORS")
        if env:
            self.visitors = [int(v) for v in env.split(",")]
        else:
            self.visitors = self.client.get("/demo/visitors").json()["visitors"] or [1]
        self.items = []

    @task(7)
    def recommend_returning(self):
        v = random.choice(self.visitors)
        r = self.client.get(f"/recommend/{v}?k=20", name="/recommend/[returning]")
        if r.ok:
            self.items = [i["item_id"] for i in r.json()["items"]] or self.items

    @task(1)
    def recommend_new(self):
        self.client.get(f"/recommend/{random.randrange(2_000_000_000, 2_100_000_000)}?k=20",
                        name="/recommend/[new]")  # fmt: skip

    @task(2)
    def click(self):
        if self.items:
            self.client.post("/events", json={"visitor_id": random.choice(self.visitors),
                                               "item_id": random.choice(self.items),
                                               "event": "view"})  # fmt: skip
