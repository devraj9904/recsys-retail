"""Tiny RetailRocket-shaped dataset for tests and offline smoke runs.

Writes the same four CSVs as the Kaggle archive, with planted structure:
visitors prefer one top-level category, funnel rates mimic the real data
(most events are views, a few carts, fewer purchases) and new items and
visitors keep appearing late in the log, so cold-start paths get exercised.

    python -m recsys.data.synthetic --out data/synthetic/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START_MS = 1_430_000_000_000  # May 2015, like RetailRocket
DAY_MS = 86_400_000


def generate(
    out: Path,
    n_visitors: int = 400,
    n_items: int = 300,
    n_days: int = 40,
    n_top: int = 5,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    # Two-level category tree: n_top roots, 3 leaves each.
    leaves = {t: [100 + t * 10 + j for j in range(3)] for t in range(n_top)}
    tree = [(t, None) for t in range(n_top)] + [
        (leaf, t) for t, ls in leaves.items() for leaf in ls
    ]
    pd.DataFrame(tree, columns=["categoryid", "parentid"]).astype(
        {"parentid": "Int64"}
    ).to_csv(out / "category_tree.csv", index=False)

    item_top = rng.integers(0, n_top, n_items)
    item_leaf = np.array([rng.choice(leaves[t]) for t in item_top])
    # Items are released over time; the last ~10% appear only in the final week.
    release_day = np.where(
        np.arange(n_items) >= int(n_items * 0.9),
        rng.integers(n_days - 7, n_days, n_items),
        rng.integers(0, n_days // 2, n_items),
    )
    item_pop = rng.zipf(1.6, n_items).clip(max=50).astype(float)

    rows = []
    for item in range(n_items):
        ts = START_MS + int(release_day[item]) * DAY_MS
        rows += [
            (ts, item, "categoryid", str(item_leaf[item])),
            (ts, item, "available", str(int(rng.random() < 0.8))),
            (ts, item, "790", f"n{rng.uniform(5, 500):.3f}"),
            (ts, item, "888", f"{rng.integers(1000, 1010)} n{rng.integers(1, 9)}.000"),
            (ts, item, "283", " ".join(str(x) for x in rng.integers(0, 50, 3))),
        ]
    props = pd.DataFrame(rows, columns=["timestamp", "itemid", "property", "value"])
    half = len(props) // 2
    props.iloc[:half].to_csv(out / "item_properties_part1.csv", index=False)
    props.iloc[half:].to_csv(out / "item_properties_part2.csv", index=False)

    events, txn = [], 0
    for v in range(n_visitors):
        pref = rng.integers(0, n_top)
        first_day = rng.integers(0, n_days)
        n_sessions = 1 + rng.poisson(1.5)
        for _ in range(n_sessions):
            day = min(n_days - 1, first_day + rng.integers(0, 10))
            t = START_MS + int(day) * DAY_MS + int(rng.integers(0, DAY_MS - 3_600_000))
            live = np.flatnonzero(release_day <= day)
            p = item_pop[live] * np.where(item_top[live] == pref, 8.0, 1.0)
            for item in rng.choice(live, size=1 + rng.poisson(2), p=p / p.sum()):
                t += int(rng.integers(5_000, 300_000))
                events.append((t, v, "view", int(item), None))
                if rng.random() < 0.12:
                    t += 20_000
                    events.append((t, v, "addtocart", int(item), None))
                    if rng.random() < 0.4:
                        t += 60_000
                        events.append((t, v, "transaction", int(item), txn))
                        txn += 1
    cols = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
    ev = pd.DataFrame(events, columns=cols)
    ev["transactionid"] = ev["transactionid"].astype("Int64")
    ev.to_csv(out / "events.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/raw"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"synthetic RetailRocket in {generate(args.out, seed=args.seed)}")


if __name__ == "__main__":
    main()
