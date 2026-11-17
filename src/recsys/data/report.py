"""Render prepare()'s stats.json as a Markdown table for reports/ and the README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys.config import processed_dir


def render(stats: dict, title: str = "Dataset summary") -> str:
    mix = stats["event_mix"]
    total = sum(mix.values()) or 1
    lines = [
        f"## {title}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| events (after cleaning) | {stats['events']:,} |",
        f"| visitors | {stats['visitors']:,} |",
        f"| sessions | {stats['sessions']:,} |",
        f"| items with events | {stats['items_with_events']:,} |",
        f"| items with category / price | {stats['items_with_metadata']:,} / "
        f"{stats['items_with_price']:,} |",
        *(f"| {k} events | {v:,} ({v / total:.1%}) |" for k, v in mix.items()),
        f"| visitors with a single event | {stats['single_event_visitor_share']:.1%} |",
        f"| warm users / warm items (train) | {stats['warm_users']:,} / {stats['warm_items']:,} |",
        f"| test events from new visitors | {stats['test_new_visitor_share']:.1%} |",
        f"| test events on cold items | {stats['test_cold_item_share']:.1%} |",
        "",
        "| split | start | end | events | visitors | items | sessions |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in stats["splits"].items():
        lines.append(
            f"| {name} | {s['start'][:10]} | {s['end'][:10]} | {s['events']:,} | "
            f"{s['visitors']:,} | {s['items']:,} | {s['sessions']:,} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, default=processed_dir() / "stats.json")
    ap.add_argument("--out", type=Path, default=Path("reports/data_stats.md"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(json.loads(args.stats.read_text())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
