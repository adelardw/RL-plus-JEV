#!/usr/bin/env python
"""Total spend across every session, and what a plan's cap should therefore be.

The in-session cap sums the meters it can see, but a Kaggle session starts with
an empty working directory, so it cannot know what earlier sessions spent. The
cumulative figure lives here, on the side that persists: every fetched session's
meters plus the local ones. A plan's `budget_usd` is then set to what is
actually left, not to the original budget.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def totals() -> tuple[float, dict[str, float]]:
    per: dict[str, float] = {}
    seen: set[tuple[str, int]] = set()
    for root in (PROJECT / ".cache", PROJECT / "runs"):
        if not root.exists():
            continue
        for f in root.rglob("spend_*.json"):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            name, calls = str(d.get("name", f.stem)), int(d.get("calls", 0))
            # the same meter fetched twice must not be counted twice
            if (name, calls) in seen:
                continue
            seen.add((name, calls))
            per[name] = per.get(name, 0.0) + float(d.get("cost_usd", 0.0))
    return sum(per.values()), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=29.0,
                    help="what remains on the account")
    ap.add_argument("--set-plan", default=None,
                    help="write the remaining budget into this plan's budget_usd")
    args = ap.parse_args()

    total, per = totals()
    for name, v in sorted(per.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"  {name:44s} ${v:.4f}")
    remaining = args.budget - total
    print(f"\n  {'spent across all sessions':44s} ${total:.4f}")
    print(f"  {'remaining of ${:.2f}'.format(args.budget):44s} ${remaining:.4f}")

    if args.set_plan:
        p = Path(args.set_plan)
        d = json.loads(p.read_text())
        # leave a margin so a run cannot take the account to zero
        cap = round(max(0.0, remaining * 0.9), 2)
        d["budget_usd"] = cap
        p.write_text(json.dumps(d, indent=1))
        print(f"\n  {p.name}: budget_usd set to ${cap:.2f} "
              f"(90% of what remains, so a run cannot drain the account)")


if __name__ == "__main__":
    main()
