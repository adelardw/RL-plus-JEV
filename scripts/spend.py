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


def authoritative() -> dict | None:
    """What the provider says this key has spent.

    The local meters only see sessions whose artifacts were fetched, so they
    undercount whenever a session is cancelled or its output not collected --
    they read $0.98 against the provider's $2.55. The provider is the source of
    truth; the meters remain useful for attributing spend to a job.
    """
    try:
        import httpx

        import rljevf  # noqa: F401  -- loads the local .env

        import os

        key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not key:
            return None
        h = {"Authorization": f"Bearer {key}"}
        k = httpx.get("https://openrouter.ai/api/v1/key", headers=h, timeout=30).json()["data"]
        c = httpx.get("https://openrouter.ai/api/v1/credits", headers=h, timeout=30).json()["data"]
        return {
            "key_usage": float(k.get("usage", 0.0)),
            "account_total": float(c.get("total_credits", 0.0)),
            "account_used": float(c.get("total_usage", 0.0)),
            "account_remaining": float(c.get("total_credits", 0.0)) - float(c.get("total_usage", 0.0)),
        }
    except Exception as e:  # noqa: BLE001
        print(f"  (provider unreachable: {type(e).__name__})")
        return None


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
    prov = authoritative()
    for name, v in sorted(per.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"  {name:44s} ${v:.4f}")
    print(f"\n  {'local meters (fetched sessions only)':44s} ${total:.4f}")
    if prov:
        print(f"  {'this key, per the provider':44s} ${prov['key_usage']:.4f}")
        print(f"  {'account credits remaining':44s} ${prov['account_remaining']:.4f}")
        if prov["key_usage"] > total * 1.2:
            print(f"  {'':44s} (the meters undercount: a session whose "
                  f"artifacts were never fetched still spent)")
        remaining = prov["account_remaining"]
    else:
        remaining = args.budget - total
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
