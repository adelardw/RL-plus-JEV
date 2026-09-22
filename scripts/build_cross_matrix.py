#!/usr/bin/env python
"""Score every trained checkpoint with every reward source.

The diagonal is what a run was trained on; the off-diagonal is what the other
judges make of the same outputs. A policy that improved only on its own judge
shows up as a large diagonal-minus-off-diagonal gap, which is what separates
"this reward source works" from "this policy learned to please this judge"
-- the failure mode that makes reward-source comparisons worth running at all.

Completions are read from each run's eval output rather than re-sampled, so
every source scores exactly the same text and sampling noise cannot masquerade
as disagreement between judges.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
import re
from pathlib import Path

from rljevf.config import RunConfig
from rljevf.evaluate.cross_matrix import normalise, overoptimisation_gap, save
from rljevf.guardrails import require_experiment_host
from rljevf.ppo import _maybe_sync
from rljevf.registry import build_reward

PROJECT = Path(__file__).resolve().parent.parent

# which source each arm optimised, for the diagonal
TRAINED_ON = {
    "R1-bert": "bert_rm", "R2-self": "self_judge", "R3-rlaif": "api_judge",
    "R4-jev": "jev", "R5-ppo-learned": "jev", "R6-ppo-jevcritic": "jev",
    "A1-sg2jev-gen": "jev", "A2-sg2jev-policy": "jev",
}


def find_completions(runs_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(runs_dir.rglob("completions.json")):
        run = f.parent.parent.name if f.parent.name == "eval" else f.parent.name
        try:
            blob = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if blob.get("completions"):
            out[run] = blob
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/kaggle/working/runs")
    ap.add_argument("--sft", default="/kaggle/working/runs/R0-sft",
                    help="the self-judge column must use the same frozen SFT copy "
                         "the self-judge arm trained against, not the base model")
    ap.add_argument("--baseline", default="R0-sft",
                    help="run whose row every cell is expressed relative to")
    ap.add_argument("--sources", default="jev,bert_rm,api_judge,self_judge")
    ap.add_argument("--limit", type=int, default=200, help="completions per run")
    ap.add_argument("--out", default="/kaggle/working/results/cross_matrix.json")
    ap.add_argument("--smoke", action="store_true",
                    help="allow running off-GPU to check the code path")
    args = ap.parse_args()
    require_experiment_host("build_cross_matrix")

    runs = find_completions(Path(args.runs))
    if not runs:
        raise SystemExit(f"no completions.json under {args.runs}")
    print(f"{len(runs)} checkpoints: {sorted(runs)}", flush=True)

    # every source must see the same prompts, so intersect on the prompt list
    base_prompts = None
    for name, blob in runs.items():
        p = tuple(blob["prompts"][: args.limit])
        if base_prompts is None:
            base_prompts = p
        elif p != base_prompts:
            print(f"WARNING: {name} has a different prompt set; "
                  f"its row is not comparable", flush=True)
    prompts = list(base_prompts)

    sources = {}
    stats = {}
    for sname in [s for s in args.sources.split(",") if s]:
        cfg = RunConfig(run_id=f"cross-{sname}", reward_source=sname,
                        algo="none", sft_checkpoint=args.sft)
        try:
            fn, src = build_reward(cfg)
            sources[sname] = fn
            stats[sname] = src
            print(f"  source ready: {sname}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  source {sname} unavailable: {type(e).__name__}: {e}", flush=True)

    matrix: dict[str, dict[str, float]] = {}
    for run, blob in sorted(runs.items()):
        comps = blob["completions"][: args.limit]
        row: dict[str, float] = {}
        for sname, fn in sources.items():
            scores = _maybe_sync(fn(prompts=prompts, completions=comps))
            row[sname] = sum(scores) / max(1, len(scores))
            print(f"  {run:22s} x {sname:12s} = {row[sname]:+.4f}", flush=True)
        matrix[run] = row

    res = {"sources": list(sources), "matrix": matrix, "baseline": args.baseline}
    if args.baseline in matrix:
        res["relative"] = normalise(matrix, args.baseline)
        res["overoptimisation"] = overoptimisation_gap(matrix, TRAINED_ON, args.baseline)
        print("\nover-optimisation gap (diag - mean off-diag, baseline-relative):",
              flush=True)
        for k, v in sorted(res["overoptimisation"].items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v:+.4f}", flush=True)
    else:
        print(f"\nno baseline row {args.baseline!r}: reporting raw scores only",
              flush=True)

    res["source_stats"] = {k: (v.stats() if v is not None else {})
                           for k, v in stats.items()}
    save(res, Path(args.out))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
