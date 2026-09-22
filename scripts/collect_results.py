#!/usr/bin/env python
"""Turn run artefacts into the JSON the paper's tables are generated from.

Run directories are grouped by arm (``<arm>_seed<k>``), so seeds aggregate
automatically and a missing seed simply widens the error bar instead of
silently disappearing.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"
RESULTS = PROJECT / "results"

ARM_LABELS = {
    "R0-sft": ("baseline", "SFT (reference)"),
    "R1-bert": ("grpo", "BERT RM (DeBERTa BT)"),
    "R2-self": ("grpo", "Self-judge (frozen SFT)"),
    "R3-rlaif": ("grpo", "LLM judge (RLAIF)"),
    "R4-jev": ("grpo", "\\jev{}"),
    "R5-ppo-learned": ("ppo", "Learned value head"),
    "R6-ppo-jevcritic": ("ppo", "\\jev{} prefix critic"),
    "A1-sg2jev-gen": ("sg2jev", "Generated per prompt"),
    "A2-sg2jev-policy": ("sg2jev", "Proposed by the policy"),
    "A3-selfcert": ("grpo", "Self-certainty (no judge)"),
}
TRAINED_ON = {
    "R1-bert": "bert_rm", "R2-self": "self_judge",
    "R3-rlaif": "llm_judge", "R4-jev": "jev",
    "R5-ppo-learned": "jev", "R6-ppo-jevcritic": "jev",
}


def discover() -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for d in sorted(RUNS.glob("*_seed*")):
        m = re.match(r"(.+)_seed(\d+)$", d.name)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        ev = d / "eval" / "eval.json"
        if not ev.exists():
            ev = d / "eval.json"
        if not ev.exists():
            continue
        rec = json.loads(ev.read_text())
        rec["seed"] = seed
        for extra in ("run_stats.json", "reward_source_stats.json", "calibration.json"):
            p = d / extra
            if p.exists():
                rec[extra.replace(".json", "")] = json.loads(p.read_text())
        arms.setdefault(arm, []).append(rec)
    return arms


def group(arms, kind: str) -> dict:
    out = {"arms": []}
    for name, runs in arms.items():
        k, label = ARM_LABELS.get(name, (None, name))
        if k != kind:
            continue
        out["arms"].append({"name": name, "label": label,
                            "seeds": sorted(runs, key=lambda r: r["seed"])})
    out["arms"].sort(key=lambda a: a["name"])
    return out


def costs(arms) -> dict:
    rows = []
    for name, runs in sorted(arms.items()):
        usd = calls = 0.0
        p50 = p95 = None
        gpu = 0.0
        for r in runs:
            for blob in (r.get("reward_source_stats"), (r.get("run_stats") or {}).get("reward_source"),
                         (r.get("run_stats") or {}).get("critic")):
                if isinstance(blob, dict):
                    usd += blob.get("cost_usd", 0.0)
                    calls += blob.get("calls", 0) or blob.get("judge_forward_passes", 0) or 0
                    p50 = blob.get("latency_p50_s", p50)
                    p95 = blob.get("latency_p95_s", p95)
            gpu += r.get("gpu_hours", 0.0)
        n = max(1, len(runs))
        rows.append({"arm": ARM_LABELS.get(name, (None, name))[1], "calls": int(calls / n),
                     "usd": usd / n, "p50_s": p50, "p95_s": p95,
                     "gpu_hours": gpu / n if gpu else None})
    return {"rows": rows}


def measure_jev_properties() -> dict:
    """Live measurement so the paper quotes our own numbers."""
    from rljevf.jevclient import JevClient, Meter
    from rljevf.rubric import GENERAL_RUBRIC, build_state

    c = JevClient(use_cache=False, concurrency=32, meter=Meter(name="jev_props"))
    states = [build_state(f"Explain concept {i} clearly in three sentences.",
                          "It is a thing that does something, in a way that matters, for reasons. " * 4)
              for i in range(32)]
    t = time.perf_counter()
    c.ask_many(states, GENERAL_RUBRIC.jev_questions())
    wall = time.perf_counter() - t
    s = c.meter.summary()
    return {"rows": [
        ("Model", "\\texttt{typesafe/jev-1.13}"),
        ("Access", "OpenRouter \\texttt{/api/alpha/decisions}"),
        ("Question types", "\\textsc{noul}, \\textsc{choice}, \\textsc{score}"),
        ("Output", "typed values with probabilities; no text"),
        ("Context", "32{,}000 tokens"),
        ("Measured cost", f"\\${s['cost_usd']/max(1,s['calls'])*1000:.4f} per 1{{,}}000 calls "
                          f"(5-question rubric)"),
        ("Measured latency p50 / p95", f"{s['latency_p50_s']:.2f}\\,s / {s['latency_p95_s']:.2f}\\,s"),
        ("Measured throughput", f"{32/wall:.1f} calls/s at concurrency 32"),
        ("Mean input tokens per call", f"{s['input_tokens']/max(1,s['calls']):.0f}"),
    ], "raw": s, "wall_s": wall}


def setup_rows() -> dict:
    from rljevf.config import (BERT_RM_MODEL, EVAL_JUDGE_MODEL, JEV_MODEL,
                               JUDGE_MODEL, POLICY_MODEL)
    import scripts.run_study as rs  # noqa
    return {"rows": [
        ("Policy", f"\\texttt{{{POLICY_MODEL}}}"),
        ("SFT data", "UltraFeedback (binarized), chosen responses"),
        ("RL prompts", "UltraFeedback \\texttt{train\\_prefs}, fixed order"),
        ("Held-out prompts", "UltraFeedback \\texttt{test\\_prefs}, 200"),
        ("Ground-truth task", "GSM8K, 250 test problems"),
        ("BERT RM", f"\\texttt{{{BERT_RM_MODEL}}}"),
        ("RLAIF judge", f"\\texttt{{{JUDGE_MODEL}}}, read from logits"),
        ("Self-judge", "frozen copy of the SFT checkpoint"),
        ("\\jev{}", f"\\texttt{{{JEV_MODEL}}}"),
        ("Evaluation judge", f"\\texttt{{{EVAL_JUDGE_MODEL}}} (unrelated to every training judge)"),
        ("GRPO", f"$G={rs.GRPO_G}$, {rs.GRPO_PROMPTS} prompts/step, {rs.GRPO_STEPS} steps, "
                 f"lr $10^{{-6}}$, $\\beta=0.04$"),
        ("PPO", f"{rs.PPO_PROMPTS} prompts/step, {rs.PPO_STEPS} steps, lr $10^{{-6}}$, "
                f"KL coef 0.05, GAE $\\lambda=0.95$"),
        ("Reward-call budget", f"{rs.GRPO_CALLS:,} per GRPO run (matched across sources)"),
        ("Decoding", "temperature 1.0, top-$p$ 1.0, max 384 new tokens"),
        ("Seeds", "3 per core arm"),
        ("Hardware", "2$\\times$NVIDIA T4 (16\\,GB), Kaggle"),
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-live", action="store_true", help="do not call Jev for the properties table")
    ap.add_argument("--force-live", action="store_true",
                    help="measure Jev from here even without a GPU (not for the paper)")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    arms = discover()
    print(f"found {len(arms)} arms: {sorted(arms)}")

    (RESULTS / "grpo_main.json").write_text(json.dumps(group(arms, "grpo"), indent=1))
    (RESULTS / "ppo_main.json").write_text(json.dumps(group(arms, "ppo"), indent=1))
    (RESULTS / "sg2jev.json").write_text(json.dumps(group(arms, "sg2jev"), indent=1))
    (RESULTS / "cost.json").write_text(json.dumps(costs(arms), indent=1))
    (RESULTS / "setup.json").write_text(json.dumps(setup_rows(), indent=1))
    # Latency and throughput depend on where they are measured, so the live
    # probe only runs on an experiment host; elsewhere the table stays a
    # placeholder rather than quietly acquiring a laptop's network round-trip.
    if not args.skip_live:
        import torch

        if torch.cuda.is_available() or args.force_live:
            (RESULTS / "jev_properties.json").write_text(
                json.dumps(measure_jev_properties(), indent=1))
        else:
            print("skipping the live Jev measurement: not on an experiment host "
                  "(pass --force-live to override)")
    print("wrote results/*.json")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT))
    main()
