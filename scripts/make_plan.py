#!/usr/bin/env python
"""Turn the experiment matrix into session-sized job plans.

Scheduling by guesswork is how a study discovers in week three that it never
fit. This reads the measured seconds-per-step from the GPU benchmark, converts
each arm into an estimated runtime, and packs arms into sessions that respect
both the 10.5h session budget and the weekly GPU quota. If no benchmark is
available it says so and uses a conservative default rather than pretending.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import run_study  # noqa: E402

DEFAULT_S_PER_STEP = {"grpo": 12.0, "ppo": 20.0}   # conservative fallback
EVAL_SECONDS = 900                                  # per checkpoint


def measured_s_per_step(bench: Path) -> dict[str, float]:
    if not bench.exists():
        print(f"no benchmark at {bench}; using conservative defaults "
              f"{DEFAULT_S_PER_STEP}", flush=True)
        return dict(DEFAULT_S_PER_STEP)
    d = json.loads(bench.read_text())
    out = dict(DEFAULT_S_PER_STEP)
    for k, v in d.items():
        if not isinstance(v, dict) or "s_per_step" not in v:
            continue
        if k.startswith("grpo"):
            out["grpo"] = min(out.get("grpo", 1e9), v["s_per_step"])
        elif k.startswith("ppo"):
            out["ppo"] = v["s_per_step"]
    print(f"measured seconds/step: {out}", flush=True)
    return out


def arm_seconds(arm, sps: dict[str, float]) -> int:
    if arm.name == "R0-sft":
        return 2400
    steps = run_study.PPO_STEPS if "ppo" in arm.name else run_study.GRPO_STEPS
    key = "ppo" if "ppo" in arm.name else "grpo"
    # the Jev critic adds a round trip per probe batch that the benchmark,
    # which uses a dummy reward, does not capture
    overhead = 1.35 if "jevcritic" in arm.name else (1.15 if "jev" in arm.name else 1.0)
    return int(steps * sps[key] * overhead) + 600      # + model load and save


def build(sps, session_seconds, reserve_seconds, budget_usd, with_eval):
    jobs = []
    for arm in run_study.ARMS:
        est = arm_seconds(arm, sps)
        for slug, cmd in arm.commands():
            name = slug.replace("rljevf-", "")
            timeout = int(est * 1.8)
            jobs.append({
                "name": name,
                "command": (f"scripts/guard.py --name {name} --timeout {timeout} "
                            f"--silence-timeout 1800 --heartbeat 180 -- {cmd}"),
                "est_seconds": est,
                "needs_api": ("jev" in arm.name) or ("sg2jev" in arm.name),
                "why": arm.note,
            })
            if with_eval and arm.name != "R0-sft":
                jobs.append({
                    "name": f"eval-{name}",
                    "command": (
                        f"scripts/guard.py --name eval-{name} --timeout {EVAL_SECONDS*2} "
                        f"--silence-timeout 1200 --heartbeat 120 -- "
                        f"scripts/evaluate_run.py --run-id {name} "
                        f"--checkpoint /kaggle/working/runs/{name}/final "
                        f"--baseline {run_study.SFT} "
                        f"--out /kaggle/working/runs/{name}/eval"),
                    "est_seconds": EVAL_SECONDS,
                    "needs_api": True,
                    "why": "win-rate, GSM8K, hacking detectors for this checkpoint",
                })
    if with_eval:
        # Last: score every checkpoint with every source. Needs all the eval
        # completions to exist, so it can only run after the arms above.
        jobs.append({
            "name": "cross-reward-matrix",
            "command": ("scripts/guard.py --name cross-matrix --timeout 3600 "
                        "--silence-timeout 1200 --heartbeat 120 -- "
                        "scripts/build_cross_matrix.py --runs /kaggle/working/runs "
                        "--out /kaggle/working/results/cross_matrix.json"),
            "est_seconds": 1800,
            "needs_api": True,
            "why": "separates 'this reward source works' from 'this policy learned "
                   "to please this judge'",
        })
    return jobs


def pack(jobs, session_seconds, reserve_seconds):
    """Greedy packing into sessions, preserving order (arms depend on R0)."""
    sessions, cur, used = [], [], 0.0
    usable = session_seconds - reserve_seconds
    for j in jobs:
        if cur and used + j["est_seconds"] > usable:
            sessions.append(cur)
            cur, used = [], 0.0
        cur.append(j)
        used += j["est_seconds"]
    if cur:
        sessions.append(cur)
    return sessions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="results/gpu_benchmark.json")
    ap.add_argument("--session-seconds", type=int, default=37800)
    ap.add_argument("--reserve-seconds", type=int, default=1200)
    ap.add_argument("--budget-usd", type=float, default=25.0)
    ap.add_argument("--quota-hours", type=float, default=30.0)
    ap.add_argument("--no-eval", dest="with_eval", action="store_false", default=True)
    ap.add_argument("--out-dir", default="jobs")
    args = ap.parse_args()

    sps = measured_s_per_step(PROJECT / args.benchmark)
    jobs = build(sps, args.session_seconds, args.reserve_seconds,
                 args.budget_usd, args.with_eval)
    sessions = pack(jobs, args.session_seconds, args.reserve_seconds)

    total_h = sum(j["est_seconds"] for j in jobs) / 3600
    print(f"\n{len(jobs)} jobs, {total_h:.1f} GPU-h estimated, "
          f"{len(sessions)} session(s) of up to {args.session_seconds/3600:.1f}h")
    if total_h > args.quota_hours:
        print(f"WARNING: {total_h:.1f}h exceeds the {args.quota_hours:.0f}h weekly "
              f"quota -- it will span more than one quota window")

    out_dir = PROJECT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    for i, sess in enumerate(sessions, 1):
        plan = {
            "name": f"study-session{i}",
            "budget_usd": args.budget_usd,
            "session_seconds": args.session_seconds,
            "reserve_seconds": args.reserve_seconds,
            "pip": ["trl==1.13.0", "transformers==5.17.0", "datasets==5.0.1", "httpx"],
            "done": [],
            "jobs": sess,
        }
        p = out_dir / f"study_session{i}.json"
        p.write_text(json.dumps(plan, indent=1))
        h = sum(j["est_seconds"] for j in sess) / 3600
        print(f"  {p.name}: {len(sess):2d} jobs, ~{h:.1f}h")
        for j in sess:
            print(f"      {j['name']:26s} ~{j['est_seconds']/60:5.0f} min")


if __name__ == "__main__":
    main()
