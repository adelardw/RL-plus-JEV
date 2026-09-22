#!/usr/bin/env python
"""How much of the local judge's disadvantage is configuration?

A wall-clock comparison between a remote judge at high concurrency and a local
one at a small batch size measures the two deployments, not the two models. This
sweeps the local judge's batch size to separate the part that tuning can fix
from the part that cannot, and records the peak VRAM at each setting -- because
during RL that memory is taken from the policy, which is the cost that no amount
of batching removes.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path

import torch

from rljevf.data import preference_pairs
from rljevf.guardrails import require_experiment_host
from rljevf.jevclient import JevClient, Meter
from rljevf.rubric import GENERAL_RUBRIC, build_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--batch-sizes", default="4,8,16,32")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrencies", default="8,16,32,48")
    ap.add_argument("--out", default="/kaggle/working/results/judge_throughput.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    require_experiment_host("judge_throughput")

    ds = preference_pairs(n=args.n * 2, seed=3, split="test_prefs")
    rows = [ds[i] for i in range(len(ds))][: args.n]
    prompts = [r["prompt"] for r in rows]
    comps = [r["chosen"] for r in rows]
    res = {"n": len(prompts), "model": args.model, "local": {}, "jev": {}}

    from rljevf.rewards.llm_judge import LLMJudgeRewardSource

    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            j = LLMJudgeRewardSource(model_name=args.model, batch_size=bs)
            t = time.perf_counter()
            j(prompts=prompts, completions=comps)
            dt = time.perf_counter() - t
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            res["local"][str(bs)] = {
                "wall_s": round(dt, 1),
                "judgements_per_s": round(len(prompts) / dt, 2),
                "peak_vram_gb": round(peak, 2),
            }
            print(f"  local bs={bs:3d}: {len(prompts)/dt:6.2f} judgements/s, "
                  f"peak {peak:.2f} GB", flush=True)
            del j
            import gc; gc.collect()
        except Exception as e:  # noqa: BLE001
            res["local"][str(bs)] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  local bs={bs}: {type(e).__name__}", flush=True)

    states = [build_state(p, c) for p, c in zip(prompts, comps)]
    for conc in [int(c) for c in args.concurrencies.split(",")]:
        c = JevClient(use_cache=False, concurrency=conc,
                      meter=Meter(name=f"throughput_jev_{conc}"))
        t = time.perf_counter()
        c.ask_many(states, GENERAL_RUBRIC.jev_questions())
        dt = time.perf_counter() - t
        s = c.meter.summary()
        res["jev"][str(conc)] = {
            "wall_s": round(dt, 1),
            "judgements_per_s": round(len(states) / dt, 2),
            "cost_usd": s["cost_usd"], "p50_s": s["latency_p50_s"],
            "peak_vram_gb": 0.0,
        }
        print(f"  jev conc={conc:3d}: {len(states)/dt:6.2f} judgements/s, "
              f"${s['cost_usd']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
