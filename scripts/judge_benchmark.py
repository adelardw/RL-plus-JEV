#!/usr/bin/env python
"""Experiment 0 -- how good is each reward source, before any RL?

Every arm of this study is only as meaningful as its judge. A reward source that
cannot tell a preferred response from a dispreferred one will not teach a policy
anything, and comparing against such a source would be comparing against a straw
man. So we first measure each source on held-out human preference pairs:
given (prompt, chosen, rejected), does the source score `chosen` higher?

This also fixes the RLAIF judge size honestly -- we pick the smallest judge that
is actually competent, and report the curve rather than asserting a choice.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Sequence

import torch

from rljevf.data import preference_pairs
from rljevf.jevclient import JevClient, Meter
from rljevf.rubric import GENERAL_RUBRIC, build_state
from rljevf.guardrails import require_experiment_host


def scored_in_blocks(fn, prompts, comps, label: str, block: int = 50) -> list[float]:
    """Score in blocks and report progress.

    The guard aborts a job that has printed nothing for too long, which is the
    right default for a hung process and the wrong one for a 7B judge that
    works silently for forty minutes. Long loops must say they are alive.
    """
    out: list[float] = []
    n = len(prompts)
    t0 = time.perf_counter()
    for i in range(0, n, block):
        out.extend(fn(prompts=prompts[i:i + block], completions=comps[i:i + block]))
        done = min(i + block, n)
        rate = done / max(1e-9, time.perf_counter() - t0)
        print(f"    {label}: {done}/{n} ({rate:.1f}/s, "
              f"eta {(n - done) / max(rate, 1e-9) / 60:.1f} min)", flush=True)
    return out



def per_pair_correct(chosen: Sequence[float], rejected: Sequence[float]) -> list[bool]:
    """Per-item correctness, kept so judges can be compared *paired* rather
    than as two independent proportions (see rljevf/stats.py)."""
    return [c > r for c, r in zip(chosen, rejected)]


def agreement(chosen: list[float], rejected: list[float]) -> dict:
    n = len(chosen)
    wins = sum(c > r for c, r in zip(chosen, rejected))
    ties = sum(c == r for c, r in zip(chosen, rejected))
    margins = [c - r for c, r in zip(chosen, rejected)]
    mean = sum(margins) / n
    sd = (sum((m - mean) ** 2 for m in margins) / max(1, n - 1)) ** 0.5
    return {
        "accuracy": (wins + 0.5 * ties) / n,
        "n": n,
        "mean_margin": mean,
        "margin_sd": sd,
        # effect size: how many s.d. of margin separates chosen from rejected
        "cohens_d": mean / sd if sd > 0 else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--judges", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--bert-rm", default="OpenAssistant/reward-model-deberta-v3-large-v2")
    ap.add_argument("--skip-jev", action="store_true")
    ap.add_argument("--api-judges", default="deepseek/deepseek-v4-flash-0731",
                    help="comma-separated hosted judges read via logprobs")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--split", default="test_prefs",
                    choices=["test_prefs", "train_prefs"],
                    help="judges are measured on the held-out split")
    ap.add_argument("--out", default="results/judge_benchmark.json")
    ap.add_argument("--smoke", action="store_true",
                    help="allow running off-GPU to check the code path")
    args = ap.parse_args()
    require_experiment_host("judge_benchmark")

    # test_prefs: disjoint from the prompts the policies train on
    ds = preference_pairs(n=args.n * 2, seed=7, split=args.split)
    rows = [ds[i] for i in range(len(ds))]
    rows = [r for r in rows if len(r["chosen"]) < args.max_chars and len(r["rejected"]) < args.max_chars]
    rows = rows[: args.n]
    prompts = [r["prompt"] for r in rows]
    chosen = [r["chosen"] for r in rows]
    rejected = [r["rejected"] for r in rows]
    print(f"{len(rows)} preference pairs", flush=True)

    out: dict = {"n": len(rows), "sources": {}}

    if not args.skip_jev:
        client = JevClient(meter=Meter(name="judge_benchmark_jev"))
        t = time.perf_counter()
        qs = GENERAL_RUBRIC.jev_questions()
        both = asyncio.run(client.aask_many(
            [build_state(p, c) for p, c in zip(prompts, chosen)]
            + [build_state(p, r) for p, r in zip(prompts, rejected)], qs))
        sc = [GENERAL_RUBRIC.scalarise_jev(x["answers"]) for x in both[: len(rows)]]
        sr = [GENERAL_RUBRIC.scalarise_jev(x["answers"]) for x in both[len(rows):]]
        out["sources"]["jev"] = {
            **agreement(sc, sr), "wall_s": round(time.perf_counter() - t, 1),
            "cost": client.meter.summary(),
            "correct": per_pair_correct(sc, sr),
        }
        print("jev", {k: round(v, 3) for k, v in out["sources"]["jev"].items() if isinstance(v, float)}, flush=True)

    for mid in [x for x in args.api_judges.split(",") if x]:
        try:
            from rljevf.rewards.api_judge import APIJudgeRewardSource
            t = time.perf_counter()
            aj = APIJudgeRewardSource(mid, concurrency=32)
            sc = asyncio.run(aj(prompts=prompts, completions=chosen))
            sr = asyncio.run(aj(prompts=prompts, completions=rejected))
            key = "api_judge:" + mid.split("/")[-1]
            st = aj.stats()
            out["sources"][key] = {
                **agreement(sc, sr), "wall_s": round(time.perf_counter() - t, 1),
                "cost_usd": st["cost_usd"], "calls": st["calls"],
                "unreadable_fraction": st["unreadable_fraction"],
                "correct": per_pair_correct(sc, sr),
            }
            print(key, {k: round(v, 4) for k, v in out["sources"][key].items()
                        if isinstance(v, float)}, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{mid} FAILED: {type(e).__name__}: {e}", flush=True)
            out["sources"]["api_judge:" + mid.split("/")[-1]] = {"error": str(e)}

    if args.bert_rm:
        from rljevf.rewards.bert_rm import BertRMRewardSource
        t = time.perf_counter()
        rm = BertRMRewardSource(args.bert_rm, batch_size=8)
        sc = scored_in_blocks(lambda prompts, completions: rm(prompts, completions),
                               prompts, chosen, "bert_rm chosen", block=100)
        sr = scored_in_blocks(lambda prompts, completions: rm(prompts, completions),
                               prompts, rejected, "bert_rm rejected", block=100)
        out["sources"]["bert_rm"] = {**agreement(sc, sr), "wall_s": round(time.perf_counter() - t, 1),
                                     "correct": per_pair_correct(sc, sr)}
        print("bert_rm", {k: round(v, 3) for k, v in out["sources"]["bert_rm"].items() if isinstance(v, float)}, flush=True)
        del rm
        import gc; gc.collect()

    from rljevf.rewards.llm_judge import LLMJudgeRewardSource
    for m in [x for x in args.judges.split(",") if x]:
        try:
            t = time.perf_counter()
            j = LLMJudgeRewardSource(model_name=m, batch_size=4)
            sc = scored_in_blocks(j, prompts, chosen, f"{m.split('/')[-1]} chosen")
            sr = scored_in_blocks(j, prompts, rejected, f"{m.split('/')[-1]} rejected")
            key = "llm_judge:" + m.split("/")[-1]
            out["sources"][key] = {**agreement(sc, sr), "wall_s": round(time.perf_counter() - t, 1),
                                   "correct": per_pair_correct(sc, sr)}
            print(key, {k: round(v, 3) for k, v in out["sources"][key].items() if isinstance(v, float)}, flush=True)
            del j
            import gc; gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"{m} FAILED: {type(e).__name__}: {e}", flush=True)
            out["sources"]["llm_judge:" + m.split("/")[-1]] = {"error": str(e)}

    # paired comparisons: every source scored the same pairs, so use that
    from rljevf.stats import paired_compare, required_n

    named = {k: v["correct"] for k, v in out["sources"].items() if "correct" in v}
    pairs = {}
    keys = sorted(named, key=lambda k: -sum(named[k]))
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            r = paired_compare(named[ka], named[kb], n_boot=5000)
            pairs[f"{ka} vs {kb}"] = {**r.to_dict(), "verdict": r.verdict()}
            print(f"  {ka} vs {kb}: {r.acc_a:.3f} vs {r.acc_b:.3f} "
                  f"diff {r.diff:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}] {r.verdict()}",
                  flush=True)
    out["paired"] = pairs
    out["power_note"] = {
        "required_n_rho0.0": required_n(0.65, 0.60, 0.0),
        "required_n_rho0.5": required_n(0.65, 0.60, 0.5),
        "comment": "paired items needed to resolve a 5-point accuracy gap at 80% power",
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                      for k, v in out["sources"].items()}, indent=1))


if __name__ == "__main__":
    main()
