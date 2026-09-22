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

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import torch

from rljevf.data import preference_pairs
from rljevf.jevclient import JevClient, Meter
from rljevf.rubric import GENERAL_RUBRIC, build_state


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
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--out", default="results/judge_benchmark.json")
    args = ap.parse_args()

    ds = preference_pairs(n=args.n * 2, seed=7)
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
        }
        print("jev", {k: round(v, 3) for k, v in out["sources"]["jev"].items() if isinstance(v, float)}, flush=True)

    if args.bert_rm:
        from rljevf.rewards.bert_rm import BertRMRewardSource
        t = time.perf_counter()
        rm = BertRMRewardSource(args.bert_rm, batch_size=8)
        sc = rm(prompts, chosen)
        sr = rm(prompts, rejected)
        out["sources"]["bert_rm"] = {**agreement(sc, sr), "wall_s": round(time.perf_counter() - t, 1)}
        print("bert_rm", {k: round(v, 3) for k, v in out["sources"]["bert_rm"].items() if isinstance(v, float)}, flush=True)
        del rm
        import gc; gc.collect()

    from rljevf.rewards.llm_judge import LLMJudgeRewardSource
    for m in [x for x in args.judges.split(",") if x]:
        try:
            t = time.perf_counter()
            j = LLMJudgeRewardSource(model_name=m, batch_size=4)
            sc = j(prompts=prompts, completions=chosen)
            sr = j(prompts=prompts, completions=rejected)
            key = "llm_judge:" + m.split("/")[-1]
            out["sources"][key] = {**agreement(sc, sr), "wall_s": round(time.perf_counter() - t, 1)}
            print(key, {k: round(v, 3) for k, v in out["sources"][key].items() if isinstance(v, float)}, flush=True)
            del j
            import gc; gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"{m} FAILED: {type(e).__name__}: {e}", flush=True)
            out["sources"]["llm_judge:" + m.split("/")[-1]] = {"error": str(e)}

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                      for k, v in out["sources"].items()}, indent=1))


if __name__ == "__main__":
    main()
