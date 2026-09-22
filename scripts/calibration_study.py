#!/usr/bin/env python
"""Experiment 1 -- is Jev's prefix probability actually a value function?

Everything downstream of "Jev as a critic" rests on one empirical claim: that
asking Jev about a *partial* response yields a number that tracks the quality
of the *finished* response. This script measures that directly, before any RL
is run, so the critic arm is justified rather than assumed.

It reports:
  * the affine calibration V = a*p + b against the terminal rubric reward,
    with R^2 -- the map the PPO critic uses to put TD errors in reward units;
  * how early the prefix value separates eventually-good from eventually-bad
    responses (AUC at each prefix fraction) -- how much signal the critic
    carries before the response is finished, which is the whole point of
    having a critic rather than only a terminal reward;
  * monotonicity of the value curve within each group.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
from pathlib import Path

import torch

from rljevf.config import POLICY_MODEL
from rljevf.critic.jev_critic import JevCritic, fit_calibration
from rljevf.critic.judge_critic import APIJudgePrefixCritic
from rljevf.data import eval_prompts
from rljevf.evaluate.generate import load_policy, sample
from rljevf.jevclient import JevClient, Meter
from rljevf.rubric import GENERAL_RUBRIC, build_state
from rljevf.guardrails import require_experiment_host


def calibration_spread(probs: Sequence[float]) -> dict:
    """How much of [0,1] a critic actually uses.

    A critic that answers only 0 or 1 is useless *as a critic* even if it is a
    fine judge: TD errors are differences between neighbouring values, so a
    saturated value curve carries no signal in the middle of a response, which
    is exactly where credit assignment is needed.
    """
    n = len(probs)
    if not n:
        return {}
    saturated = sum(1 for p in probs if p < 0.02 or p > 0.98) / n
    mean = sum(probs) / n
    var = sum((p - mean) ** 2 for p in probs) / n
    # how many of 10 equal bins are occupied
    bins = {min(9, int(p * 10)) for p in probs}
    return {"saturated_frac": saturated, "std": var ** 0.5,
            "bins_occupied": len(bins), "mean": mean}


def auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC; ties get half credit."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx = torch.tensor(rank(x), dtype=torch.float64)
    ry = torch.tensor(rank(y), dtype=torch.float64)
    rx = (rx - rx.mean()) / rx.std().clamp_min(1e-9)
    ry = (ry - ry.mean()) / ry.std().clamp_min(1e-9)
    return float((rx * ry).mean())


async def run_critic(prompts, comps, fracs, critic, rewards, labels) -> dict:
    """The same measurement for any prefix critic, so the three are comparable."""
    by_frac: dict[float, list[float]] = {}
    for f in fracs:
        prefixes = [" ".join(c.split()[: max(0, int(len(c.split()) * f))]) for c in comps]
        by_frac[f] = await critic.aprefix_probs_batch(list(prompts), prefixes)
    cal = fit_calibration(by_frac[1.0], rewards)
    return {
        "calibration_at_full": cal.to_dict(),
        "spread_at_full": calibration_spread(by_frac[1.0]),
        "by_fraction": {
            f"{f:.2f}": {
                "mean_value": sum(by_frac[f]) / len(by_frac[f]),
                "auc_vs_final_quality": auc(by_frac[f], labels),
                "spearman_vs_final_reward": spearman(by_frac[f], rewards),
                "calibration": fit_calibration(by_frac[f], rewards).to_dict(),
                "spread": calibration_spread(by_frac[f]),
            }
            for f in fracs
        },
        "mean_value_curve_good": {
            f"{f:.2f}": sum(v for v, l in zip(by_frac[f], labels) if l == 1) / max(1, sum(labels))
            for f in fracs},
        "mean_value_curve_bad": {
            f"{f:.2f}": sum(v for v, l in zip(by_frac[f], labels) if l == 0)
            / max(1, len(labels) - sum(labels)) for f in fracs},
        "stats": critic.stats(),
    }


async def run(prompts, comps, fracs, client, critic) -> dict:
    # terminal rubric reward
    term = await client.aask_many(
        [build_state(p, c) for p, c in zip(prompts, comps)], GENERAL_RUBRIC.jev_questions()
    )
    rewards = [GENERAL_RUBRIC.scalarise_jev(t["answers"]) for t in term]
    parts = [GENERAL_RUBRIC.breakdown_jev(t["answers"]) for t in term]

    # prefix probabilities at fixed fractions of each completion (word-level,
    # so the cut points mean the same thing across responses of different length)
    by_frac: dict[float, list[float]] = {}
    for f in fracs:
        prefixes = []
        for c in comps:
            w = c.split()
            prefixes.append(" ".join(w[: max(0, int(len(w) * f))]))
        by_frac[f] = await critic.aprefix_probs_batch(prompts, prefixes)

    med = sorted(rewards)[len(rewards) // 2]
    labels = [1 if r > med else 0 for r in rewards]

    cal = fit_calibration(by_frac[1.0], rewards)
    return {
        "n": len(prompts),
        "reward_mean": sum(rewards) / len(rewards),
        "reward_median": med,
        "rubric_components": {
            k: sum(p[k] for p in parts) / len(parts) for k in GENERAL_RUBRIC.keys
        },
        "calibration_at_full": cal.to_dict(),
        "by_fraction": {
            f"{f:.2f}": {
                "mean_value": sum(by_frac[f]) / len(by_frac[f]),
                "auc_vs_final_quality": auc(by_frac[f], labels),
                "spearman_vs_final_reward": spearman(by_frac[f], rewards),
                "calibration": fit_calibration(by_frac[f], rewards).to_dict(),
            }
            for f in fracs
        },
        "mean_value_curve": {f"{f:.2f}": sum(by_frac[f]) / len(by_frac[f]) for f in fracs},
        "mean_value_curve_good": {
            f"{f:.2f}": sum(v for v, l in zip(by_frac[f], labels) if l == 1) / max(1, sum(labels))
            for f in fracs
        },
        "mean_value_curve_bad": {
            f"{f:.2f}": sum(v for v, l in zip(by_frac[f], labels) if l == 0)
            / max(1, len(labels) - sum(labels))
            for f in fracs
        },
        "raw": {"rewards": rewards, "by_fraction": {str(k): v for k, v in by_frac.items()}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=POLICY_MODEL)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--critics", default="jev,api",
                    help="comma-separated: jev, api, local -- the same prefix "
                         "measurement run with different judges")
    ap.add_argument("--out", default="results/calibration_study.json")
    ap.add_argument("--smoke", action="store_true",
                    help="allow running off-GPU to check the code path")
    args = ap.parse_args()
    require_experiment_host("calibration_study")

    ds = eval_prompts(n=args.n, seed=0)
    prompts = [ds[i]["prompt"] for i in range(len(ds))]
    model, tok, device = load_policy(args.policy)
    # Sampling dominates wall-clock here; cache it so a re-analysis is instant
    # and so the measured numbers refer to one fixed set of completions.
    cache = Path(args.out).with_suffix(".completions.json")
    if cache.exists():
        comps = json.loads(cache.read_text())["completions"]
        print(f"reusing {len(comps)} cached completions", flush=True)
    else:
        print(f"sampling {len(prompts)} completions from {args.policy} on {device}...", flush=True)
        comps = sample(model, tok, prompts, device, max_new_tokens=args.max_new_tokens,
                       greedy=False, seed=args.seed, batch_size=8)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"prompts": prompts, "completions": comps}, indent=1))
    keep = [i for i, c in enumerate(comps) if c.strip()]
    prompts = [prompts[i] for i in keep]
    comps = [comps[i] for i in keep]
    print(f"{len(comps)} non-empty completions", flush=True)

    client = JevClient(meter=Meter(name="calibration_study"))
    critic = JevCritic(tokenizer=tok, client=client)
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    res = asyncio.run(run(prompts, comps, fracs, client, critic))
    res["policy"] = args.policy
    res["cost"] = client.meter.summary()

    # The same measurement with other judges, so the paper can claim the
    # mechanism rather than one vendor's API.
    wanted = [c for c in args.critics.split(",") if c and c != "jev"]
    if wanted:
        rewards = res["raw"]["rewards"]
        med = res["reward_median"]
        labels = [1 if r > med else 0 for r in rewards]
        res["critics"] = {"jev": {
            "calibration_at_full": res["calibration_at_full"],
            "by_fraction": res["by_fraction"],
            "mean_value_curve_good": res["mean_value_curve_good"],
            "mean_value_curve_bad": res["mean_value_curve_bad"],
            "spread_at_full": calibration_spread(res["raw"]["by_fraction"]["1.0"]),
        }}
        for kind in wanted:
            try:
                if kind == "api":
                    from rljevf.config import API_JUDGE_MODEL
                    c = APIJudgePrefixCritic(tok, API_JUDGE_MODEL, concurrency=24)
                elif kind == "local":
                    from rljevf.config import JUDGE_MODEL
                    from rljevf.critic.judge_critic import LocalJudgePrefixCritic
                    from rljevf.rewards.llm_judge import LLMJudgeRewardSource
                    c = LocalJudgePrefixCritic(tok, LLMJudgeRewardSource(model_name=JUDGE_MODEL, batch_size=8))
                else:
                    continue
                print(f"running the {kind} prefix critic...", flush=True)
                res["critics"][kind] = asyncio.run(
                    run_critic(prompts, comps, fracs, c, rewards, labels))
            except Exception as e:  # noqa: BLE001
                print(f"{kind} critic FAILED: {type(e).__name__}: {e}", flush=True)
                res.setdefault("critics", {})[kind] = {"error": str(e)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    show = {k: v for k, v in res.items() if k != "raw"}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
