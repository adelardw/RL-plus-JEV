#!/usr/bin/env python
"""How wrong is a sparsely-probed critic, and how fast does that shrink with K?

The critic queries the judge at K prefix positions and interpolates between
them, so it is a biased estimate of

    V*(s_t) = E[R | prompt, tokens_<t].

Two consequences are worth stating precisely, because they decide how to choose
K and what the resulting advantages actually mean.

**Credit assignment is piecewise.** With gamma = 1 the TD error is
delta_t = r_t + V(s_{t+1}) - V(s_t). Between two probes the interpolated value
is affine, so V(s_{t+1}) - V(s_t) is a constant, and delta_t varies only through
the per-token KL shaping. The critic therefore assigns credit at the granularity
of K segments rather than of tokens -- a fact that no amount of probing accuracy
changes, and that a learned value head does not share.

**The bias obeys a standard interpolation bound.** If V* is L-Lipschitz in the
token index, linear interpolation on a segment of width h errs by at most L h / 2,
so with K probes over T tokens the error is O(T/K). If V* additionally has a
bounded second difference M, the bound tightens to M h^2 / 8 = O(T^2/K^2).

This script measures which regime holds. It probes *every* position on a small
sample -- expensive in calls, trivial in dollars -- then rebuilds the sparse
estimate for each K from those same probes and reports the error decay, the
fitted exponent, and the advantage error that follows from it.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import math
from pathlib import Path

import torch

from rljevf.config import POLICY_MODEL
from rljevf.critic.jev_critic import JevCritic, _interp, probe_positions
from rljevf.data import eval_prompts
from rljevf.evaluate.generate import load_policy, sample
from rljevf.jevclient import JevClient, Meter
from rljevf.guardrails import require_experiment_host


def sparse_estimate(dense: list[float], k: int) -> torch.Tensor:
    """Rebuild the critic's interpolated curve from K of the dense probes."""
    L = len(dense)
    pos = probe_positions(L, k)
    xs = torch.tensor(pos, dtype=torch.float32)
    ys = torch.tensor([dense[p] for p in pos], dtype=torch.float32)
    return _interp(torch.arange(L, dtype=torch.float32), xs, ys)


def fit_exponent(ks: list[int], errs: list[float]) -> float:
    """Slope of log(error) against log(K): -1 is the Lipschitz rate, -2 the
    smooth one."""
    xs = [math.log(k) for k, e in zip(ks, errs) if e > 0]
    ys = [math.log(e) for e in errs if e > 0]
    if len(xs) < 2:
        return float("nan")
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    return num / den if den else float("nan")


async def dense_probe(critic, prompt: str, ids: list[int], stride: int) -> list[float]:
    positions = list(range(0, len(ids), stride))
    prefixes = [critic.tok.decode(ids[:t], skip_special_tokens=True) for t in positions]
    probs = await critic.aprefix_probs_batch([prompt] * len(prefixes), prefixes)
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=POLICY_MODEL)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--stride", type=int, default=4, help="dense probe spacing in tokens")
    ap.add_argument("--ks", default="2,3,4,6,8,12,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_density.json")
    ap.add_argument("--smoke", action="store_true",
                    help="allow running off-GPU to check the code path")
    args = ap.parse_args()
    require_experiment_host("probe_density_study")

    ds = eval_prompts(n=args.n, seed=11)
    prompts = [ds[i]["prompt"] for i in range(len(ds))]
    model, tok, device = load_policy(args.policy)
    comps = sample(model, tok, prompts, device, max_new_tokens=args.max_new_tokens,
                   greedy=False, seed=args.seed, batch_size=8)
    keep = [i for i, c in enumerate(comps) if len(c.split()) > 12]
    prompts = [prompts[i] for i in keep]
    comps = [comps[i] for i in keep]
    print(f"{len(comps)} completions kept", flush=True)

    client = JevClient(meter=Meter(name="probe_density"))
    critic = JevCritic(tokenizer=tok, client=client)

    dense: list[list[float]] = []
    for p, c in zip(prompts, comps):
        ids = tok.encode(c, add_special_tokens=False)
        dense.append(asyncio.run(dense_probe(critic, p, ids, args.stride)))
    lens = [len(d) for d in dense]
    print(f"dense curves: {len(dense)} x ~{sum(lens)/len(lens):.0f} probes "
          f"({sum(lens)} calls, ${client.meter.summary()['cost_usd']:.3f})", flush=True)

    ks = [int(k) for k in args.ks.split(",")]
    rows = {}
    for k in ks:
        mean_abs, max_abs, adv_err = [], [], []
        for d in dense:
            if len(d) < 3:
                continue
            truth = torch.tensor(d, dtype=torch.float32)
            est = sparse_estimate(d, k)
            e = (est - truth).abs()
            mean_abs.append(float(e.mean()))
            max_abs.append(float(e.max()))
            # with gamma = lambda = 1 the advantage error is exactly the value
            # error, so this is the quantity the optimiser actually sees
            adv_err.append(float(e.mean()))
        rows[k] = {
            "mean_abs_error": sum(mean_abs) / len(mean_abs),
            "max_abs_error": sum(max_abs) / len(max_abs),
            "advantage_mean_abs_error": sum(adv_err) / len(adv_err),
            "segments": k - 1,
        }
        print(f"  K={k:3d}: mean |V_hat - V| = {rows[k]['mean_abs_error']:.4f}  "
              f"max = {rows[k]['max_abs_error']:.4f}", flush=True)

    alpha_mean = fit_exponent(ks, [rows[k]["mean_abs_error"] for k in ks])
    alpha_max = fit_exponent(ks, [rows[k]["max_abs_error"] for k in ks])
    print(f"\nfitted decay: mean error ~ K^{alpha_mean:.2f}, max error ~ K^{alpha_max:.2f}")
    print("  (-1 is the Lipschitz rate, -2 the bounded-curvature rate)")

    res = {
        "policy": args.policy, "n_completions": len(dense),
        "stride_tokens": args.stride,
        "mean_dense_probes": sum(lens) / len(lens),
        "by_k": rows,
        "fitted_exponent_mean": alpha_mean,
        "fitted_exponent_max": alpha_max,
        "cost": client.meter.summary(),
        "dense_curves": dense,
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
