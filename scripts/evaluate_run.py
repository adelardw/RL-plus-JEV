#!/usr/bin/env python
"""Evaluate one checkpoint: win-rate vs the SFT baseline, hacking detectors,
capability tax, and the completions other stages reuse.

Completions are written out so the cross-reward matrix and the KL curves can be
built later without re-sampling -- sampling twice would add noise that has
nothing to do with the arms being compared.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rljevf.config import EVAL_JUDGE_MODEL
from rljevf.data import eval_prompts, gsm8k
from rljevf.evaluate.generate import load_policy, sample
from rljevf.evaluate.gsm8k_eval import accuracy
from rljevf.evaluate.metrics import distinct_n, length_stats, self_praise_rate, sequence_kl
from rljevf.evaluate.winrate import win_rate
from rljevf.jevclient import Meter
from rljevf.orclient import ChatClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--baseline", required=True, help="the R0 SFT checkpoint")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-gsm8k", type=int, default=250)
    ap.add_argument("--judge", default=EVAL_JUDGE_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--sample", dest="greedy", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-winrate", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"run_id": args.run_id, "checkpoint": args.checkpoint, "judge": args.judge}

    model, tok, device = load_policy(args.checkpoint)

    # --- open-ended prompts ------------------------------------------------
    ds = eval_prompts(n=args.n_eval, seed=0)
    prompts = [ds[i]["prompt"] for i in range(len(ds))]
    comps = sample(model, tok, prompts, device, max_new_tokens=args.max_new_tokens,
                   greedy=args.greedy, seed=args.seed, batch_size=args.batch_size)
    (out / "completions.json").write_text(
        json.dumps({"prompts": prompts, "completions": comps}, indent=1)
    )

    res["length"] = length_stats(comps, tok)
    res["self_praise_rate"] = self_praise_rate(comps)
    res["distinct_2"] = distinct_n(comps, 2)
    res["empty_rate"] = sum(1 for c in comps if not c.strip()) / max(1, len(comps))

    # --- capability tax ----------------------------------------------------
    g = gsm8k(n=args.n_gsm8k, seed=0)
    gp = [g[i]["prompt"] for i in range(len(g))]
    gc = sample(model, tok, gp, device, max_new_tokens=320, greedy=True,
                seed=args.seed, batch_size=args.batch_size)
    res.update(accuracy(gc, [g[i]["gold"] for i in range(len(g))]))

    # --- divergence from the start point -----------------------------------
    base_model, base_tok, _ = load_policy(args.baseline, device=device)
    res["kl_per_token_vs_sft"] = sequence_kl(
        model, base_model, tok, prompts[:64], comps[:64], device=device, batch_size=2
    )

    # --- win-rate ----------------------------------------------------------
    if not args.skip_winrate:
        base_path = Path(args.baseline)
        cache = base_path / "baseline_completions.json"
        if cache.exists():
            bcomps = json.loads(cache.read_text())["completions"]
        else:
            bcomps = sample(base_model, base_tok, prompts, device,
                            max_new_tokens=args.max_new_tokens, greedy=args.greedy,
                            seed=args.seed, batch_size=args.batch_size)
            cache.write_text(json.dumps({"prompts": prompts, "completions": bcomps}, indent=1))
        judge = ChatClient(args.judge, concurrency=12,
                           meter=Meter(name=f"evaljudge_{args.judge.replace('/', '_')}"))
        wr = win_rate(prompts, comps, bcomps, judge, seed=args.seed)
        res["winrate"] = wr.to_dict()
        res["judge_cost"] = judge.meter.summary()

    (out / "eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
