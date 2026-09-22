#!/usr/bin/env python
"""Phase 2 -- PPO, where the *critic* is the thing that varies.

Two arms share this script and differ by one flag:

  --critic learned   a value head on the policy backbone (textbook PPO)
  --critic jev       Jev valuing token prefixes; no value head, no value loss

Before training with --critic jev we calibrate Jev's prefix probability against
the terminal rubric reward on SFT samples and record the fit, so the reported
TD errors are in reward units and the quality of the calibration is visible
rather than assumed.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rljevf.config import RunConfig
from rljevf.data import filter_by_prompt_tokens, fingerprint, rl_prompts
from rljevf.evaluate.generate import sample
from rljevf.ppo import PPOArgs, PPOTrainer, _maybe_sync
from rljevf.registry import build_critic, build_reward
from rljevf.resume import describe, find_resume


def calibrate(critic, policy, tok, device, reward_fn, n: int, seed: int) -> dict:
    """Fit V = a * p_jev + b on fresh SFT samples."""
    ds = rl_prompts(n=n, seed=1000 + seed)          # disjoint seed from training
    prompts = [ds[i]["prompt"] for i in range(len(ds))]
    comps = sample(policy, tok, prompts, device, max_new_tokens=192, seed=seed, batch_size=8)
    rewards = _maybe_sync(reward_fn(prompts=prompts, completions=comps))
    cal = asyncio.run(critic.afit_calibration(prompts, comps, rewards))
    print(f"calibration: V = {cal.a:.3f}*p + {cal.b:.3f}  "
          f"(r={cal.pearson:.3f}, R2={cal.r2:.3f}, n={cal.n})", flush=True)
    return cal.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--reward", required=True,
                    choices=["jev", "bert_rm", "llm_judge", "self_judge"])
    ap.add_argument("--critic", required=True, choices=["learned", "jev"])
    ap.add_argument("--sft", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-prompts", type=int, default=16)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--gen-bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--value-lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--num-prefixes", type=int, default=6)
    ap.add_argument("--max-completion-length", type=int, default=384)
    ap.add_argument("--calibration-n", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=8192)
    ap.add_argument("--reward-call-budget", type=int, default=None)
    ap.add_argument("--resume", default="auto", choices=["auto", "off"])
    ap.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()

    cfg = RunConfig(
        run_id=args.run_id, reward_source=args.reward, algo="ppo", critic=args.critic,
        seed=args.seed, sft_checkpoint=args.sft, max_steps=args.steps,
        prompts_per_step=args.batch_prompts, learning_rate=args.lr,
        ppo_lr_value=args.value_lr, ppo_kl_coef=args.kl_coef,
        jev_critic_num_prefixes=args.num_prefixes,
        max_completion_length=args.max_completion_length,
        reward_call_budget=args.reward_call_budget,
    )
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=1))

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.float32          # PPO in fp32; T4 has no bf16 and fp16 PPO is brittle
    tok = AutoTokenizer.from_pretrained(args.sft)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.sft, dtype=dtype)
    ref = AutoModelForCausalLM.from_pretrained(args.sft, dtype=dtype)
    policy.config.use_cache = False

    reward_fn, source = build_reward(cfg, policy=policy, tokenizer=tok)
    critic = build_critic(cfg, policy=policy, tokenizer=tok)

    cal_info = None
    if args.critic == "jev":
        policy.to(device)
        cal_info = calibrate(critic, policy, tok, device, reward_fn,
                             n=args.calibration_n, seed=args.seed)
        (out / "calibration.json").write_text(json.dumps(cal_info, indent=1))

    ds = rl_prompts(n=args.n_prompts, seed=0)
    ds = filter_by_prompt_tokens(ds, tok, cfg.max_prompt_length)
    print("prompt set:", fingerprint(ds), flush=True)

    pargs = PPOArgs(
        learning_rate=args.lr, value_learning_rate=args.value_lr,
        kl_coef=args.kl_coef,
        batch_prompts=args.batch_prompts, micro_batch_size=args.micro_bs,
        generation_batch_size=args.gen_bs,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps, seed=args.seed,
        save_every=args.save_every,
        reward_call_budget=args.reward_call_budget,
    )

    trainer = PPOTrainer(
        policy=policy, ref_policy=ref, tokenizer=tok, reward_fn=reward_fn,
        critic=critic, args=pargs, train_dataset=ds, out_dir=out, device=device,
    )

    ckpt = find_resume(args.run_id, out) if args.resume == "auto" else None
    print(describe(ckpt), flush=True)
    if ckpt is not None:
        trainer.load_checkpoint(ckpt)
    trainer.train()

    stats = {
        "reward_source": source.stats() if source is not None else {},
        "critic": critic.stats(),
        "calibration": cal_info,
    }
    (out / "run_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
