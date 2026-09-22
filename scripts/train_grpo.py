#!/usr/bin/env python
"""Phase 1 -- GRPO, where the reward source is the only thing that varies."""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rljevf.config import RUN_ROOT, RunConfig, supports_bf16
from rljevf.data import filter_by_prompt_tokens, fingerprint, rl_prompts
from rljevf.registry import build_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--reward", required=True,
                    choices=["jev", "bert_rm", "llm_judge", "self_judge", "self_certainty"])
    ap.add_argument("--rubric-source", default="static", choices=["static", "generated", "policy"])
    ap.add_argument("--sft", required=True, help="path to the R0 checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max-completion-length", type=int, default=384)
    ap.add_argument("--n-prompts", type=int, default=8192)
    ap.add_argument("--reward-call-budget", type=int, default=None)
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    cfg = RunConfig(
        run_id=args.run_id, reward_source=args.reward, algo="grpo",
        rubric_source=args.rubric_source, seed=args.seed, sft_checkpoint=args.sft,
        num_generations=args.num_generations, prompts_per_step=args.prompts_per_step,
        learning_rate=args.lr, beta=args.beta, max_steps=args.steps,
        max_completion_length=args.max_completion_length,
        reward_call_budget=args.reward_call_budget,
    )
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=1))

    tok = AutoTokenizer.from_pretrained(args.sft)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Data: identical prompts and identical order in every arm.
    ds = rl_prompts(n=args.n_prompts, seed=0)
    ds = filter_by_prompt_tokens(ds, tok, cfg.max_prompt_length)
    print("prompt set:", fingerprint(ds), flush=True)

    bf16 = supports_bf16()
    model = AutoModelForCausalLM.from_pretrained(
        args.sft, dtype=torch.bfloat16 if bf16 else torch.float32
    )

    reward_fn, source = build_reward(cfg, policy=model, tokenizer=tok)

    gen_bs = args.prompts_per_step * args.num_generations
    grad_accum = max(1, gen_bs // args.micro_bs)
    gcfg = GRPOConfig(
        output_dir=str(out),
        max_steps=args.steps,
        learning_rate=args.lr,
        beta=args.beta,                       # set explicitly: TRL's default drifted
        num_generations=args.num_generations,
        generation_batch_size=gen_bs,
        per_device_train_batch_size=args.micro_bs,
        gradient_accumulation_steps=grad_accum,
        max_completion_length=args.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        seed=args.seed,
        data_seed=0,                          # same data order for every seed
        logging_steps=1,
        save_steps=max(1, args.steps // 5),
        save_strategy="steps",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        # gradient checkpointing segfaults on MPS (torch 2.14); it is only
        # needed for VRAM on the T4s anyway
        gradient_checkpointing=torch.cuda.is_available(),
        log_completions=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
    )

    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn, args=gcfg,
        train_dataset=ds, processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))

    stats = source.stats() if source is not None else {}
    (out / "reward_source_stats.json").write_text(json.dumps(stats, indent=1))
    print("reward source stats:", json.dumps(stats, indent=1))
    print("saved ->", out / "final")


if __name__ == "__main__":
    main()
