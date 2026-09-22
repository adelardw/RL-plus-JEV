#!/usr/bin/env python
"""Phase 1 -- GRPO, where the reward source is the only thing that varies."""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rljevf.config import RUN_ROOT, RunConfig, supports_bf16
from rljevf.data import filter_by_prompt_tokens, fingerprint, rl_prompts
from rljevf.registry import build_reward
from rljevf.resume import describe, find_resume
from rljevf.guardrails import require_experiment_host


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
    ap.add_argument("--lora-r", type=int, default=0,
                    help="LoRA rank; 0 means full fine-tuning. On a T4 full "
                         "fine-tuning of a 0.5B policy does not fit at any "
                         "batch size (see scripts/memory_budget.py)")
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--optim", default="adamw_torch",
                    help="adamw_torch | adafactor | adamw_bnb_8bit -- the optimiser's\n"
                         "state is 4GB of a T4 for a 0.5B model in fp32")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--scale-rewards", default="group",
                    help="pinned for the same reason as beta: TRL defaults drift")
    ap.add_argument("--max-completion-length", type=int, default=384)
    ap.add_argument("--n-prompts", type=int, default=8192)
    ap.add_argument("--reward-call-budget", type=int, default=None)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--resume", default="auto", choices=["auto", "off"],
                    help="continue an interrupted run from its saved state")
    ap.add_argument("--smoke", action="store_true",
                    help="allow running off-GPU to check the code path")
    args = ap.parse_args()
    require_experiment_host("train_grpo")

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
        scale_rewards=args.scale_rewards,     # same reason; pin it rather than inherit
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
        optim=args.optim,
        log_completions=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
    )

    peft_config = None
    if args.lora_r > 0:
        from peft import LoraConfig

        # With PEFT, TRL sets ref_model = None and computes the reference
        # logprobs by disabling adapters, so this removes the second copy of
        # the policy as well as the optimiser state.
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        print(f"LoRA r={args.lora_r} alpha={args.lora_alpha}", flush=True)

    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn, args=gcfg,
        train_dataset=ds, processing_class=tok, peft_config=peft_config,
    )
    if peft_config is not None:
        tr = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in trainer.model.parameters())
        print(f"trainable {tr/1e6:.1f}M of {tot/1e6:.1f}M ({100*tr/tot:.2f}%)", flush=True)

    # TRL's Trainer restores weights, optimiser, LR schedule and global step
    # from a checkpoint directory, so an interrupted run continues rather than
    # silently restarting and double-spending the reward budget.
    ckpt = find_resume(args.run_id, out) if args.resume == "auto" else None
    print(describe(ckpt), flush=True)
    trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None)
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    _stash_resume_state(out)

    stats = source.stats() if source is not None else {}
    (out / "reward_source_stats.json").write_text(json.dumps(stats, indent=1))
    print("reward source stats:", json.dumps(stats, indent=1))
    print("saved ->", out / "final")


def _stash_resume_state(out: Path) -> None:
    """Move the newest TRL checkpoint to `resume_state`, out of the glob the
    Kaggle runner deletes, and drop the older ones."""
    import shutil

    cks = [d for d in out.glob("checkpoint-*") if d.is_dir()]
    if not cks:
        return

    def step(d: Path) -> int:
        try:
            return int(d.name.split("-")[-1])
        except ValueError:
            return -1

    newest = max(cks, key=step)
    target = out / "resume_state"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(newest), str(target))
    for d in cks:
        if d != newest and d.exists():
            shutil.rmtree(d, ignore_errors=True)
    print(f"resume state stashed -> {target}", flush=True)


if __name__ == "__main__":
    main()
