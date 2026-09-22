#!/usr/bin/env python
"""Measure what one GPU-hour buys, so the study can be scheduled honestly.

With a 6 GPU-hour weekly quota the schedule is the binding constraint, and
guessing throughput would mean discovering half-way through week three that the
plan never fit. This runs a few steps of each trainer at the real settings and
reports seconds per step, then extrapolates.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from pathlib import Path

import torch

from rljevf.config import supports_bf16


def bench_grpo(model_name, steps, prompts, G, max_new, micro_bs, use_vllm):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from rljevf.data import filter_by_prompt_tokens, rl_prompts

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = filter_by_prompt_tokens(rl_prompts(n=256, seed=0), tok, 512)
    bf16 = supports_bf16()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if bf16 else torch.float32)

    def rew(prompts_, completions, **kw):
        return [float(len(c)) / 1000 for c in completions]

    gen_bs = prompts * G
    cfg = GRPOConfig(
        output_dir="/tmp/bench_grpo", max_steps=steps, num_generations=G,
        generation_batch_size=gen_bs, per_device_train_batch_size=micro_bs,
        gradient_accumulation_steps=max(1, gen_bs // micro_bs),
        max_completion_length=max_new, logging_steps=1, save_strategy="no",
        gradient_checkpointing=torch.cuda.is_available(), report_to=[],
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        use_vllm=use_vllm, temperature=1.0, top_p=1.0,
    )
    t = time.perf_counter()
    tr = GRPOTrainer(model=model, reward_funcs=rew, args=cfg,
                     train_dataset=ds, processing_class=tok)
    build = time.perf_counter() - t
    t = time.perf_counter()
    tr.train()
    total = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    return {"build_s": round(build, 1), "total_s": round(total, 1),
            "s_per_step": round(total / steps, 2), "peak_vram_gb": round(peak, 2),
            "completions_per_step": gen_bs, "use_vllm": use_vllm}


def bench_ppo(model_name, steps, prompts, max_new, micro_bs, gen_bs):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rljevf.data import filter_by_prompt_tokens, rl_prompts
    from rljevf.ppo import PPOArgs, PPOTrainer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = filter_by_prompt_tokens(rl_prompts(n=256, seed=0), tok, 512)
    policy = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    ref = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    policy.config.use_cache = False

    from rljevf.critic.jev_critic import LearnedValueCritic
    critic = LearnedValueCritic(policy)

    def rew(prompts_, completions, **kw):
        return [float(len(c)) / 1000 for c in completions]

    args = PPOArgs(max_steps=steps, batch_prompts=prompts, micro_batch_size=micro_bs,
                   generation_batch_size=gen_bs, max_completion_length=max_new,
                   save_every=0, log_every=1)
    tr = PPOTrainer(policy=policy, ref_policy=ref, tokenizer=tok, reward_fn=rew,
                    critic=critic, args=args, train_dataset=ds, out_dir=Path("/tmp/bench_ppo"))
    t = time.perf_counter()
    hist = tr.train()
    total = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    return {"total_s": round(total, 1), "s_per_step": round(total / steps, 2),
            "peak_vram_gb": round(peak, 2), "completions_per_step": prompts,
            "step_times": [h["step_time_s"] for h in hist]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--grpo-prompts", type=int, default=16)
    ap.add_argument("--grpo-g", type=int, default=8)
    ap.add_argument("--ppo-prompts", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--gen-bs", type=int, default=16)
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--out", default="/kaggle/working/results/gpu_benchmark.json")
    args = ap.parse_args()

    res = {"model": args.model, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    for vllm in ([False] if args.skip_vllm else [False, True]):
        key = f"grpo_vllm={vllm}"
        try:
            res[key] = bench_grpo(args.model, args.steps, args.grpo_prompts, args.grpo_g,
                                  args.max_new, args.micro_bs, vllm)
            print(key, res[key], flush=True)
        except Exception as e:  # noqa: BLE001
            res[key] = {"error": f"{type(e).__name__}: {e}"}
            print(key, "FAILED:", res[key]["error"][:300], flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    if not args.skip_ppo:
        try:
            res["ppo"] = bench_ppo(args.model, args.steps, args.ppo_prompts,
                                   args.max_new, args.micro_bs, args.gen_bs)
            print("ppo", res["ppo"], flush=True)
        except Exception as e:  # noqa: BLE001
            res["ppo"] = {"error": f"{type(e).__name__}: {e}"}
            print("ppo FAILED:", res["ppo"]["error"][:300], flush=True)

    # what the weekly quota buys
    sched = {}
    for k, v in res.items():
        if isinstance(v, dict) and "s_per_step" in v:
            sched[k] = {
                "minutes_per_250_step_run": round(v["s_per_step"] * 250 / 60, 1),
                "runs_per_6h_quota": round(6 * 3600 / (v["s_per_step"] * 250), 2),
            }
    res["schedule"] = sched
    print(json.dumps(sched, indent=1), flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))

    # A benchmark that measured nothing must not report success: the runner
    # schedules the rest of the study off these numbers.
    measured = [k for k, v in res.items() if isinstance(v, dict) and "s_per_step" in v]
    if not measured:
        print("no benchmark produced a timing", flush=True)
        raise SystemExit(2)
    print("measured:", measured, flush=True)


if __name__ == "__main__":
    main()
