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
from rljevf.guardrails import require_experiment_host


def bench_grpo(model_name, steps, G, max_new, use_vllm,
               micro_bs=8, prompts=16, optim="adamw_torch", lora_r=0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from rljevf.data import filter_by_prompt_tokens, rl_prompts

    probe = MemoryProbe()
    probe.mark("start")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = filter_by_prompt_tokens(rl_prompts(n=256, seed=0), tok, 512)
    bf16 = supports_bf16()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if bf16 else torch.float32)
    probe.mark("policy loaded")

    def rew(prompts, completions, **kw):   # TRL passes prompts= by keyword
        return [float(len(c)) / 1000 for c in completions]

    gen_bs = prompts * G
    cfg = GRPOConfig(
        output_dir="/tmp/bench_grpo", max_steps=steps, num_generations=G,
        generation_batch_size=gen_bs, per_device_train_batch_size=micro_bs,
        gradient_accumulation_steps=max(1, gen_bs // micro_bs),
        max_completion_length=max_new, logging_steps=1, save_strategy="no",
        gradient_checkpointing=torch.cuda.is_available(), report_to=[],
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        use_vllm=use_vllm, temperature=1.0, top_p=1.0, optim=optim,
    )
    peft_config = None
    if lora_r > 0:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])

    t = time.perf_counter()
    tr = GRPOTrainer(model=model, reward_funcs=rew, args=cfg,
                     train_dataset=ds, processing_class=tok,
                     peft_config=peft_config)
    build = time.perf_counter() - t
    probe.mark("trainer built (+reference)")
    t = time.perf_counter()
    try:
        tr.train()
    except torch.cuda.OutOfMemoryError:
        probe.mark("at OOM")
        del tr, model
        raise
    total = time.perf_counter() - t
    probe.mark("after training")
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    return {"build_s": round(build, 1), "total_s": round(total, 1),
            "s_per_step": round(total / steps, 2), "peak_vram_gb": round(peak, 2),
            "completions_per_step": gen_bs, "use_vllm": use_vllm,
            "optim": optim, "lora_r": lora_r, "vram_phases": probe.report()}


def bench_ppo(model_name, steps, prompts, max_new, micro_bs, gen_bs,
              forward_chunk=2, lora_r=0):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rljevf.data import filter_by_prompt_tokens, rl_prompts
    from rljevf.ppo import PPOArgs, PPOTrainer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = filter_by_prompt_tokens(rl_prompts(n=256, seed=0), tok, 512)
    policy = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    policy.config.use_cache = False
    if lora_r > 0:
        from peft import LoraConfig, get_peft_model

        policy = get_peft_model(policy, LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        ref = None
    else:
        ref = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

    from rljevf.critic.jev_critic import LearnedValueCritic
    critic = LearnedValueCritic(policy)

    def rew(prompts, completions, **kw):   # TRL passes prompts= by keyword
        return [float(len(c)) / 1000 for c in completions]

    args = PPOArgs(max_steps=steps, batch_prompts=prompts, micro_batch_size=micro_bs,
                   generation_batch_size=gen_bs, forward_chunk=forward_chunk,
                   max_completion_length=max_new, save_every=0, log_every=1)
    tr = PPOTrainer(policy=policy, ref_policy=ref, tokenizer=tok, reward_fn=rew,
                    critic=critic, args=args, train_dataset=ds, out_dir=Path("/tmp/bench_ppo"))
    t = time.perf_counter()
    try:
        hist = tr.train()
    except torch.cuda.OutOfMemoryError:
        del tr, policy, ref, critic
        raise
    total = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    return {"total_s": round(total, 1), "s_per_step": round(total / steps, 2),
            "peak_vram_gb": round(peak, 2), "completions_per_step": prompts,
            "lora_r": lora_r,
            "step_times": [h["step_time_s"] for h in hist]}



class MemoryProbe:
    """Record VRAM at each phase, so an OOM names its cause.

    Four rounds of guessing produced three wrong explanations for the same
    failure -- the micro-batch, then the optimiser, then the KV cache, each
    ruled out only by the next experiment. Measuring per phase replaces the
    guessing: whatever is resident when generation starts is printed, not
    inferred.
    """

    def __init__(self):
        self.marks: list[tuple[str, float, float]] = []

    def mark(self, phase: str) -> None:
        import torch

        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        self.marks.append((phase, round(alloc, 2), round(reserved, 2)))
        print(f"    [vram] {phase:28s} allocated {alloc:5.2f} GB  "
              f"reserved {reserved:5.2f} GB", flush=True)

    def report(self) -> list[dict]:
        return [{"phase": p, "allocated_gb": a, "reserved_gb": r}
                for p, a, r in self.marks]


def autofit(fn, label: str, configs: list[dict], **kw) -> dict:
    """Search the configurations that actually control memory, in order.

    There are two independent axes, and confusing them wastes a session. The
    micro-batch governs the training logits; the *generation* batch, which in
    TRL is the completions produced per optimisation step, governs the KV cache
    and is unaffected by the micro-batch. A full-fine-tuning run OOMed at
    micro-batch 8, 4, 2 and 1 with the same 2.23 GiB allocation each time --
    the constant size was the clue that the micro-batch was the wrong knob.
    """
    import torch

    for cfg in configs:
        mb = cfg["micro_bs"]
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            print(f"  {label}: trying {cfg}", flush=True)
            r = fn(**cfg, **kw)
            r.update(cfg)
            r["fit"] = True
            return r
        except torch.cuda.OutOfMemoryError as e:
            msg = str(e)[:70]
            print(f"  {label}: {cfg} OOM ({msg})", flush=True)
            _release(label)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            _release(label)
            return {"error": err, **cfg, "fit": False}
    return {"error": "no configuration fitted", "fit": False,
            "tried": configs}


def _release(label: str) -> None:
    """Free the failed attempt before trying a smaller one.

    `empty_cache()` releases cached blocks but not memory still referenced, and
    a raised exception keeps every frame's locals alive through its traceback --
    including the model. The bench functions therefore `del` their model and
    trainer before re-raising; this collects what that leaves and reports
    whether the card actually came back, because a search that silently
    inherits its predecessor reports "does not fit at any size" when what it
    measured was a leak. The symptom was PPO failing to allocate 18 MiB.
    """
    import gc

    import torch

    if not torch.cuda.is_available():
        return
    before = torch.cuda.memory_allocated() / 2**30
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    print(f"    [vram] released {before - after:.2f} GB, "
          f"{after:.2f} GB still resident", flush=True)
    if after > 0.5:
        print(f"    [vram] WARNING: {after:.2f} GB did not free -- the next "
              f"attempt starts handicapped and its result is not comparable",
              flush=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--grpo-prompts", type=int, default=16)
    ap.add_argument("--grpo-g", type=int, default=8)
    ap.add_argument("--ppo-prompts", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--gen-bs", type=int, default=8)
    ap.add_argument("--forward-chunk", type=int, default=2)
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-ppo", action="store_true")
    ap.add_argument("--optims", default="adamw_torch",
                    help="optimisers to try, cheapest state last")
    ap.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank to benchmark; 0 also measures full "
                         "fine-tuning, which does not fit on a T4 for this "
                         "policy at any batch size")
    ap.add_argument("--out", default="/kaggle/working/results/gpu_benchmark.json")
    ap.add_argument("--smoke", action="store_true",
                    help="allow running off-GPU to check the code path")
    args = ap.parse_args()
    require_experiment_host("gpu_benchmark")

    res = {"model": args.model, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    # Vary the generation batch (prompts per step) as well as the micro-batch:
    # they control different allocations and only one of them affects the KV
    # cache. Largest total work first.
    configs = [
        {"micro_bs": mb, "prompts": pr}
        for pr in (args.grpo_prompts, args.grpo_prompts // 2, args.grpo_prompts // 4)
        for mb in (8, 4, 2)
        if pr >= 1 and mb <= args.micro_bs
    ]
    for optim in [o for o in args.optims.split(",") if o]:
        for vllm in ([False] if args.skip_vllm else [False, True]):
            key = f"grpo_vllm={vllm}_optim={optim}_lora={args.lora_r}"
            res[key] = autofit(
                lambda _o=optim, _v=vllm, **cfg: bench_grpo(
                    args.model, args.steps, args.grpo_g, args.max_new, _v,
                    optim=_o, lora_r=args.lora_r, **cfg),
                key, configs)
            print(key, res[key], flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    if not args.skip_ppo:
        res["ppo"] = autofit(
            lambda **cfg: bench_ppo(
                args.model, args.steps, cfg["prompts"], args.max_new,
                cfg["micro_bs"], args.gen_bs,
                min(args.forward_chunk, cfg["micro_bs"]), args.lora_r),
            "ppo", [{"micro_bs": mb, "prompts": pr}
                    for pr in (args.ppo_prompts, args.ppo_prompts // 2)
                    for mb in (4, 2, 1) if mb <= args.micro_bs])
        print("ppo", res["ppo"], flush=True)

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
