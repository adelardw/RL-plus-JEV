#!/usr/bin/env python
"""Predict what fits on the card, instead of discovering it a session at a time.

Two OOMs cost two Kaggle sessions, both at settings that a back-of-envelope
calculation rules out immediately. The dominant terms for a small policy with a
large vocabulary are not what one expects: the optimiser state and the logits
tensor each exceed the model weights.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse

GB = 1024 ** 3

OPTIM_BYTES_PER_PARAM = {
    "adamw_torch": 8,      # two fp32 moments
    "adamw_bnb_8bit": 2,   # two int8 moments
    "adafactor": 0.5,      # factored second moment, no first moment
    "sgd": 0,
}


def budget(params: float, vocab: int, seq: int, micro_bs: int,
           optim: str = "adamw_torch", param_bytes: int = 4,
           ref_model: bool = True, ref_bytes: int = 4,
           grad_checkpointing: bool = True, n_layers: int = 24,
           hidden: int = 896, gen_batch: int = 128, kv_heads: int = 2,
           head_dim: int = 64, kv_bytes: int = 4,
           fragmentation: float = 0.15, lora_frac: float = 0.0) -> dict:
    # With LoRA only the adapters carry gradients and optimiser state, and the
    # reference policy is the base model with adapters disabled -- so the three
    # terms that dominate full fine-tuning nearly vanish at once.
    trainable = params * lora_frac if lora_frac else params
    weights = params * param_bytes
    grads = trainable * param_bytes
    opt = trainable * OPTIM_BYTES_PER_PARAM.get(optim, 8)
    ref = 0.0 if lora_frac else (params * ref_bytes if ref_model else 0.0)

    # the logits are (micro_bs, seq, vocab) in fp32 for the loss
    logits = micro_bs * seq * vocab * 4
    # activations: with checkpointing, roughly one layer's worth live at a time
    act_per_layer = micro_bs * seq * hidden * 4 * 8
    acts = act_per_layer * (1 if grad_checkpointing else n_layers)

    # Rollout KV cache. It is freed before the backward pass, but the caching
    # allocator holds the blocks, so it still counts against the peak -- the
    # OOM messages show it as "reserved but unallocated".
    kv = 2 * n_layers * gen_batch * seq * kv_heads * head_dim * kv_bytes

    live = weights + grads + opt + ref + acts + kv
    total = (live + logits) * (1 + fragmentation)
    return {
        "weights_gb": weights / GB,
        "grads_gb": grads / GB,
        "optimizer_gb": opt / GB,
        "reference_model_gb": ref / GB,
        "kv_cache_gb": kv / GB,
        "logits_gb": logits / GB,
        "activations_gb": acts / GB,
        "total_gb": total / GB,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, default=494e6, help="policy parameter count")
    ap.add_argument("--vocab", type=int, default=151936)
    ap.add_argument("--seq", type=int, default=896, help="prompt + completion tokens")
    ap.add_argument("--card-gb", type=float, default=14.56)
    ap.add_argument("--headroom-gb", type=float, default=0.8,
                    help="CUDA context and driver overhead")
    ap.add_argument("--gen-batch", type=int, default=128,
                    help="completions generated per step (prompts x G)")
    ap.add_argument("--lora-frac", type=float, default=0.0,
                    help="trainable fraction under LoRA, e.g. 0.01; 0 means full fine-tuning")
    ap.add_argument("--validate", action="store_true",
                    help="check the model against the OOM we actually observed")
    args = ap.parse_args()

    usable = args.card_gb - args.headroom_gb
    print(f"policy {args.params/1e6:.0f}M params, vocab {args.vocab:,}, "
          f"seq {args.seq}, card {args.card_gb:.1f}GB "
          f"({usable:.1f}GB usable)\n")

    print(f"{'optimiser':16s} {'mb':>3s} {'w+g':>6s} {'optim':>7s} {'ref':>6s} "
          f"{'kv':>6s} {'logits':>7s} {'TOTAL':>7s}  fits?")
    best = {}
    for optim in ("adamw_torch", "adamw_bnb_8bit", "adafactor"):
        for mb in (16, 8, 4, 2, 1):
            b = budget(args.params, args.vocab, args.seq, mb, optim,
                       gen_batch=args.gen_batch, lora_frac=args.lora_frac)
            fits = b["total_gb"] <= usable
            if fits and optim not in best:
                best[optim] = mb
            print(f"{optim:16s} {mb:3d} {b['weights_gb']+b['grads_gb']:6.2f} "
                  f"{b['optimizer_gb']:7.2f} {b['reference_model_gb']:6.2f} "
                  f"{b['kv_cache_gb']:6.2f} {b['logits_gb']:7.2f} "
                  f"{b['total_gb']:7.2f}  {'yes' if fits else 'NO'}")
        print()

    print("largest micro-batch predicted to fit:")
    for k, v in best.items():
        print(f"  {k:16s} {v}")
    if not best:
        print("  none -- the configuration does not fit at any batch size")
    print("\nnote: the optimiser state and the logits each exceed the weights here,")
    print("      which is why the batch size is the third lever, not the first.")

    if args.validate:
        # Observed on a Kaggle T4: GRPO, 16 prompts x G=8, micro_bs=8,
        # adamw_torch -- 12.08 GiB in use when a 3.48 GiB allocation failed.
        b = budget(args.params, args.vocab, args.seq, 8, "adamw_torch",
                   gen_batch=args.gen_batch)
        predicted_live = (b["total_gb"] / 1.15) - b["logits_gb"]
        print(f"\nvalidation against the observed OOM:")
        print(f"  predicted resident before the failing allocation: "
              f"{predicted_live:.2f} GB   (observed 12.08 GB)")
        print(f"  predicted failing allocation: {b['logits_gb']:.2f} GB"
              f"   (observed 3.48 GB)")
        print(f"  predicted total {b['total_gb']:.2f} GB vs {usable:.1f} GB usable "
              f"-> {'OOM' if b['total_gb'] > usable else 'fits'}  (observed OOM)")


if __name__ == "__main__":
    main()
