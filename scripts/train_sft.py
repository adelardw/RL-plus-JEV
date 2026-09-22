#!/usr/bin/env python
"""R0 -- the shared starting point.

Every RL arm starts from this one checkpoint, so that differences downstream
are attributable to the reward source and nothing else.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

from rljevf.guardrails import pin_single_gpu

pin_single_gpu()

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from rljevf.config import POLICY_MODEL, RUN_ROOT, supports_bf16
from rljevf.data import fingerprint, sft_dataset
from rljevf.guardrails import require_experiment_host


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=POLICY_MODEL)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RUN_ROOT / "R0-sft"))
    ap.add_argument("--smoke", action="store_true",
                    help="allow running off-GPU to check the code path")
    args = ap.parse_args()
    require_experiment_host("train_sft")

    ds = sft_dataset(n=args.n, seed=args.seed)
    print("SFT data:", fingerprint(ds, key="messages"), flush=True)

    bf16 = supports_bf16()
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        logging_steps=20,
        save_strategy="no",
        seed=args.seed,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        # gradient checkpointing segfaults on MPS (torch 2.14); it is only
        # needed for VRAM on the T4s anyway
        gradient_checkpointing=torch.cuda.is_available(),
        report_to=[],
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if bf16 else torch.float32
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    Path(args.out, "sft_meta.json").write_text(
        json.dumps({"base": args.model, "n": args.n, "seed": args.seed}, indent=1)
    )
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
