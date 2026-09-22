"""Batch sampling from a checkpoint, with the same decoding settings used at
training time so eval and training see the same distribution."""
from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_policy(path: str, device: str | None = None, dtype=None):
    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if dtype is None:
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok, device


@torch.no_grad()
def sample(
    model, tok, prompts: Sequence[str], device: str,
    max_new_tokens: int = 384, temperature: float = 1.0, top_p: float = 1.0,
    batch_size: int = 8, seed: int = 0, greedy: bool = False,
) -> list[str]:
    torch.manual_seed(seed)
    tok.padding_side = "left"
    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else top_p,
            pad_token_id=tok.pad_token_id,
        )
        out.extend(
            tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        )
    return out
