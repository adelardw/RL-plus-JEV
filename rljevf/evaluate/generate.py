"""Batch sampling from a checkpoint, with the same decoding settings used at
training time so eval and training see the same distribution."""
from __future__ import annotations

from typing import Sequence

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_policy(path: str, device: str | None = None, dtype=None):
    """Load a checkpoint, applying a LoRA adapter if that is what was saved.

    This is explicit because the implicit path fails silently and catastrophically:
    `AutoModelForCausalLM.from_pretrained` on an adapter-only directory returns
    the *base* model with the adapter ignored -- no error, no warning, weights
    bit-identical to the untrained checkpoint. Every arm would then evaluate as
    its own baseline and the study would conclude that the reward source makes
    no difference.
    """
    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if dtype is None:
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    p = Path(path)
    adapter_cfg = p / "adapter_config.json"
    if adapter_cfg.is_file():
        import json

        from peft import PeftModel

        base_id = json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")
        if not base_id:
            raise ValueError(f"{adapter_cfg} names no base model")
        if not (Path(base_id).exists() or "/" in str(base_id)):
            raise FileNotFoundError(
                f"adapter at {p} needs its base model {base_id!r}, which is not "
                f"present -- an adapter cannot be evaluated without it"
            )
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
        model = PeftModel.from_pretrained(base, str(p), dtype=dtype)
        # merge so downstream code sees an ordinary causal LM
        model = model.merge_and_unload()
        tok_src = str(p) if (p / "tokenizer_config.json").exists() else base_id
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        tok_src = path

    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(tok_src)
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
