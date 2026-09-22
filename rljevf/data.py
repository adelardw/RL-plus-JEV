"""Datasets, fixed and shared across every arm.

The prompt set and its order are controlled variables (plan section 4): every
run sees the same prompts in the same order, so differences between arms cannot
come from the data.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from datasets import Dataset, load_dataset

ULTRAFEEDBACK = "HuggingFaceH4/ultrafeedback_binarized"
GSM8K = "openai/gsm8k"


def _first_user_turn(messages: Any) -> str | None:
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content")
    return None


def _clean(ds: Dataset, min_chars: int = 16, max_chars: int = 4000) -> Dataset:
    return ds.filter(
        lambda r: isinstance(r["prompt"], str) and min_chars <= len(r["prompt"]) <= max_chars
    )


@lru_cache(maxsize=8)
def _ultrafeedback(split: str) -> Dataset:
    return load_dataset(ULTRAFEEDBACK, split=split)


def rl_prompts(n: int = 4096, seed: int = 0, split: str = "train_prefs") -> Dataset:
    """Prompts used for RL rollouts. Identical across arms; only the reward
    source differs."""
    ds = _ultrafeedback(split)
    ds = ds.map(lambda r: {"prompt": r["prompt"]}, remove_columns=[c for c in ds.column_names if c != "prompt"])
    ds = _clean(ds).shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


def eval_prompts(n: int = 200, seed: int = 0) -> Dataset:
    """Held-out prompts for win-rate. Disjoint from `rl_prompts` by split."""
    ds = _ultrafeedback("test_prefs")
    ds = ds.map(lambda r: {"prompt": r["prompt"]}, remove_columns=[c for c in ds.column_names if c != "prompt"])
    ds = _clean(ds).shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


def preference_pairs(n: int = 8000, seed: int = 0) -> Dataset:
    """(prompt, chosen, rejected) for training / sanity-checking the BERT RM
    and for the DPO baseline."""
    ds = _ultrafeedback("train_prefs").shuffle(seed=seed)

    def fmt(r):
        return {
            "prompt": r["prompt"],
            "chosen": _first_assistant(r["chosen"]),
            "rejected": _first_assistant(r["rejected"]),
        }

    ds = ds.map(fmt, remove_columns=ds.column_names)
    ds = ds.filter(lambda r: bool(r["chosen"]) and bool(r["rejected"]))
    return ds.select(range(min(n, len(ds))))


def _first_assistant(messages: Any) -> str:
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "assistant":
                return m.get("content") or ""
    return messages if isinstance(messages, str) else ""


def sft_dataset(n: int = 8000, seed: int = 0) -> Dataset:
    """SFT on the *chosen* responses -- gives every arm the same R0 start."""
    ds = preference_pairs(n=n, seed=seed)
    return ds.map(
        lambda r: {
            "messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["chosen"]},
            ]
        },
        remove_columns=ds.column_names,
    )


GSM8K_INSTRUCTION = (
    "Solve the problem. Show your reasoning, then give the final numeric answer "
    "on the last line in the form '#### <number>'."
)


def gsm8k(n: int = 250, seed: int = 0, split: str = "test") -> Dataset:
    """Ground-truth task: capability tax and reward-hacking detector (plan 8.2).

    A rubric-following judge can be talked into liking a response; an arithmetic
    answer cannot.
    """
    ds = load_dataset(GSM8K, "main", split=split).shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return ds.map(
        lambda r: {
            "prompt": f"{GSM8K_INSTRUCTION}\n\n{r['question']}",
            "answer_text": r["answer"],
            "gold": _gsm8k_gold(r["answer"]),
        },
        remove_columns=ds.column_names,
    )


def _gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "") if "####" in answer else ""


def fingerprint(ds: Dataset, key: str = "prompt", k: int = 64) -> str:
    """Cheap proof in the logs that two runs really saw the same data."""
    h = hashlib.sha256()
    for i in range(min(k, len(ds))):
        h.update(str(ds[i][key]).encode())
    return f"{h.hexdigest()[:16]}:n={len(ds)}"


def filter_by_prompt_tokens(ds: Dataset, tokenizer, max_tokens: int, key: str = "prompt") -> Dataset:
    """TRL 1.13 dropped GRPOConfig.max_prompt_length, so prompt-length control
    moved to the dataset. We drop over-long prompts rather than truncating them
    mid-sentence: a clipped request is a different task, and the resulting
    reward would be noise. Deterministic, so every arm still sees one set."""
    return ds.filter(
        lambda r: len(tokenizer.encode(r[key], add_special_tokens=False)) <= max_tokens
    )
