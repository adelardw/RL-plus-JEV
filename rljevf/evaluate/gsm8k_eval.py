"""GSM8K accuracy: the capability tax / hacking detector.

A judge-based reward can be satisfied by a response that only *looks* right.
Arithmetic cannot. A run whose training reward climbs while GSM8K falls is
optimising the judge, not the task.
"""
from __future__ import annotations

import re
from typing import Sequence

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str:
    """Prefer the '#### x' form we asked for; otherwise the last number."""
    if "####" in text:
        tail = text.split("####")[-1]
        m = _NUM.search(tail)
        if m:
            return _norm(m.group())
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else ""


def _norm(s: str) -> str:
    s = s.replace(",", "").strip().rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def accuracy(completions: Sequence[str], golds: Sequence[str]) -> dict[str, float]:
    hits = [extract_answer(c) == _norm(g) for c, g in zip(completions, golds)]
    parsed = [bool(extract_answer(c)) for c in completions]
    n = max(1, len(hits))
    return {
        "gsm8k_acc": sum(hits) / n,
        "gsm8k_parse_rate": sum(parsed) / n,
        "gsm8k_n": len(hits),
    }
