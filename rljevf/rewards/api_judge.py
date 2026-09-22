"""RLAIF with a hosted judge, read from logits rather than by parsing prose.

The fairness protocol (paper 3.2) requires every judge to return the same kind
of object Jev does: a calibrated number per rubric question. OpenRouter exposes
`logprobs` with `top_logprobs`, so a hosted judge can be read the same way as
the local one --

    noul   ->  P("Yes") / (P("Yes") + P("No"))
    score  ->  renormalised distribution over the level digits

-- provided reasoning is switched off, since a reasoning model spends its first
tokens on thought and returns nothing to read.

Prompt order is deliberate. The five rubric questions about one completion
share a long prefix (system + state) and differ only in a short suffix, so the
state goes *first* and the question last: that is what the provider's prefix
cache can reuse, and it is the opposite of the usual "instructions first"
advice, which optimises the wrong axis here.
"""
from __future__ import annotations

import asyncio
import math
from typing import Any, Sequence

from ..jevclient import Meter
from ..orclient import ChatClient
from ..rubric import GENERAL_RUBRIC, Rubric, RubricItem
from .base import BaseRewardSource, texts

_SYSTEM = (
    "You are a strict evaluator. You are shown a user request and an AI assistant's "
    "response, then asked one question about the response. Reply with exactly one word."
)

_YES = ("yes", "y", "true")
_NO = ("no", "n", "false")


class APIJudgeRewardSource(BaseRewardSource):
    """A hosted instruct model as the RLAIF judge, read parse-free."""

    def __init__(
        self,
        model: str,
        rubric: Rubric = GENERAL_RUBRIC,
        concurrency: int = 24,
        name: str = "api_judge",
        max_state_chars: int = 6000,
        unit_scale: bool = False,
        top_logprobs: int = 12,
    ):
        self.name = name
        self.model = model
        self.rubric = rubric
        self.unit_scale = unit_scale
        self.top_logprobs = top_logprobs
        self.max_state_chars = max_state_chars
        self.client = ChatClient(
            model, concurrency=concurrency, meter=Meter(name=f"apijudge_{name}"),
            temperature=0.0, max_tokens=4,
        )
        self.unreadable = 0          # calls whose logprobs could not be read
        self.total_calls = 0

    # -- prompts ------------------------------------------------------------
    def _question_text(self, item: RubricItem) -> str:
        if item.kind == "noul":
            q = f"Question: {item.instructions}"
            if item.criteria:
                q += (f"\nAnswer Yes if: {item.criteria['true']}"
                      f"\nAnswer No if: {item.criteria['false']}")
            return q + "\nAnswer with one word: Yes or No."
        levels = "\n".join(f"{i}. {c}" for i, c in enumerate(item.criteria))
        return (f"Question: {item.instructions}\nLevels:\n{levels}\n"
                f"Reply with a single digit 0-{len(item.criteria) - 1}.")

    def _messages(self, prompt: str, completion: str, item: RubricItem) -> list[dict]:
        state = f"[USER REQUEST]\n{prompt}\n\n[ASSISTANT RESPONSE]\n{completion}"
        if len(state) > self.max_state_chars:      # keep the response, clip the request
            state = state[-self.max_state_chars:]
        return [
            {"role": "system", "content": _SYSTEM},
            # state before question: the five calls about one completion then
            # share the expensive prefix and only the last lines differ
            {"role": "user", "content": f"{state}\n\n{self._question_text(item)}"},
        ]

    # -- reading the distribution -------------------------------------------
    @staticmethod
    def _first_readable(logprobs: Any, wanted: tuple[str, ...]) -> dict[str, float] | None:
        content = (logprobs or {}).get("content") or []
        for pos in content[:4]:
            top = pos.get("top_logprobs") or []
            probs: dict[str, float] = {}
            for t in top:
                tok = str(t.get("token", "")).strip().lower()
                if not tok:
                    continue
                probs[tok] = probs.get(tok, 0.0) + math.exp(t["logprob"])
            if any(any(tok.startswith(w) for w in wanted) for tok in probs):
                return probs
        return None

    def _noul(self, logprobs: Any) -> float | None:
        probs = self._first_readable(logprobs, _YES + _NO)
        if probs is None:
            return None
        y = sum(v for k, v in probs.items() if any(k.startswith(w) for w in _YES))
        n = sum(v for k, v in probs.items() if any(k.startswith(w) for w in _NO))
        return y / (y + n) if (y + n) > 0 else None

    def _score(self, logprobs: Any, levels: int) -> float | None:
        digits = tuple(str(i) for i in range(levels))
        probs = self._first_readable(logprobs, digits)
        if probs is None:
            return None
        mass = {d: sum(v for k, v in probs.items() if k.startswith(d)) for d in digits}
        total = sum(mass.values())
        if total <= 0:
            return None
        exp = sum(int(d) * m for d, m in mass.items()) / total
        return exp / max(1, levels - 1)

    # -- scoring -------------------------------------------------------------
    async def __call__(
        self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any
    ) -> list[float]:
        P, C = texts(prompts), texts(completions)
        items = list(self.rubric)
        convs = [self._messages(p, c, it) for it in items for p, c in zip(P, C)]

        raw = await self.client.acomplete_raw(
            convs, logprobs=True, top_logprobs=self.top_logprobs,
            reasoning={"enabled": False}, max_tokens=4,
            # OpenRouter load-balances across providers and only some return
            # logprobs; without this the judge silently degrades to parsing
            # text for whichever calls happened to be routed elsewhere.
            provider={"require_parameters": True},
        )

        self.total_calls += len(convs)
        per_item: dict[str, list[float]] = {}
        k = 0
        for it in items:
            vals: list[float] = []
            for _ in range(len(P)):
                d = raw[k]; k += 1
                lp = ((d.get("choices") or [{}])[0] or {}).get("logprobs")
                v = self._noul(lp) if it.kind == "noul" else self._score(lp, it.n_levels)
                if v is None:
                    # Fall back to the emitted word rather than dropping the
                    # sample; counted so the paper can report how often.
                    self.unreadable += 1
                    txt = (((d.get("choices") or [{}])[0] or {}).get("message") or {}).get("content") or ""
                    v = _fallback(txt, it)
                vals.append(v)
            per_item[it.key] = vals

        self._log(kwargs, self.name,
                  {k: sum(v) / len(v) for k, v in per_item.items() if v})
        return [
            self.rubric.scalarise({k: v[i] for k, v in per_item.items()},
                                  unit=self.unit_scale)
            for i in range(len(P))
        ]

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model,
                "unreadable_logprobs": self.unreadable,
                "unreadable_fraction": round(self.unreadable / max(1, self.total_calls), 4),
                **self.client.meter.summary()}


def _fallback(text: str, item: RubricItem) -> float:
    t = (text or "").strip().lower()
    if item.kind == "noul":
        if t.startswith(_YES):
            return 1.0
        if t.startswith(_NO):
            return 0.0
        return 0.5
    for i in range(item.n_levels):
        if t.startswith(str(i)):
            return i / max(1, item.n_levels - 1)
    return 0.5
