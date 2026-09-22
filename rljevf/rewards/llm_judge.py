"""R2/R3 -- an LLM as the judge (RLAIF), and the self-judge.

Fairness note (plan 6.2). A text-generating judge is normally read by parsing
its output, which is lossy and error-prone, and would hand Jev an unearned
advantage. Instead we read *probabilities straight off the logits*:

    noul   ->  P("yes") / (P("yes") + P("no"))
    score  ->  softmax over the level-index tokens, then the expected level

so this judge returns exactly the same kind of object Jev does -- a calibrated
number per rubric question -- and the two arms differ only in the machine.
Every question is its own prefill over a shared prefix, again mirroring Jev.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rubric import GENERAL_RUBRIC, Rubric, RubricItem
from .base import BaseRewardSource, texts

_SYSTEM = (
    "You are a strict evaluator. You are shown a user request and an AI assistant's "
    "response, then asked one question about the response. Answer with a single word."
)

_YES = ["Yes", " Yes", "yes", " yes", "YES"]
_NO = ["No", " No", "no", " no", "NO"]


def _ids_for(tok, words: list[str]) -> list[int]:
    out: set[int] = set()
    for w in words:
        enc = tok.encode(w, add_special_tokens=False)
        if enc:
            out.add(enc[0])
    return sorted(out)


class LLMJudgeRewardSource(BaseRewardSource):
    def __init__(
        self,
        model_name: str | None = None,
        model=None,
        tokenizer=None,
        rubric: Rubric = GENERAL_RUBRIC,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int = 16,
        name: str = "llm_judge",
        max_state_tokens: int = 1024,
        unit_scale: bool = False,
    ):
        self.name = name
        self.rubric = rubric
        self.batch_size = batch_size
        self.max_state_tokens = max_state_tokens
        self.unit_scale = unit_scale
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        if dtype is None:
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        if model is None:
            assert model_name, "pass model_name or a model"
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            model.to(self.device)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self._yes = _ids_for(self.tok, _YES)
        self._no = _ids_for(self.tok, _NO)
        self._digits = {
            n: _ids_for(self.tok, [str(n), f" {n}"]) for n in range(10)
        }
        self.n_forward = 0

    # -- prompt construction -------------------------------------------------
    def _question_text(self, item: RubricItem) -> str:
        if item.kind == "noul":
            q = f"Question: {item.instructions}"
            if item.criteria:
                q += (
                    f"\nAnswer Yes if: {item.criteria['true']}"
                    f"\nAnswer No if: {item.criteria['false']}"
                )
            return q + "\nAnswer Yes or No."
        levels = "\n".join(f"{i}. {c}" for i, c in enumerate(item.criteria))
        return (
            f"Question: {item.instructions}\nLevels:\n{levels}\n"
            f"Answer with a single digit 0-{len(item.criteria) - 1}."
        )

    def _render(self, prompt: str, completion: str, item: RubricItem) -> str:
        state = f"[USER REQUEST]\n{prompt}\n\n[ASSISTANT RESPONSE]\n{completion}"
        ids = self.tok.encode(state, add_special_tokens=False)
        if len(ids) > self.max_state_tokens:  # keep the response, clip the request
            state = self.tok.decode(ids[-self.max_state_tokens :])
        msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{state}\n\n{self._question_text(item)}"},
        ]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # -- scoring -------------------------------------------------------------
    @torch.no_grad()
    def _probs(self, rendered: list[str]) -> torch.Tensor:
        """-> (N, V) next-token probability rows."""
        rows = []
        for i in range(0, len(rendered), self.batch_size):
            chunk = rendered[i : i + self.batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits[:, -1, :].float()
            rows.append(torch.softmax(logits, dim=-1).cpu())
            self.n_forward += len(chunk)
        return torch.cat(rows, dim=0)

    def _score_item(self, probs: torch.Tensor, item: RubricItem) -> torch.Tensor:
        if item.kind == "noul":
            y = probs[:, self._yes].sum(-1)
            n = probs[:, self._no].sum(-1)
            return y / (y + n).clamp_min(1e-9)
        L = item.n_levels
        cols = torch.stack([probs[:, self._digits[k]].sum(-1) for k in range(L)], dim=-1)
        cols = cols / cols.sum(-1, keepdim=True).clamp_min(1e-9)
        levels = torch.arange(L, dtype=cols.dtype)
        return (cols * levels).sum(-1) / max(1, L - 1)

    def __call__(
        self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any
    ) -> list[float]:
        P, C = texts(prompts), texts(completions)
        per_item: dict[str, torch.Tensor] = {}
        for item in self.rubric:
            rendered = [self._render(p, c, item) for p, c in zip(P, C)]
            per_item[item.key] = self._score_item(self._probs(rendered), item)

        self._log(
            kwargs, self.name, {k: float(v.mean()) for k, v in per_item.items()}
        )
        out = []
        for i in range(len(P)):
            out.append(
                self.rubric.scalarise(
                    {k: float(v[i]) for k, v in per_item.items()}, unit=self.unit_scale
                )
            )
        return out

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "judge_forward_passes": self.n_forward}


class SelfJudgeRewardSource(LLMJudgeRewardSource):
    """R2 -- a frozen copy of the SFT checkpoint grades its own outputs."""

    def __init__(self, *a, **kw):
        kw.setdefault("name", "self_judge")
        super().__init__(*a, **kw)
