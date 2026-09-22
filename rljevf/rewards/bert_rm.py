"""R1 -- a classical Bradley-Terry reward model (DeBERTa/BERT classifier).

The conventional baseline: a small encoder trained on human preference pairs,
emitting one scalar. Its structural limits are part of the story -- a 512-token
window and a fixed preference domain (plan 6.3).
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import BaseRewardSource, texts


class BertRMRewardSource(BaseRewardSource):
    name = "bert_rm"

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        if dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=dtype
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_forward = 0

    def _truncate_prompt(self, prompt: str, completion: str) -> str:
        """Budget the 512 tokens: the response is never cut, the request is
        clipped from the left (plan 6.3)."""
        c_ids = self.tok.encode(completion, add_special_tokens=False)
        room = max(16, self.max_length - len(c_ids) - 8)
        p_ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(p_ids) > room:
            prompt = self.tok.decode(p_ids[-room:])
        return prompt

    @torch.no_grad()
    def __call__(
        self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any
    ) -> list[float]:
        P, C = texts(prompts), texts(completions)
        out: list[float] = []
        for i in range(0, len(P), self.batch_size):
            bp = [self._truncate_prompt(p, c) for p, c in zip(P[i : i + self.batch_size], C[i : i + self.batch_size])]
            bc = C[i : i + self.batch_size]
            enc = self.tok(
                bp, bc, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.float()
            out.extend(logits[:, 0].cpu().tolist())
            self.n_forward += len(bp)
        self._log(kwargs, "bert_rm", {"mean_logit": sum(out) / max(1, len(out))})
        return out

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "rm_forward_passes": self.n_forward}
