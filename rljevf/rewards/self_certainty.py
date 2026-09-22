"""R5 (optional) -- reward with no external judge at all.

Self-certainty: the mean KL from the uniform distribution of the policy's own
next-token distributions, i.e. log|V| - H(p), averaged over generated tokens.
Included as the degenerate corner of the design space: it costs nothing and
consults nobody, and we expect it to collapse to short confident text on open
-ended prompts, which makes it a useful control for what "reward went up" is
worth on its own.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .base import BaseRewardSource, texts


class SelfCertaintyRewardSource(BaseRewardSource):
    name = "self_certainty"

    def __init__(self, model, tokenizer, batch_size: int = 4, max_length: int = 1024):
        self.model = model
        self.tok = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_forward = 0

    @torch.no_grad()
    def __call__(self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> list[float]:
        P, C = texts(prompts), texts(completions)
        device = next(self.model.parameters()).device
        self.tok.padding_side = "right"
        out: list[float] = []
        logV = math.log(len(self.tok))
        for i in range(0, len(P), self.batch_size):
            bp, bc = P[i : i + self.batch_size], C[i : i + self.batch_size]
            prefixes = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in bp
            ]
            full = [pre + c for pre, c in zip(prefixes, bc)]
            enc = self.tok(full, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length, add_special_tokens=False).to(device)
            logits = self.model(**enc, use_cache=False).logits.float()
            logp = torch.log_softmax(logits, -1)
            ent = -(logp.exp() * logp).sum(-1)              # (B, T)
            self.n_forward += len(bp)
            for b in range(len(bp)):
                s = len(self.tok.encode(prefixes[b], add_special_tokens=False))
                e = int(enc["attention_mask"][b].sum())
                if e <= s:
                    out.append(0.0)
                    continue
                out.append(float(logV - ent[b, s - 1 : e - 1].mean()))
        self._log(kwargs, "self_certainty", {"mean": sum(out) / max(1, len(out))})
        return out

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "forward_passes": self.n_forward}
