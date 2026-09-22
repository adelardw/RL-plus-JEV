"""The prefix critic, generalised beyond any one judge.

The claim the paper rests on is not "Jev is useful" but something structural:
*any judge whose belief can be read as a calibrated probability can serve as a
zero-shot critic over token prefixes*, because

    V*(s_t) = E[R | prompt, tokens_<t]

is a question about a partial state with a yes/no answer. Jev answers it as a
typed value; a hosted instruct model answers it through `logprobs`; a local
model answers it through its logits. All three are read here through one
interface, so the paper can show the mechanism transferring across judges
rather than reporting a property of one vendor's API.

They differ in what the mechanism costs, which is the practical point: a
critic is queried at K positions per completion, so its query count is K times
the reward count, and that constant is what decides whether the idea is
reachable at all.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from ..rubric import PREFIX_VALUE_ITEM, build_state
from .jev_critic import Calibration, JevCritic, _interp, _mean_slope, fit_calibration, probe_positions


class BasePrefixCritic:
    """Shared machinery: probe K prefixes, interpolate, calibrate, report.

    Subclasses provide only `aprefix_probs_batch`, i.e. how this particular
    judge turns (prompt, partial response) into a probability.
    """

    name = "prefix_critic"

    def __init__(self, tokenizer, num_prefixes: int = 6,
                 calibration: Calibration | None = None):
        self.tok = tokenizer
        self.num_prefixes = num_prefixes
        self.calibration = calibration or Calibration()

    async def aprefix_probs_batch(self, prompts: Sequence[str],
                                  prefixes: Sequence[str]) -> list[float]:
        raise NotImplementedError

    # -- identical to the Jev path, by construction -------------------------
    async def avalues(self, prompts, completion_ids, lengths=None):
        B = len(prompts)
        T = max((len(c) for c in completion_ids), default=1)
        lens = list(lengths) if lengths is not None else [len(c) for c in completion_ids]

        flat_prompts: list[str] = []
        flat_prefixes: list[str] = []
        index: list[tuple[int, list[int]]] = []
        for b in range(B):
            L = max(1, int(lens[b]))
            pos = probe_positions(L, self.num_prefixes)
            index.append((b, pos))
            for t in pos:
                flat_prompts.append(prompts[b])
                flat_prefixes.append(
                    self.tok.decode(list(completion_ids[b])[:t], skip_special_tokens=True))

        probs = await self.aprefix_probs_batch(flat_prompts, flat_prefixes)

        values = torch.zeros(B, T, dtype=torch.float32)
        cursor = 0
        curves: list[list[float]] = []
        for b, pos in index:
            p = probs[cursor:cursor + len(pos)]
            cursor += len(pos)
            curves.append(p)
            L = max(1, int(lens[b]))
            values[b, :L] = _interp(
                torch.arange(L, dtype=torch.float32),
                torch.tensor(pos, dtype=torch.float32),
                torch.tensor(p, dtype=torch.float32),
            )
        values = self.calibration.apply(values)
        return values, {
            f"{self.name}/probe_calls": len(flat_prefixes),
            f"{self.name}/mean_prob": sum(probs) / max(1, len(probs)),
            f"{self.name}/mean_slope": _mean_slope(curves),
        }

    def values(self, prompts, completion_ids, lengths=None):
        from ..jevclient import _run_sync
        return _run_sync(self.avalues(prompts, completion_ids, lengths))

    async def afit_calibration(self, prompts, completions, rewards) -> Calibration:
        probs = await self.aprefix_probs_batch(list(prompts), list(completions))
        self.calibration = fit_calibration(probs, rewards)
        return self.calibration

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "num_prefixes": self.num_prefixes,
                "calibration": self.calibration.to_dict()}


class APIJudgePrefixCritic(BasePrefixCritic):
    """A hosted instruct model as the critic, read through logprobs."""

    name = "api_judge_critic"

    def __init__(self, tokenizer, model: str, num_prefixes: int = 6,
                 concurrency: int = 24, calibration: Calibration | None = None):
        super().__init__(tokenizer, num_prefixes, calibration)
        from ..jevclient import Meter
        from ..orclient import ChatClient
        from ..rewards.api_judge import APIJudgeRewardSource

        self.model = model
        self.client = ChatClient(model, concurrency=concurrency,
                                 meter=Meter(name=f"apicritic_{model.replace('/', '_')}"),
                                 temperature=0.0, max_tokens=4)
        self._reader = APIJudgeRewardSource.__new__(APIJudgeRewardSource)
        self._reader.top_logprobs = 12
        self.unreadable = 0

    def _messages(self, prompt: str, prefix: str) -> list[dict]:
        state = (f"[USER REQUEST]\n{prompt}\n\n"
                 f"[PARTIAL ASSISTANT RESPONSE SO FAR]\n{prefix}")
        q = (f"Question: {PREFIX_VALUE_ITEM.instructions}\n"
             f"Answer Yes if: {PREFIX_VALUE_ITEM.criteria['true']}\n"
             f"Answer No if: {PREFIX_VALUE_ITEM.criteria['false']}\n"
             "Answer with one word: Yes or No.")
        return [
            {"role": "system", "content":
             "You are a strict evaluator. Reply with exactly one word."},
            {"role": "user", "content": f"{state}\n\n{q}"},
        ]

    async def aprefix_probs_batch(self, prompts, prefixes) -> list[float]:
        convs = [self._messages(p, pre) for p, pre in zip(prompts, prefixes)]
        raw = await self.client.acomplete_raw(
            convs, logprobs=True, top_logprobs=12, max_tokens=4,
            reasoning={"enabled": False},
            provider={"require_parameters": True},
        )
        out = []
        for d in raw:
            ch = (d.get("choices") or [{}])[0]
            v = self._reader._noul(ch.get("logprobs"))
            if v is None:
                self.unreadable += 1
                txt = ((ch.get("message") or {}).get("content") or "").strip().lower()
                v = 1.0 if txt.startswith("y") else 0.0 if txt.startswith("n") else 0.5
            out.append(v)
        return out

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "model": self.model,
                "unreadable": self.unreadable, **self.client.meter.summary()}


class LocalJudgePrefixCritic(BasePrefixCritic):
    """A local model as the critic, read from its logits. Costs GPU rather
    than dollars, and is the weakest of the three at the sizes that fit."""

    name = "local_judge_critic"

    def __init__(self, tokenizer, judge_source, num_prefixes: int = 6,
                 calibration: Calibration | None = None):
        super().__init__(tokenizer, num_prefixes, calibration)
        self.judge = judge_source          # an LLMJudgeRewardSource

    async def aprefix_probs_batch(self, prompts, prefixes) -> list[float]:
        rendered = [self.judge._render(p, pre, PREFIX_VALUE_ITEM)
                    for p, pre in zip(prompts, prefixes)]
        probs = self.judge._probs(rendered)
        return self.judge._score_item(probs, PREFIX_VALUE_ITEM).tolist()

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), **self.judge.stats()}


def build_prefix_critic(kind: str, tokenizer, **kw):
    if kind == "jev":
        return JevCritic(tokenizer=tokenizer, **kw)
    if kind == "api":
        return APIJudgePrefixCritic(tokenizer=tokenizer, **kw)
    if kind == "local":
        return LocalJudgePrefixCritic(tokenizer=tokenizer, **kw)
    raise ValueError(f"unknown prefix critic {kind!r}")
