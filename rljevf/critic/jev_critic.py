"""Jev as a critic: a frozen, zero-shot, calibrated value function.

The idea. In a language-model MDP with a single terminal reward R, the value
function is exactly

    V*(s_t) = E[ R | prompt, tokens_{<t} ]

-- the expected final quality of a response given the prefix produced so far.
Jev answers precisely that shape of question ("if the assistant continues from
here, will the finished response be correct and helpful?") and returns a
calibrated probability, in ~100 ms for ~$0.00002. So Jev can stand in for PPO's
learned value network directly, which removes the value model from the
optimisation: no value head to initialise, no value loss, no extra optimiser
state, and no bootstrapping from a critic that is itself being trained.

Two things make this workable rather than merely cute:

1. Calibration. Jev's probability lives in [0, 1]; the rubric reward lives in
   [REWARD_MIN, REWARD_MAX]. We fit an affine map a*p + b on held-out
   (prefix, reward) pairs by least squares and report the R^2, so the TD errors
   are in reward units rather than in arbitrary ones.

2. Sparse probing plus interpolation. Querying every token position is
   unnecessary: we probe K positions per completion and interpolate V between
   them. K is an ablation axis.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, asdict
from typing import Any, Sequence

import torch

from ..jevclient import JevClient, Meter
from ..rubric import PREFIX_VALUE_QUESTIONS, build_state


@dataclass
class Calibration:
    """V_reward = a * p_jev + b."""

    a: float = 1.0
    b: float = 0.0
    r2: float = float("nan")
    n: int = 0
    pearson: float = float("nan")

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        return self.a * p + self.b

    def to_dict(self) -> dict:
        return asdict(self)


def fit_calibration(p: Sequence[float], r: Sequence[float]) -> Calibration:
    x = torch.tensor(list(p), dtype=torch.float64)
    y = torch.tensor(list(r), dtype=torch.float64)
    n = x.numel()
    if n < 2 or float(x.var()) < 1e-12:
        return Calibration(a=1.0, b=0.0, r2=float("nan"), n=int(n))
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    b = ym - a * xm
    pred = a * x + b
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1.0 - float(ss_res / ss_tot) if float(ss_tot) > 0 else float("nan")
    # population moments on both sides, otherwise r != sqrt(r2)
    denom = float(x.std(correction=0) * y.std(correction=0))
    pearson = float(((x - xm) * (y - ym)).mean()) / denom if denom > 0 else float("nan")
    return Calibration(a=float(a), b=float(b), r2=r2, n=int(n), pearson=pearson)


def probe_positions(seq_len: int, k: int) -> list[int]:
    """K cut points in [0, seq_len-1]; 0 means 'nothing generated yet'."""
    if seq_len <= 1:
        return [0]
    k = max(2, min(k, seq_len))
    step = (seq_len - 1) / (k - 1)
    return sorted({int(round(i * step)) for i in range(k)})


class JevCritic:
    """Values token prefixes with Jev. Frozen: no parameters, no optimiser."""

    name = "jev_critic"

    def __init__(
        self,
        tokenizer,
        client: JevClient | None = None,
        num_prefixes: int = 6,
        calibration: Calibration | None = None,
        questions: dict | None = None,
    ):
        self.tok = tokenizer
        self.client = client or JevClient(meter=Meter(name="jev_critic"))
        self.num_prefixes = num_prefixes
        self.calibration = calibration or Calibration()
        self.questions = questions or PREFIX_VALUE_QUESTIONS
        self._qkey = next(iter(self.questions))

    # -- raw probabilities ---------------------------------------------------
    async def aprefix_probs(
        self, prompt: str, prefixes: Sequence[str]
    ) -> list[float]:
        states = [build_state(prompt, pre, partial=True) for pre in prefixes]
        res = await self.client.aask_many(states, self.questions)
        return [float(r["answers"][self._qkey]["noul"]) for r in res]

    # -- token-level value curve --------------------------------------------
    async def avalues(
        self,
        prompts: Sequence[str],
        completion_ids: Sequence[Sequence[int]],
        lengths: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """-> (B, T) values in *reward units*, plus diagnostics.

        V[b, t] estimates E[R | prompt, tokens_{<t}]. Positions past a
        sequence's own length are zero-filled; the caller masks them.
        """
        B = len(prompts)
        T = max((len(c) for c in completion_ids), default=1)
        lens = list(lengths) if lengths is not None else [len(c) for c in completion_ids]

        flat_states: list[Any] = []
        index: list[tuple[int, list[int]]] = []  # (batch idx, probe token positions)
        for b in range(B):
            L = max(1, int(lens[b]))
            pos = probe_positions(L, self.num_prefixes)
            index.append((b, pos))
            for t in pos:
                text = self.tok.decode(list(completion_ids[b])[:t], skip_special_tokens=True)
                flat_states.append(build_state(prompts[b], text, partial=True))

        res = await self.client.aask_many(flat_states, self.questions)
        probs = [float(r["answers"][self._qkey]["noul"]) for r in res]

        values = torch.zeros(B, T, dtype=torch.float32)
        cursor = 0
        raw_curves: list[list[float]] = []
        for b, pos in index:
            p = probs[cursor : cursor + len(pos)]
            cursor += len(pos)
            raw_curves.append(p)
            L = max(1, int(lens[b]))
            xs = torch.tensor(pos, dtype=torch.float32)
            ys = torch.tensor(p, dtype=torch.float32)
            grid = torch.arange(L, dtype=torch.float32)
            values[b, :L] = _interp(grid, xs, ys)

        values = self.calibration.apply(values)
        diag = {
            "jev_critic/probe_calls": len(flat_states),
            "jev_critic/mean_prob": sum(probs) / max(1, len(probs)),
            "jev_critic/mean_slope": _mean_slope(raw_curves),
        }
        return values, diag

    def values(self, prompts, completion_ids, lengths=None):
        from ..jevclient import _run_sync

        return _run_sync(self.avalues(prompts, completion_ids, lengths))

    # -- calibration ---------------------------------------------------------
    async def afit_calibration(
        self, prompts: Sequence[str], completions: Sequence[str], rewards: Sequence[float]
    ) -> Calibration:
        """Regress the terminal rubric reward on Jev's prefix probability at
        the *complete* response. Reported in the paper (R^2 / Pearson r)."""
        probs = [
            float(r["answers"][self._qkey]["noul"])
            for r in await self.client.aask_many(
                [build_state(p, c, partial=True) for p, c in zip(prompts, completions)],
                self.questions,
            )
        ]
        self.calibration = fit_calibration(probs, rewards)
        return self.calibration

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_prefixes": self.num_prefixes,
            "calibration": self.calibration.to_dict(),
            **self.client.meter.summary(),
        }


def _interp(grid: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation, flat outside the probe range."""
    if xs.numel() == 1:
        return torch.full_like(grid, float(ys[0]))
    idx = torch.searchsorted(xs, grid.clamp(float(xs[0]), float(xs[-1])), right=True)
    idx = idx.clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    w = torch.where(x1 > x0, (grid - x0) / (x1 - x0).clamp_min(1e-6), torch.zeros_like(grid))
    return (y0 + w.clamp(0, 1) * (y1 - y0)).to(torch.float32)


def _mean_slope(curves: list[list[float]]) -> float:
    """Mean end-minus-start of the value curve: positive if Jev tends to grow
    more confident as responses complete."""
    d = [c[-1] - c[0] for c in curves if len(c) >= 2]
    return sum(d) / len(d) if d else 0.0


class LearnedValueCritic(torch.nn.Module):
    """Control arm: the conventional PPO value head on the policy backbone."""

    name = "learned_critic"

    def __init__(self, backbone, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        h = hidden_size or backbone.config.hidden_size
        self.v_head = torch.nn.Linear(h, 1)
        torch.nn.init.normal_(self.v_head.weight, std=1.0 / (h + 1) ** 0.5)
        torch.nn.init.zeros_(self.v_head.bias)

    def forward(self, input_ids, attention_mask=None) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return self.v_head(out.hidden_states[-1]).squeeze(-1)

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "params": sum(p.numel() for p in self.v_head.parameters())}


async def _aprefix_probs_batch(self, prompts, prefixes):
    """One Jev call per (prompt, prefix) pair, all in flight together."""
    states = [build_state(p, pre, partial=True) for p, pre in zip(prompts, prefixes)]
    res = await self.client.aask_many(states, self.questions)
    return [float(r["answers"][self._qkey]["noul"]) for r in res]


JevCritic.aprefix_probs_batch = _aprefix_probs_batch
