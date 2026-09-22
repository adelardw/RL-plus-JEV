"""Head-to-head win-rate against the SFT baseline, length-controlled.

Three precautions, each of which the literature has been burned by:

  * The judge shares no family with any *training* reward source (plan 8.1),
    so we are not measuring how well a policy learned to please its own judge.
  * Every pair is judged twice with the positions swapped, and the two verdicts
    averaged, which removes position bias.
  * The headline number is length-controlled in the AlpacaEval-2 sense: we fit
    P(win) = sigmoid(theta_system + gamma * standardised length difference) and
    report sigmoid(theta_system), i.e. the win-rate the system would get if it
    wrote answers the same length as the baseline. Without this, "the policy
    learned to pad" reads as "the policy got better".
"""
from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, asdict
from typing import Sequence

import torch

from ..orclient import ChatClient, extract_json

_JUDGE_SYSTEM = (
    "You compare two AI assistant responses to the same user request and decide which "
    "is better overall: more correct, more genuinely useful, and free of padding, "
    "evasion and self-congratulation. Length is not quality. "
    'Reply with JSON only: {"winner": "A"} or {"winner": "B"} or {"winner": "tie"}.'
)

_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}},
            "required": ["winner"],
            "additionalProperties": False,
        },
    },
}


def _pair_msg(prompt: str, a: str, b: str) -> list[dict]:
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"[USER REQUEST]\n{prompt}\n\n"
                f"[RESPONSE A]\n{a}\n\n[RESPONSE B]\n{b}\n\nWhich is better?"
            ),
        },
    ]


@dataclass
class WinRateResult:
    n: int
    raw_win_rate: float
    lc_win_rate: float
    ties: float
    ci_low: float
    ci_high: float
    mean_len_system: float
    mean_len_baseline: float
    length_coef: float

    def to_dict(self) -> dict:
        return asdict(self)


async def awin_rate(
    prompts: Sequence[str],
    system: Sequence[str],
    baseline: Sequence[str],
    judge: ChatClient,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> WinRateResult:
    # each pair judged in both orders
    convs: list[list[dict]] = []
    for p, s, b in zip(prompts, system, baseline):
        convs.append(_pair_msg(p, s, b))   # system as A
        convs.append(_pair_msg(p, b, s))   # system as B
    raws = await judge.acomplete_many(convs, response_format=_JUDGE_SCHEMA)

    wins: list[float] = []
    for i in range(len(prompts)):
        v = []
        for j, sys_is in ((2 * i, "A"), (2 * i + 1, "B")):
            try:
                w = extract_json(raws[j]).get("winner", "tie")
            except Exception:
                w = "tie"
            v.append(1.0 if w == sys_is else 0.5 if w == "tie" else 0.0)
        wins.append(sum(v) / 2)

    w = torch.tensor(wins, dtype=torch.float64)
    ls = torch.tensor([len(x) for x in system], dtype=torch.float64)
    lb = torch.tensor([len(x) for x in baseline], dtype=torch.float64)
    dl = ls - lb
    dl = (dl - dl.mean()) / dl.std().clamp_min(1e-6)

    theta, gamma = _fit_lc(w, dl)
    lc = 1.0 / (1.0 + math.exp(-theta))

    rng = random.Random(seed)
    idx = range(len(wins))
    boots = []
    for _ in range(n_bootstrap):
        s = [wins[rng.choice(idx)] for _ in idx]
        boots.append(sum(s) / len(s))
    boots.sort()

    return WinRateResult(
        n=len(prompts),
        raw_win_rate=float(w.mean()),
        lc_win_rate=lc,
        ties=float((w == 0.5).float().mean()),
        ci_low=boots[int(0.025 * n_bootstrap)],
        ci_high=boots[int(0.975 * n_bootstrap) - 1],
        mean_len_system=float(ls.mean()),
        mean_len_baseline=float(lb.mean()),
        length_coef=gamma,
    )


def _fit_lc(w: torch.Tensor, dl: torch.Tensor, iters: int = 800) -> tuple[float, float]:
    """Logistic fit on soft targets; returns (theta_system, gamma_length)."""
    theta = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    gamma = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta, gamma], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = theta + gamma * dl
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, w)
        loss.backward()
        return loss

    opt.step(closure)
    return float(theta.detach()), float(gamma.detach())


def win_rate(*a, **kw) -> WinRateResult:
    from ..jevclient import _run_sync

    return _run_sync(awin_rate(*a, **kw))
