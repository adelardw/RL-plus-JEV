"""Common contract for every reward source.

A reward source is a callable with TRL's signature

    (prompts, completions, **kwargs) -> list[float]

so the trainer is byte-identical across arms and only this object changes.
Each source also exposes `.stats()` for the cost/latency table (paper 8.4).
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence


def as_text(x: Any) -> str:
    """TRL hands completions back as plain strings (standard datasets) or as
    lists of chat messages (conversational datasets). Normalise both."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(
            m.get("content", "") if isinstance(m, dict) else str(m) for m in x
        )
    if isinstance(x, dict):
        return x.get("content", "")
    return str(x)


def texts(xs: Sequence[Any]) -> list[str]:
    return [as_text(x) for x in xs]


class RewardSource(Protocol):
    name: str

    def __call__(
        self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any
    ) -> list[float]: ...

    def stats(self) -> dict[str, Any]: ...


class BaseRewardSource:
    name: str = "base"

    def stats(self) -> dict[str, Any]:
        return {"name": self.name}

    # Sub-rubric logging: forwarded to TRL's log_metric when available, so the
    # per-component curves (self_promo over training, etc.) land in W&B.
    @staticmethod
    def _log(kwargs: dict, prefix: str, values: dict[str, float]) -> None:
        log_metric = kwargs.get("log_metric")
        if log_metric is None:
            return
        for k, v in values.items():
            try:
                log_metric(f"{prefix}/{k}", float(v))
            except Exception:
                pass
