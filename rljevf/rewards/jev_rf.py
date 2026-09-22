"""R3 -- Jev as the reward function.

One /alpha/decisions call per completion carries the whole rubric, because Jev
evaluates every question against the same state in parallel. That is the
structural reason this arm is cheap: five rubric questions cost one request.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..jevclient import JevClient, Meter
from ..rubric import GENERAL_RUBRIC, Rubric, build_state
from ..sg2jev import RubricSource, StaticRubricSource
from .base import BaseRewardSource, texts


class JevRewardSource(BaseRewardSource):
    name = "jev"

    def __init__(
        self,
        client: JevClient | None = None,
        rubric: Rubric = GENERAL_RUBRIC,
        rubric_source: RubricSource | None = None,
        unit_scale: bool = False,
    ):
        self.client = client or JevClient(meter=Meter(name="jev_reward"))
        self.rubric = rubric
        self.unit_scale = unit_scale
        # StructuredGeneration2JEV: decides which questions get asked.
        self.rubric_source = rubric_source or StaticRubricSource(rubric)

    async def __call__(
        self, prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any
    ) -> list[float]:
        P, C = texts(prompts), texts(completions)
        rubrics = await self.rubric_source.arubrics_for(P, C, **kwargs)

        # Group by identical question-set so a static rubric stays one batch.
        rewards: list[float] = [0.0] * len(P)
        agg: dict[str, list[float]] = {}
        buckets: dict[str, list[int]] = {}
        for i, rb in enumerate(rubrics):
            buckets.setdefault(rb.name, []).append(i)

        by_name = {rb.name: rb for rb in rubrics}
        for name, idxs in buckets.items():
            rb = by_name[name]
            states = [build_state(P[i], C[i]) for i in idxs]
            responses = await self.client.aask_many(states, rb.jev_questions())
            for i, resp in zip(idxs, responses):
                answers = resp["answers"]
                rewards[i] = rb.scalarise_jev(answers, unit=self.unit_scale)
                for k, v in rb.breakdown_jev(answers).items():
                    agg.setdefault(k, []).append(v)

        self._log(kwargs, "jev", {k: sum(v) / len(v) for k, v in agg.items() if v})
        return rewards

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, **self.client.meter.summary()}
