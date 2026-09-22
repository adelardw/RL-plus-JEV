"""The evaluation rubric, defined once and rendered into every judge's native
format.

This module is the backbone of the paper's fairness claim: Jev, the RLAIF
judge and the self-judge answer *the same questions* with *the same
scalarisation*. The only thing that differs between those arms is the machine
that produces the probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RubricItem:
    key: str
    kind: Literal["noul", "score"]
    instructions: str
    weight: float = 1.0
    sign: int = 1                          # +1 desirable, -1 undesirable
    criteria: Any = None                   # noul: {"true":..,"false":..}; score: [levels]

    @property
    def n_levels(self) -> int:
        return len(self.criteria) if self.kind == "score" else 2

    def jev_question(self) -> dict:
        q: dict[str, Any] = {"type": self.kind, "instructions": self.instructions}
        if self.criteria is not None:
            q["criteria"] = self.criteria
        return q

    def normalise(self, answer: dict) -> float:
        """Jev answer -> [0, 1]."""
        if self.kind == "noul":
            return float(answer["noul"])
        # score: the API returns the expected level index in [0, L-1]
        return float(answer["score"]) / max(1, self.n_levels - 1)


@dataclass(frozen=True)
class Rubric:
    items: tuple[RubricItem, ...]
    name: str = "general-helpfulness-v1"

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def keys(self) -> list[str]:
        return [i.key for i in self.items]

    def jev_questions(self) -> dict[str, dict]:
        return {i.key: i.jev_question() for i in self.items}

    # -- scalarisation ------------------------------------------------------
    @property
    def reward_range(self) -> tuple[float, float]:
        lo = -sum(i.weight for i in self.items if i.sign < 0)
        hi = sum(i.weight for i in self.items if i.sign > 0)
        return lo, hi

    def scalarise(self, normalised: dict[str, float], unit: bool = False) -> float:
        """Identical for every judge. Range for the default rubric: [-3, +2].

        `unit=True` rescales to [-1, +1]. Only needed when the number of rubric
        items varies across arms (the SG2JEV ablation), so that a longer
        generated rubric does not silently inflate the reward scale.
        """
        raw = sum(i.weight * i.sign * normalised[i.key] for i in self.items)
        if not unit:
            return raw
        lo, hi = self.reward_range
        span = max(abs(lo), abs(hi)) or 1.0
        return raw / span

    def scalarise_jev(self, answers: dict[str, dict], unit: bool = False) -> float:
        return self.scalarise(
            {i.key: i.normalise(answers[i.key]) for i in self.items}, unit=unit
        )

    def breakdown_jev(self, answers: dict[str, dict]) -> dict[str, float]:
        return {i.key: i.normalise(answers[i.key]) for i in self.items}


# --------------------------------------------------------------------------- #
#  The shared rubric (plan section 6.1)
# --------------------------------------------------------------------------- #
GENERAL_RUBRIC = Rubric(
    items=(
        RubricItem(
            key="correct",
            kind="noul",
            instructions=(
                "Does the assistant's response answer the user's request correctly, "
                "with no factual, logical or arithmetic errors?"
            ),
            criteria={
                "true": "Everything the response asserts is accurate and it addresses what was asked.",
                "false": "The response contains an error, or answers something other than what was asked.",
            },
            weight=1.0,
            sign=+1,
        ),
        RubricItem(
            key="helpful",
            kind="score",
            instructions="How completely does the response serve the user's actual need?",
            criteria=[
                "Useless: off-topic, empty, or gives the user nothing they can act on.",
                "Partial: addresses part of the request, or answers it but leaves obvious gaps.",
                "Complete: fully addresses the request with the detail the user needed.",
            ],
            weight=1.0,
            sign=+1,
        ),
        RubricItem(
            key="padded",
            kind="noul",
            instructions=(
                "Does the response contain filler, restatement of the question, "
                "repetition, or padding that carries no information?"
            ),
            criteria={
                "true": "There is material that could be deleted without losing content.",
                "false": "Essentially every sentence carries information.",
            },
            weight=1.0,
            sign=-1,
        ),
        RubricItem(
            key="evasive",
            kind="noul",
            instructions=(
                "Does the response dodge the request -- refusing, deferring, or "
                "talking around it -- instead of answering it?"
            ),
            criteria={
                "true": "The response avoids committing to an answer.",
                "false": "The response commits to an answer.",
            },
            weight=1.0,
            sign=-1,
        ),
        # Mandatory: the policy's own text sits inside the judge's state, so the
        # policy can learn to assert its own quality at the judge (plan 6.1).
        RubricItem(
            key="self_promo",
            kind="noul",
            instructions=(
                "Does the response make claims about its own quality, correctness, "
                "completeness or helpfulness?"
            ),
            criteria={
                "true": "The response praises or vouches for itself, e.g. 'this is a complete and accurate answer'.",
                "false": "The response simply answers, without commenting on itself.",
            },
            weight=1.0,
            sign=-1,
        ),
    )
)

# Used by the Jev-as-critic arm: a single forward-looking question about a
# *partial* response. Validated empirically to be monotone in prefix quality.
PREFIX_VALUE_ITEM = RubricItem(
    key="will_be_good",
    kind="noul",
    instructions=(
        "The assistant has produced only part of its response so far. "
        "If it continues in the same direction and finishes, will the completed "
        "response be a correct and genuinely helpful answer to the user's request?"
    ),
    criteria={
        "true": "The partial response is on track to become a correct, helpful, non-padded answer.",
        "false": "The partial response is off-track: wrong, evasive, padded, self-promoting, or going nowhere useful.",
    },
)
PREFIX_VALUE_QUESTIONS = {PREFIX_VALUE_ITEM.key: PREFIX_VALUE_ITEM.jev_question()}


def build_state(prompt: str, completion: str, partial: bool = False) -> dict:
    """The one canonical way a (prompt, completion) pair is shown to Jev."""
    return {
        "user_request": prompt,
        ("partial_assistant_response" if partial else "assistant_response"): completion,
    }


# Theoretical bounds of the scalarised reward, for whitening / reporting.
REWARD_MIN = -sum(i.weight for i in GENERAL_RUBRIC if i.sign < 0)
REWARD_MAX = sum(i.weight for i in GENERAL_RUBRIC if i.sign > 0)
