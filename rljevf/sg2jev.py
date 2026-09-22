"""StructuredGeneration2JEV -- the component that decides *which questions*
Jev is asked about a completion.

The pipeline sketch left open whether this component is needed at all. We make
it a swappable axis instead of a guess, with three sources:

  static    the fixed hand-written rubric (main pipeline; the only setting in
            which Jev, the RLAIF judge and the self-judge are comparable).
  generated a frozen LLM writes task-specific questions once per *prompt*
            (never per completion -- otherwise the reward is not a function of
            the completion alone and the RL objective is ill-posed). Cached.
  policy    the policy under training proposes its own questions. This is the
            reward-hacking probe: nothing stops it from proposing questions it
            trivially satisfies.

Every generated question is schema-validated and carries an explicit sign, so
scalarisation stays identical to the static case.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Protocol, Sequence

from .orclient import ChatClient, extract_json
from .rubric import GENERAL_RUBRIC, Rubric, RubricItem

MAX_GENERATED_ITEMS = 3

_GEN_SYSTEM = """You are a rubric writer. You never answer the request you are shown.

You will be given a user request that was sent to some AI assistant. Your only job is to
write at most {k} evaluation questions that a grader will later use to score *the
assistant's response* to that request. You must not attempt the request yourself.

Each question:
  key            snake_case identifier
  kind           "noul" (yes/no) or "score" (ordered levels)
  instructions   the question, phrased about the assistant's response
  desirable      true if YES / a higher level means a BETTER response, false if WORSE
  criteria_true  for kind="noul": what makes the answer yes   (else "")
  criteria_false for kind="noul": what makes the answer no    (else "")
  levels         for kind="score": 2-10 ordered level descriptions, worst first (else [])

Rules:
- Questions must be answerable from the request and the response alone.
- Do NOT restate generic criteria (correctness, helpfulness, padding, evasiveness,
  self-promotion) -- those are always asked separately. Add only what is specific to
  this request.
- Prefer questions that separate a genuinely good response from a plausible-looking
  bad one."""


_GEN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rubric_questions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "kind": {"type": "string", "enum": ["noul", "score"]},
                            "instructions": {"type": "string"},
                            "desirable": {"type": "boolean"},
                            "criteria_true": {"type": "string"},
                            "criteria_false": {"type": "string"},
                            "levels": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "key", "kind", "instructions", "desirable",
                            "criteria_true", "criteria_false", "levels",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["questions"],
            "additionalProperties": False,
        },
    },
}

_GEN_USER = (
    "Below, between the markers, is a user request sent to an AI assistant. "
    "Do not answer it. Write the evaluation questions for grading a response to it.\n\n"
    "<<<BEGIN USER REQUEST>>>\n{prompt}\n<<<END USER REQUEST>>>"
)


class RubricSource(Protocol):
    mode: str

    async def arubrics_for(
        self, prompts: Sequence[str], completions: Sequence[str], **kwargs: Any
    ) -> list[Rubric]: ...


class StaticRubricSource:
    mode = "static"

    def __init__(self, rubric: Rubric = GENERAL_RUBRIC):
        self.rubric = rubric

    async def arubrics_for(self, prompts, completions, **kwargs) -> list[Rubric]:
        return [self.rubric] * len(prompts)


def _parse_items(payload: Any) -> tuple[RubricItem, ...]:
    """Validate model-written questions. Anything malformed is dropped rather
    than guessed at -- a silently wrong sign would corrupt the reward."""
    if isinstance(payload, dict):
        payload = payload.get("questions", payload.get("items", []))
    out: list[RubricItem] = []
    seen: set[str] = set()
    for raw in payload if isinstance(payload, list) else []:
        if not isinstance(raw, dict):
            continue
        key, kind = raw.get("key"), raw.get("kind")
        instr, desirable = raw.get("instructions"), raw.get("desirable")
        crit = raw.get("criteria")
        if not (isinstance(key, str) and key and isinstance(instr, str) and instr):
            continue
        if kind not in ("noul", "score") or not isinstance(desirable, bool):
            continue
        key = "gen_" + "".join(c if c.isalnum() else "_" for c in key.lower())[:40]
        if key in seen or key in GENERAL_RUBRIC.keys:
            continue
        if kind == "noul":
            ct, cf = raw.get("criteria_true"), raw.get("criteria_false")
            if isinstance(crit, dict) and {"true", "false"} <= set(crit):
                crit = {"true": str(crit["true"]), "false": str(crit["false"])}
            elif isinstance(ct, str) and ct.strip() and isinstance(cf, str) and cf.strip():
                crit = {"true": ct.strip(), "false": cf.strip()}
            else:
                crit = None
        else:
            levels = crit if isinstance(crit, list) else raw.get("levels")
            if not (isinstance(levels, list) and 2 <= len(levels) <= 10):
                continue
            crit = [str(c) for c in levels]
        seen.add(key)
        out.append(
            RubricItem(
                key=key,
                kind=kind,
                instructions=instr,
                weight=1.0,
                sign=1 if desirable else -1,
                criteria=crit,
            )
        )
        if len(out) >= MAX_GENERATED_ITEMS:
            break
    return tuple(out)


class GeneratedRubricSource:
    """Task-specific questions written once per prompt by a frozen LLM."""

    mode = "generated"

    def __init__(
        self,
        client: ChatClient,
        base: Rubric = GENERAL_RUBRIC,
        max_items: int = MAX_GENERATED_ITEMS,
    ):
        self.client = client
        self.base = base
        self.max_items = max_items
        self._memo: dict[str, Rubric] = {}
        self._lock = asyncio.Lock()

    def _compose(self, extra: tuple[RubricItem, ...]) -> Rubric:
        items = self.base.items + extra
        digest = hashlib.sha1(
            "|".join(f"{i.key}:{i.sign}:{i.instructions}" for i in items).encode()
        ).hexdigest()[:12]
        return Rubric(items=items, name=f"gen-{digest}")

    async def _rubric_for_prompt(self, prompt: str) -> Rubric:
        h = hashlib.sha1(prompt.encode()).hexdigest()
        async with self._lock:
            cached = self._memo.get(h)
        if cached is not None:
            return cached
        msgs = [
            {"role": "system", "content": _GEN_SYSTEM.format(k=self.max_items)},
            {"role": "user", "content": _GEN_USER.format(prompt=prompt)},
        ]
        try:
            raw = (await self.client.acomplete_many([msgs], response_format=_GEN_SCHEMA))[0]
            extra = _parse_items(extract_json(raw))
        except Exception:
            extra = ()  # degrade to the static rubric rather than crash a run
        rubric = self._compose(extra)
        async with self._lock:
            self._memo[h] = rubric
        return rubric

    async def arubrics_for(self, prompts, completions, **kwargs) -> list[Rubric]:
        uniq = list(dict.fromkeys(prompts))
        built = await asyncio.gather(*(self._rubric_for_prompt(p) for p in uniq))
        table = dict(zip(uniq, built))
        return [table[p] for p in prompts]


class PolicyProposedRubricSource(GeneratedRubricSource):
    """Reward-hacking probe: the policy writes the questions it is graded on.

    Identical machinery to `generated`, but the generator is the model being
    trained. Reported as an ablation with the expectation that it degrades.
    """

    mode = "policy"

    def __init__(self, generate_fn, base: Rubric = GENERAL_RUBRIC, max_items: int = MAX_GENERATED_ITEMS):
        self.generate_fn = generate_fn  # (list[str]) -> list[str], the live policy
        self.base = base
        self.max_items = max_items
        self._memo: dict[str, Rubric] = {}
        self._lock = asyncio.Lock()

    async def _rubric_for_prompt(self, prompt: str) -> Rubric:
        h = hashlib.sha1(prompt.encode()).hexdigest()
        async with self._lock:
            cached = self._memo.get(h)
        if cached is not None:
            return cached
        try:
            raw = self.generate_fn(
                [
                    _GEN_SYSTEM.format(k=self.max_items)
                    + "\n\n"
                    + _GEN_USER.format(prompt=prompt)
                    + "\n\nJSON:"
                ]
            )[0]
            extra = _parse_items(extract_json(raw))
        except Exception:
            extra = ()
        rubric = self._compose(extra)
        async with self._lock:
            self._memo[h] = rubric
        return rubric


def build_rubric_source(mode: str, **kw) -> RubricSource:
    if mode == "static":
        return StaticRubricSource(kw.get("rubric", GENERAL_RUBRIC))
    if mode == "generated":
        return GeneratedRubricSource(kw["client"])
    if mode == "policy":
        return PolicyProposedRubricSource(kw["generate_fn"])
    raise ValueError(f"unknown rubric source: {mode}")
