"""The cross-reward matrix (plan 8.3).

Every final checkpoint is scored by *every* reward source. The diagonal is what
each run was trained on; the off-diagonal is what everyone else thinks of it.
A large diagonal-minus-off-diagonal gap is over-optimisation: the policy moved
towards its own judge rather than towards quality any judge would recognise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def build_matrix(
    completions_by_run: Mapping[str, Sequence[str]],
    prompts: Sequence[str],
    sources: Mapping[str, Callable[..., Sequence[float]]],
) -> dict[str, dict[str, float]]:
    from ..ppo import _maybe_sync

    matrix: dict[str, dict[str, float]] = {}
    for run, comps in completions_by_run.items():
        row: dict[str, float] = {}
        for sname, src in sources.items():
            scores = _maybe_sync(src(prompts=list(prompts), completions=list(comps)))
            row[sname] = sum(scores) / max(1, len(scores))
        matrix[run] = row
    return matrix


def normalise(matrix: dict[str, dict[str, float]], baseline_run: str) -> dict[str, dict[str, float]]:
    """Express every cell as a delta from the SFT baseline row, so that reward
    sources with different natural scales (a BERT logit vs a rubric sum) can be
    read in one table."""
    base = matrix[baseline_run]
    return {
        run: {s: v - base[s] for s, v in row.items()}
        for run, row in matrix.items()
    }


def overoptimisation_gap(
    matrix: dict[str, dict[str, float]], trained_on: Mapping[str, str], baseline_run: str
) -> dict[str, float]:
    """diag - mean(off-diag), in baseline-relative units."""
    rel = normalise(matrix, baseline_run)
    out: dict[str, float] = {}
    for run, row in rel.items():
        own = trained_on.get(run)
        if own is None or own not in row:
            continue
        others = [v for s, v in row.items() if s != own]
        out[run] = row[own] - (sum(others) / len(others) if others else 0.0)
    return out


def save(matrix: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(matrix, indent=1))
