"""Finding the state a previous session left behind.

A run may be interrupted by the 12-hour session wall, by the guard, or by a
pre-emption. The resume state then lives in one of two places: the current
session's working directory (the run was interrupted and restarted in the same
session), or a dataset mounted from a previous session's output. Both are
searched, newest first, so a restarted study continues rather than repeating.

The directory is deliberately called `resume_state` and never `checkpoint-*`:
the Kaggle runner deletes the latter glob to keep session output small.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RESUME_DIRNAME = "resume_state"


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def find_resume(run_id: str, out_dir: Path, extra_roots=()) -> Path | None:
    """Newest usable resume directory for this run, or None."""
    candidates: list[Path] = []

    local = Path(out_dir) / RESUME_DIRNAME
    if local.is_dir():
        candidates.append(local)

    roots = [Path("/kaggle/input")] + [Path(r) for r in extra_roots]
    env_root = os.environ.get("RLJEVF_RESUME_ROOT")
    if env_root:
        roots.insert(0, Path(env_root))
    for root in roots:
        if not root.exists():
            continue
        try:
            for d in root.rglob(RESUME_DIRNAME):
                if d.is_dir() and run_id in str(d):
                    candidates.append(d)
        except OSError:
            continue

    usable = [c for c in candidates if _is_usable(c)]
    if not usable:
        return None
    return max(usable, key=_mtime)


def _is_usable(d: Path) -> bool:
    """A half-written checkpoint is worse than none."""
    if (d / "trainer_state.pt").exists() and (d / "policy").is_dir():
        return True                                   # our PPO trainer
    if (d / "trainer_state.json").exists():           # HF Trainer / TRL
        return True
    return False


def resume_step(d: Path) -> int | None:
    try:
        if (d / "trainer_state.json").exists():
            return int(json.loads((d / "trainer_state.json").read_text())["global_step"])
        if (d / "history.json").exists():
            return len(json.loads((d / "history.json").read_text()))
    except Exception:  # noqa: BLE001
        pass
    return None


def describe(d: Path | None) -> str:
    if d is None:
        return "no resume state found; starting from the SFT checkpoint"
    st = resume_step(d)
    return f"resuming from {d}" + (f" at step {st}" if st is not None else "")
