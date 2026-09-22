#!/usr/bin/env python
"""Drive the study across sessions.

One Kaggle session cannot hold the whole study, and a session can die at any
point, so progress has to survive the gap between sessions. After each session
this:

  1. waits for the session to reach a terminal state;
  2. downloads its output and verifies it against the manifest;
  3. lifts any resume_state directories into `state/`, which is shipped back
     inside the code dataset, so an interrupted run continues where it stopped
     instead of restarting and re-spending its reward budget;
  4. works out which jobs are finished, writes the next plan with those marked
     done, and publishes it to the dataset.

Pushing the kernel would drop its secret attachment, so the last step is a
click on "Save & Run All" rather than an API call. Everything else is
automatic, including the retry of a failed job in the next batch.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

STATE_DIR = PROJECT / "state"
FETCH_DIR = PROJECT / "runs" / "kaggle"


def sh(*argv: str) -> int:
    print("+", " ".join(argv), flush=True)
    return subprocess.run([sys.executable, *argv], cwd=PROJECT).returncode


def wait_terminal(api, ref: str, poll: float, timeout_s: float) -> str:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            st = api.kernels_status(ref)
            s = getattr(st, "status", None)
            s = getattr(s, "name", str(s))
        except Exception as e:  # noqa: BLE001
            s = f"POLL-ERROR:{type(e).__name__}"
        if s != last:
            print(f"[{(time.time()-t0)/60:6.1f}m] {s}", flush=True)
            last = s
        if any(k in str(s).upper() for k in ("COMPLETE", "ERROR", "CANCEL")):
            return s
        time.sleep(poll)
    return "TIMEOUT"


def lift_resume_states() -> list[str]:
    """Move fetched resume_state directories into state/ for re-upload."""
    STATE_DIR.mkdir(exist_ok=True)
    moved = []
    for d in sorted(FETCH_DIR.rglob("resume_state")):
        if not d.is_dir():
            continue
        run = d.parent.name
        target = STATE_DIR / run / "resume_state"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(d, target)
        moved.append(str(target.relative_to(PROJECT)))
    return moved


def session_outcome() -> dict:
    p = FETCH_DIR / "runner_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def next_plan(plan: dict, outcome: dict, retry_failed: bool) -> dict:
    done = set(plan.get("done", []))
    done |= {r["name"] for r in outcome.get("completed", [])}
    if not retry_failed:
        done |= {r["name"] for r in outcome.get("failed", [])}
    remaining = [j for j in plan["jobs"] if j["name"] not in done]
    nxt = dict(plan)
    nxt["done"] = sorted(done)
    nxt["jobs"] = plan["jobs"]          # the full list; the runner skips `done`
    nxt["_remaining"] = [j["name"] for j in remaining]
    return nxt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--poll", type=float, default=60)
    ap.add_argument("--timeout-hours", type=float, default=11.5)
    ap.add_argument("--no-wait", action="store_true",
                    help="the session already finished; just collect")
    ap.add_argument("--retry-failed", action="store_true", default=True)
    ap.add_argument("--no-retry-failed", dest="retry_failed", action="store_false")
    args = ap.parse_args()

    import kaggle_run as kr

    api = kr._api()
    ref = f"{kr._username()}/{kr.RUNNER_SLUG}"
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())

    if not args.no_wait:
        print(f"waiting on {ref} ...", flush=True)
        st = wait_terminal(api, ref, args.poll, args.timeout_hours * 3600)
        print("terminal status:", st, flush=True)

    rc = sh("scripts/fetch_artifacts.py", "--to", str(FETCH_DIR))
    if rc != 0:
        print("WARNING: artifacts could not be fully verified", flush=True)

    outcome = session_outcome()
    if outcome:
        print("session outcome:", json.dumps(
            {k: outcome.get(k) for k in ("completed", "failed", "skipped", "elapsed_s")},
            indent=1), flush=True)

    moved = lift_resume_states()
    if moved:
        print("resume states preserved for the next session:", flush=True)
        for m in moved:
            print("   ", m, flush=True)

    # pull any results the session produced into the analysis directory
    src = FETCH_DIR / "results"
    if src.is_dir():
        dst = PROJECT / "results"
        dst.mkdir(exist_ok=True)
        for f in src.glob("*.json"):
            shutil.copy2(f, dst / f.name)
            print("   result:", f.name, flush=True)

    nxt = next_plan(plan, outcome, args.retry_failed)
    remaining = nxt.pop("_remaining")
    if not remaining:
        print("\nplan complete -- nothing left to schedule", flush=True)
        return

    out = plan_path.with_name(plan_path.stem + "_next.json")
    out.write_text(json.dumps(nxt, indent=1))
    print(f"\n{len(remaining)} job(s) still to run: {remaining}", flush=True)
    sh("scripts/kaggle_run.py", "set-jobs", "--jobs", str(out))
    print("\nNEXT: click 'Save & Run All' on "
          f"https://www.kaggle.com/code/{kr._username()}/{kr.RUNNER_SLUG}", flush=True)
    print("(a kernel push would drop the secret attachment, so this stays manual)",
          flush=True)


if __name__ == "__main__":
    main()
