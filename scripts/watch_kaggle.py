#!/usr/bin/env python
"""Follow a Kaggle session live and emit one line per event worth acting on.

Polling `kernels_status` tells you a run died but not why or when. This
streams the session log instead, so a stuck job, a guard intervention or an
exhausted budget surfaces while the session is still running and the GPU quota
can still be saved.

Emits (one line each):
  JOB     a job started or finished, with its return code
  GUARD   a heartbeat or an intervention (timeout / silence / disk)
  SPEND   Jev / judge spend as the run reports it
  ERROR   a traceback, an OOM, a failed job
  STATUS  session state transitions, and the terminal state, then exits

Designed to be run under a monitor: the output is the event stream.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

INTERESTING = [
    ("JOB",    re.compile(r"^=== (.+?) (?:\((.*?)\)|-> rc=(-?\d+))")),
    ("GUARD",  re.compile(r"^\[guard:[^\]]+\] (.+)")),
    ("ERROR",  re.compile(r"(Traceback \(most recent call last\)|"
                          r"CUDA out of memory|OutOfMemoryError|"
                          r"BudgetExceeded|ModuleNotFoundError|"
                          r"Killed|MemoryError|RuntimeError: .*device)")),
    ("SPEND",  re.compile(r'"cost_usd":\s*([0-9.]+)|spent \$([0-9.]+)')),
    ("PLAN",   re.compile(r"^(plan|jobs|code root|secret|torch):")),
    ("RESULT", re.compile(r"^(calibration:|measured:|STATE:|reward source stats)")),
]

TERMINAL = ("COMPLETE", "ERROR", "CANCEL")


def _api():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(PROJECT))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def classify(line: str) -> tuple[str, str] | None:
    for tag, rx in INTERESTING:
        if rx.search(line):
            return tag, line.rstrip()
    return None


def status_of(api, ref: str, tries: int = 3) -> str:
    """Transient TLS drops from api.kaggle.com are common; a single failure
    must not be reported as a state change."""
    last = "?"
    for i in range(tries):
        try:
            st = api.kernels_status(ref)
            s = getattr(st, "status", None)
            return getattr(s, "name", str(s))
        except Exception as e:  # noqa: BLE001
            last = f"POLL-ERROR:{type(e).__name__}"
            time.sleep(2 * (i + 1))
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=None, help="user/slug; defaults to the runner")
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--max-seconds", type=float, default=11 * 3600)
    ap.add_argument("--all", action="store_true", help="emit every line, not just events")
    args = ap.parse_args()

    api = _api()
    if args.kernel:
        ref = args.kernel
    else:
        user = json.loads((PROJECT / "kaggle.json").read_text())["username"]
        ref = f"{user}/rljevf-runner"

    t0 = time.time()
    print(f"STATUS watching {ref}", flush=True)

    # kernels_logs_stream is a live tail: it blocks and yields chunks as the
    # session produces them, and does not end until the session does. Consume
    # it incrementally rather than collecting it, which is what made an earlier
    # version appear to hang.
    emitted = 0
    # The stream restarts from the beginning after a dropped connection, so
    # track how far we got and re-emit nothing.
    last_t = -1.0
    while time.time() - t0 < args.max_seconds:
        st = status_of(api, ref)
        print(f"STATUS {st}", flush=True)
        terminal = any(k in str(st).upper() for k in TERMINAL)

        try:
            for chunk in api.kernels_logs_stream(ref):
                data = chunk.get("data", "") if isinstance(chunk, dict) else str(chunk)
                when = float(chunk.get("time", 0) if isinstance(chunk, dict) else 0)
                if when <= last_t:
                    continue
                last_t = when
                for ln in str(data).rstrip("\n").splitlines():
                    emitted += 1
                    if args.all:
                        print(f"{when/60:6.1f}m {ln}"[:400], flush=True)
                    else:
                        hit = classify(ln)
                        if hit:
                            print(f"{hit[0]} [{when/60:.0f}m] {hit[1][:360]}", flush=True)
                if time.time() - t0 > args.max_seconds:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"STATUS stream-ended ({type(e).__name__})", flush=True)

        st = status_of(api, ref)
        if any(k in str(st).upper() for k in TERMINAL):
            print(f"STATUS terminal={st} after {(time.time()-t0)/60:.1f} min "
                  f"({emitted} log lines)", flush=True)
            return
        if terminal:
            return
        time.sleep(args.poll)

    print("STATUS watch-timeout", flush=True)


if __name__ == "__main__":
    main()
