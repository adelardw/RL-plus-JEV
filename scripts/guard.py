#!/usr/bin/env python
"""Supervisor for one job inside a Kaggle session.

A Kaggle session is a hostile place for a long job: it can be pre-empted, it
can run out of the 20GB working disk, a hung generation can silently eat the
whole GPU quota, and when the session does die the job gets a SIGTERM and a
few seconds to behave. The runner kernel's source is fixed (re-pushing it
would drop the OPEN_ROUTER_API_KEY attachment), so every guard has to live
here, in the code dataset, wrapped around the job:

    scripts/guard.py --timeout 3600 --name R4-jev -- scripts/train_grpo.py ...

What it guards against:
  * a hung job          -- hard timeout, and a silence timeout on stdout
  * a dying session     -- SIGTERM/SIGINT forwarded to the child, which gets a
                           grace period to checkpoint before SIGKILL
  * a full disk         -- aborts while there is still room to write results
  * an idle-looking log -- heartbeat lines keep output flowing and record
                           elapsed time, RSS, VRAM and free disk
  * losing the record   -- a status file is updated continuously, so a killed
                           session still leaves behind what happened
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

def _status_dir() -> Path:
    """Where the guard records what it saw. Falls back to a writable place so
    the guard is testable off Kaggle."""
    root = os.environ.get("RLJEVF_RUN_ROOT")
    candidates = []
    if root:
        candidates.append(Path(root).parent / "guard")
    candidates += [Path("/kaggle/working/guard"),
                   Path(__file__).resolve().parent.parent / "runs" / "guard",
                   Path.cwd() / ".guard"]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            return c
        except OSError:
            continue
    return Path.cwd()


def _free_gb(path: str = "/kaggle/working") -> float:
    try:
        st = os.statvfs(path if Path(path).exists() else ".")
        return st.f_bavail * st.f_frsize / 2**30
    except OSError:
        return -1.0


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return -1.0


def _gpu() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        rows = []
        for line in out.splitlines():
            i, used, total, util = [x.strip() for x in line.split(",")]
            rows.append({"gpu": int(i), "mem_used_gb": round(int(used) / 1024, 2),
                         "mem_total_gb": round(int(total) / 1024, 2), "util_pct": int(util)})
        return rows
    except Exception:  # noqa: BLE001
        return []


class Guard:
    # Exit codes the runner can tell apart from an ordinary job failure.
    EXIT_TIMEOUT = 124
    EXIT_SILENCE = 125
    EXIT_DISK = 123
    EXIT_SIGNAL = 126

    def __init__(self, name, cmd, timeout, silence_timeout, heartbeat,
                 min_free_gb, grace):
        self.name = name
        self.cmd = cmd
        self.timeout = timeout
        self.silence_timeout = silence_timeout
        self.heartbeat = heartbeat
        self.min_free_gb = min_free_gb
        self.grace = grace
        self.t0 = time.time()
        self.last_output = time.time()
        self.proc: subprocess.Popen | None = None
        self.reason = "ok"
        self.guard_exit: int | None = None
        self.stop = threading.Event()
        self.status_path = _status_dir() / f"{name}.json"

    # -- bookkeeping --------------------------------------------------------
    def snapshot(self, state: str, returncode=None) -> dict:
        s = {
            "name": self.name,
            "state": state,
            "reason": self.reason,
            "elapsed_s": round(time.time() - self.t0, 1),
            "silent_s": round(time.time() - self.last_output, 1),
            "free_disk_gb": round(_free_gb(), 2),
            "rss_gb": round(_rss_gb(), 2),
            "gpu": _gpu(),
            "returncode": returncode,
            "cmd": " ".join(self.cmd),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, indent=1))
            tmp.replace(self.status_path)
        except OSError:
            pass
        return s

    # -- child lifecycle ----------------------------------------------------
    def _pump(self) -> None:
        """Relay child output and note when it last spoke."""
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.last_output = time.time()
            sys.stdout.write(line)
            sys.stdout.flush()

    def _terminate(self, why: str) -> None:
        """Give the child a chance to checkpoint before killing it."""
        if self.proc is None or self.proc.poll() is not None:
            return
        self.reason = why
        print(f"\n[guard:{self.name}] {why} -- SIGTERM, {self.grace}s to shut down",
              flush=True)
        try:
            self.proc.terminate()
            self.proc.wait(timeout=self.grace)
            print(f"[guard:{self.name}] child exited on SIGTERM", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[guard:{self.name}] grace expired -- SIGKILL", flush=True)
            self.proc.kill()
        except Exception as e:  # noqa: BLE001
            print(f"[guard:{self.name}] terminate error: {e}", flush=True)

    def _watch(self) -> None:
        while not self.stop.wait(self.heartbeat):
            if self.proc is None or self.proc.poll() is not None:
                return
            s = self.snapshot("running")
            gpu = " ".join(f"g{g['gpu']}={g['mem_used_gb']:.1f}/{g['mem_total_gb']:.0f}GB@{g['util_pct']}%"
                           for g in s["gpu"])
            print(f"[guard:{self.name}] t={s['elapsed_s']/60:.1f}m "
                  f"silent={s['silent_s']:.0f}s disk={s['free_disk_gb']:.1f}GB "
                  f"rss={s['rss_gb']:.1f}GB {gpu}", flush=True)

            if self.timeout and s["elapsed_s"] > self.timeout:
                self.guard_exit = self.EXIT_TIMEOUT
                self._terminate(f"timeout after {self.timeout}s")
                return
            if self.silence_timeout and s["silent_s"] > self.silence_timeout:
                self.guard_exit = self.EXIT_SILENCE
                self._terminate(f"no output for {self.silence_timeout}s")
                return
            if 0 <= s["free_disk_gb"] < self.min_free_gb:
                self.guard_exit = self.EXIT_DISK
                self._terminate(f"free disk {s['free_disk_gb']:.1f}GB below {self.min_free_gb}GB")
                return

    def run(self) -> int:
        # A dying session SIGTERMs us; pass it on so the child can checkpoint.
        def on_signal(signum, _frame):
            self.guard_exit = self.EXIT_SIGNAL
            self._terminate(f"received signal {signum}")

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, on_signal)

        print(f"[guard:{self.name}] start: {' '.join(self.cmd)}", flush=True)
        self.snapshot("starting")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", *self.cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        watcher = threading.Thread(target=self._watch, daemon=True)
        watcher.start()
        try:
            self._pump()
        finally:
            rc = self.proc.wait()
            self.stop.set()
            # A job the guard cut short is not a success, even if the child
            # managed to checkpoint and exit 0 on the way out. And a child
            # killed by signal N reports -N, which sys.exit cannot express.
            if self.guard_exit is not None:
                out = self.guard_exit
            elif rc is not None and rc < 0:
                out = 128 + abs(rc)
            else:
                out = rc or 0
            s = self.snapshot("finished", returncode=out)
            print(f"[guard:{self.name}] done rc={out} in {s['elapsed_s']/60:.1f} min "
                  f"({self.reason})", flush=True)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="job")
    ap.add_argument("--timeout", type=float, default=0, help="hard wall-clock limit, seconds")
    ap.add_argument("--silence-timeout", type=float, default=1800,
                    help="abort if the child prints nothing for this long")
    ap.add_argument("--heartbeat", type=float, default=120)
    ap.add_argument("--min-free-gb", type=float, default=1.5)
    ap.add_argument("--grace", type=float, default=90,
                    help="seconds the child gets to checkpoint after SIGTERM")
    # argparse.REMAINDER mis-parses options that appear after the first
    # positional, so split on the literal "--" ourselves.
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: guard.py [options] -- <script> [args...]")
    split = argv.index("--")
    args = ap.parse_args(argv[:split])
    cmd = argv[split + 1:]
    if not cmd:
        raise SystemExit("usage: guard.py [options] -- <script> [args...]")
    sys.exit(Guard(args.name, cmd, args.timeout, args.silence_timeout,
                   args.heartbeat, args.min_free_gb, args.grace).run())


if __name__ == "__main__":
    main()
