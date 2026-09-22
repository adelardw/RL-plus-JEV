#!/usr/bin/env python
"""Consolidate a session's output into one verifiable bundle.

Run as the last job of every plan. Kaggle commits /kaggle/working when the
kernel exits cleanly and discards it when the session is killed, so the point
of stopping at 10.5h is to reach this step. It:

  * keeps the newest resume checkpoint per run and drops older ones, so a
    killed run can continue next session without shipping every checkpoint;
  * renames anything matching `checkpoint-*` out of the way, because the
    runner deletes that glob to keep the output small;
  * writes MANIFEST.json with a sha256 and a size per file, so the download
    side can prove it got everything intact rather than hoping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

WORK = Path(os.environ.get("RLJEVF_WORK", "/kaggle/working"))


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def newest_checkpoint(run_dir: Path) -> Path | None:
    """TRL writes checkpoint-<step>; keep the highest step only."""
    cks = [d for d in run_dir.glob("checkpoint-*") if d.is_dir()]
    if not cks:
        return None

    def step(d: Path) -> int:
        try:
            return int(d.name.split("-")[-1])
        except ValueError:
            return -1

    return max(cks, key=step)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--keep-weights", action="store_true", default=True,
                    help="keep the newest resume checkpoint per run")
    ap.add_argument("--no-weights", dest="keep_weights", action="store_false")
    ap.add_argument("--max-gb", type=float, default=17.0,
                    help="drop weights if the bundle would exceed this")
    args = ap.parse_args()

    work = Path(args.work)
    runs = work / "runs"
    out = work / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], []
    if runs.exists():
        for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
            newest = newest_checkpoint(run_dir)
            for d in run_dir.glob("checkpoint-*"):
                if not d.is_dir():
                    continue
                if args.keep_weights and newest is not None and d == newest:
                    # move out of the runner's delete glob
                    target = run_dir / "resume_state"
                    if target.exists():
                        shutil.rmtree(target, ignore_errors=True)
                    shutil.move(str(d), str(target))
                    kept.append(str(target.relative_to(work)))
                else:
                    shutil.rmtree(d, ignore_errors=True)
                    dropped.append(str(d.relative_to(work)))

    # size check: shed weights rather than produce a bundle Kaggle will refuse
    def total_gb(root: Path) -> float:
        skip = {"artifacts", ".venv", ".git", "__pycache__", ".kaggle_staging"}
        return sum(f.stat().st_size for f in root.rglob("*")
                   if f.is_file() and not skip & set(f.parts)) / 2**30

    if runs.exists() and total_gb(work) > args.max_gb:
        print(f"bundle {total_gb(work):.1f}GB over {args.max_gb}GB -- dropping weights",
              flush=True)
        for d in runs.rglob("resume_state"):
            shutil.rmtree(d, ignore_errors=True)
            dropped.append(str(d.relative_to(work)))

    SKIP_DIRS = {"artifacts", ".venv", ".git", "__pycache__", "node_modules",
                 ".kaggle_staging", ".ipynb_checkpoints"}
    SKIP_SUFFIX = {".lock", ".pyc", ".pyo", ".tmp", ".shm", ".wal"}

    files = []
    for f in sorted(work.rglob("*")):
        if not f.is_file() or SKIP_DIRS & set(f.parts):
            continue
        if f.suffix in SKIP_SUFFIX or f.name.endswith(".tmp"):
            continue
        files.append({
            "path": str(f.relative_to(work)),
            "bytes": f.stat().st_size,
            "sha256": sha256(f),
        })

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "kept_checkpoints": kept,
        "dropped_checkpoints": dropped,
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(files)} files, "
          f"{manifest['total_bytes']/2**30:.2f} GB, "
          f"{len(kept)} resume checkpoints kept, {len(dropped)} dropped", flush=True)
    for f in files[:40]:
        print(f"  {f['bytes']:>12,}  {f['path']}", flush=True)
    if len(files) > 40:
        print(f"  ... and {len(files)-40} more", flush=True)


if __name__ == "__main__":
    main()
