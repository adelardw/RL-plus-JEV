#!/usr/bin/env python
"""Download a session's output and prove it arrived intact.

Kaggle's output download is not transactional: a dropped connection leaves a
short file with no error, which would quietly corrupt an analysis three steps
later. This verifies every file against the MANIFEST.json the session wrote,
retries whatever is missing or wrong, and refuses to report success otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _api():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(PROJECT))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def verify(dest: Path, manifest: dict) -> tuple[list[str], list[str]]:
    ok, bad = [], []
    for rec in manifest["files"]:
        f = dest / rec["path"]
        if not f.exists():
            bad.append(f"{rec['path']}: missing")
            continue
        if f.stat().st_size != rec["bytes"]:
            bad.append(f"{rec['path']}: {f.stat().st_size} bytes, expected {rec['bytes']}")
            continue
        if sha256(f) != rec["sha256"]:
            bad.append(f"{rec['path']}: checksum mismatch")
            continue
        ok.append(rec["path"])
    return ok, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--to", default="runs/kaggle")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--no-verify", action="store_true",
                    help="accept the download without a manifest (not recommended)")
    args = ap.parse_args()

    api = _api()
    ref = args.kernel or (
        f"{json.loads((PROJECT / 'kaggle.json').read_text())['username']}/rljevf-runner")
    dest = Path(args.to)
    dest.mkdir(parents=True, exist_ok=True)

    last_bad: list[str] = []
    for attempt in range(1, args.attempts + 1):
        print(f"[fetch] attempt {attempt}/{args.attempts} -> {dest}", flush=True)
        try:
            api.kernels_output(ref, str(dest), force=True, quiet=True)
        except Exception as e:  # noqa: BLE001
            print(f"[fetch] download error: {type(e).__name__}: {e}", flush=True)
            time.sleep(min(60, 5 * attempt))
            continue

        mpath = dest / "artifacts" / "MANIFEST.json"
        if not mpath.exists():
            if args.no_verify:
                print("[fetch] no manifest; accepting as requested", flush=True)
                return
            print("[fetch] no MANIFEST.json in the output -- the session may have "
                  "been killed before packaging", flush=True)
            last_bad = ["MANIFEST.json missing"]
            time.sleep(min(60, 5 * attempt))
            continue

        manifest = json.loads(mpath.read_text())
        ok, bad = verify(dest, manifest)
        print(f"[fetch] verified {len(ok)}/{manifest['n_files']} files "
              f"({manifest['total_bytes']/2**20:.1f} MiB expected)", flush=True)
        if not bad:
            print("[fetch] complete and intact", flush=True)
            return
        last_bad = bad
        for b in bad[:10]:
            print(f"        BAD {b}", flush=True)
        if len(bad) > 10:
            print(f"        ... and {len(bad)-10} more", flush=True)
        time.sleep(min(60, 5 * attempt))

    print(f"[fetch] FAILED after {args.attempts} attempts; "
          f"{len(last_bad)} files still bad", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
