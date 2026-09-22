#!/usr/bin/env python
"""Ship work to Kaggle under two constraints this project actually has:

  * a 6 GPU-hour weekly quota, so the study runs in weekly batches and each
    session must stop cleanly before the quota is exhausted rather than being
    killed mid-write;
  * Kaggle secrets cannot be attached over the API, and pushing a kernel
    version drops the attachment. So the kernel source is pushed **once** and
    never again: the work it does is read from a job list inside the code
    dataset, and updating a dataset does not touch the kernel. The secret stays
    attached, and each week only needs a click on "Save & Run All".

Usage
-----
    # once
    python scripts/kaggle_run.py push-runner
    #   -> then attach OPEN_ROUTER_API_KEY under Add-ons -> Secrets, once

    # each batch
    python scripts/kaggle_run.py set-jobs --jobs jobs/week1.json
    #   -> click "Save & Run All" on the runner kernel
    python scripts/kaggle_run.py status
    python scripts/kaggle_run.py fetch --to runs/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CODE_DIRS = ["rljevf", "scripts", "jobs"]
CODE_FILES = ["pyproject.toml"]
RUNNER_SLUG = "rljevf-runner"
CODE_SLUG = "rljevf-code"


def _api():
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(PROJECT))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _username() -> str:
    return json.loads((PROJECT / "kaggle.json").read_text())["username"]


# --------------------------------------------------------------------------- #
#  The runner kernel. Its source is fixed; the work comes from the dataset.
# --------------------------------------------------------------------------- #
RUNNER_SOURCE = '''# rljevf runner -- pushed once, never edited.
# The work it performs comes from jobs/active.json inside the mounted code
# dataset, so new batches need a dataset update, not a kernel push (a push
# would drop the OPEN_ROUTER_API_KEY secret attachment).
import json, os, pathlib, shutil, subprocess, sys, time

T0 = time.time()
WORK = "/kaggle/working"
os.environ["RLJEVF_RUN_ROOT"] = WORK + "/runs"
os.environ["RLJEVF_CACHE_ROOT"] = WORK + "/cache"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_code():
    for p in sorted(pathlib.Path("/kaggle/input").rglob("rljevf/config.py")):
        return p.parent.parent
    print("contents of /kaggle/input:")
    for q in pathlib.Path("/kaggle/input").rglob("*"):
        print("   ", q)
    raise SystemExit("rljevf package not found in any mounted dataset")

CODE = find_code()
sys.path.insert(0, str(CODE))
print("code root:", CODE, flush=True)

plan_path = CODE / "jobs" / "active.json"
if not plan_path.exists():
    raise SystemExit("jobs/active.json missing from the code dataset")
plan = json.loads(plan_path.read_text())
print("plan:", plan.get("name"), "written", plan.get("written_at"), flush=True)
print("jobs:", [j["name"] for j in plan["jobs"]], flush=True)

os.environ["RLJEVF_BUDGET_USD"] = str(plan.get("budget_usd", 30.0))
BUDGET_S = float(plan.get("session_seconds", 5.0 * 3600))

# secret -------------------------------------------------------------------
key = ""
try:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("OPEN_ROUTER_API_KEY")
    print("secret: loaded (len %d)" % len(key), flush=True)
except Exception as e:
    print("secret: FAILED -", type(e).__name__, e, flush=True)
if not key:
    raise SystemExit(
        "No OPEN_ROUTER_API_KEY. Open this kernel, Add-ons -> Secrets, attach "
        "OPEN_ROUTER_API_KEY, then Save & Run All."
    )
os.environ["OPEN_ROUTER_API_KEY"] = key

# deps ---------------------------------------------------------------------
pips = plan.get("pip", [])
if pips:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pips], check=False)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      "| bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
      flush=True)

# run ----------------------------------------------------------------------
state = {"plan": plan.get("name"), "completed": [], "failed": [], "skipped": []}
state_path = pathlib.Path(WORK) / "runner_state.json"

def save_state():
    state["elapsed_s"] = round(time.time() - T0, 1)
    state_path.write_text(json.dumps(state, indent=1))

for job in plan["jobs"]:
    left = BUDGET_S - (time.time() - T0)
    need = float(job.get("est_seconds", 0))
    if left < max(300.0, need * 0.5):
        print(f"SKIP {job['name']}: {left/60:.1f} min left, needs ~{need/60:.1f} min",
              flush=True)
        state["skipped"].append(job["name"])
        continue
    cmd = [sys.executable, "-u"] + job["command"].split()
    print(f"\\n=== {job['name']} ({left/60:.0f} min of session left) ===",
          flush=True)
    print(">>", " ".join(cmd), flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=str(CODE))
    dt = round(time.time() - t, 1)
    rec = {"name": job["name"], "seconds": dt, "returncode": r.returncode}
    (state["completed"] if r.returncode == 0 else state["failed"]).append(rec)
    print(f"=== {job['name']} -> rc={r.returncode} in {dt/60:.1f} min ===", flush=True)
    save_state()

save_state()

# Keep the output small: intermediate checkpoints are not worth the transfer.
for p in pathlib.Path(WORK + "/runs").rglob("checkpoint-*"):
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)

print("\\nSTATE:", json.dumps(state, indent=1), flush=True)
sys.exit(1 if state["failed"] else 0)
'''


def push_code(api) -> str:
    ref = f"{_username()}/{CODE_SLUG}"
    staging = PROJECT / ".kaggle_staging" / "code"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for d in CODE_DIRS:
        src = PROJECT / d
        if src.exists():
            shutil.copytree(src, staging / d,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in CODE_FILES:
        shutil.copy2(PROJECT / f, staging / f)
    (staging / "dataset-metadata.json").write_text(json.dumps(
        {"title": CODE_SLUG, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    try:
        api.dataset_create_version(str(staging), version_notes=f"code {int(time.time())}",
                                   dir_mode="zip")
        print(f"updated dataset {ref}")
    except Exception as e:
        if any(k in str(e).lower() for k in ("not found", "404", "403", "forbidden")):
            api.dataset_create_new(str(staging), public=False, dir_mode="zip")
            print(f"created dataset {ref}")
        else:
            raise
    return ref


def push_runner(api, gpu: bool) -> str:
    code_ref = push_code(api)
    print("waiting 30s for the dataset version to become available...")
    time.sleep(30)
    ref = f"{_username()}/{RUNNER_SLUG}"
    kdir = PROJECT / ".kaggle_staging" / "runner"
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    (kdir / "main.py").write_text(RUNNER_SOURCE)
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": RUNNER_SLUG, "code_file": "main.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": [code_ref], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }, indent=1))
    api.kernels_push(str(kdir))
    print(f"pushed kernel {ref}")
    print("\nONE-TIME MANUAL STEP:")
    print(f"  open https://www.kaggle.com/code/{_username()}/{RUNNER_SLUG}")
    print("  Add-ons -> Secrets -> attach OPEN_ROUTER_API_KEY")
    print("  then 'Save & Run All'. Do not push this kernel again.")
    return ref


def set_jobs(api, jobs_file: Path) -> None:
    plan = json.loads(jobs_file.read_text())
    plan["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    active = PROJECT / "jobs" / "active.json"
    active.parent.mkdir(exist_ok=True)
    active.write_text(json.dumps(plan, indent=1))
    push_code(api)
    est = sum(j.get("est_seconds", 0) for j in plan["jobs"]) / 3600
    print(f"\nplan '{plan.get('name')}' is live: {len(plan['jobs'])} jobs, "
          f"~{est:.1f} GPU-h estimated")
    print(f"  now click 'Save & Run All' on "
          f"https://www.kaggle.com/code/{_username()}/{RUNNER_SLUG}")


def status(api) -> None:
    ref = f"{_username()}/{RUNNER_SLUG}"
    st = api.kernels_status(ref)
    s = getattr(st, "status", None)
    print(ref, "->", getattr(s, "name", str(s)), getattr(st, "failure_message", "") or "")
    try:
        from kagglesdk.kernels.services.kernels_api_service import (
            ApiGetAcceleratorQuotaStatisticsRequest)
        with api.build_kaggle_client() as k:
            q = k.kernels.kernels_api_client.get_accelerator_quota_statistics(
                ApiGetAcceleratorQuotaStatisticsRequest())
            g = q.gpu_quota
            used, total = _sec(g.time_used), _sec(g.total_time_allowed)
            print(f"GPU quota: {used/3600:.2f}h used of {total/3600:.2f}h, "
                  f"resets {q.quota_refresh_time}")
    except Exception as e:  # noqa: BLE001
        print("quota unavailable:", type(e).__name__, e)


def _sec(v) -> float:
    """Protobuf Durations arrive as timedelta, as '3600s', or as an object with
    .seconds depending on the SDK path. Accept all three."""
    if v is None:
        return 0.0
    for attr in ("total_seconds", "seconds"):
        got = getattr(v, attr, None)
        if callable(got):
            try:
                return float(got())
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(got, (int, float)):
            return float(got)
    try:
        return float(str(v).rstrip("s"))
    except ValueError:
        return 0.0


def logs(api) -> None:
    import tempfile
    ref = f"{_username()}/{RUNNER_SLUG}"
    d = tempfile.mkdtemp()
    api.kernels_output(ref, d)
    for f in Path(d).rglob("*.log"):
        for e in json.loads(f.read_text()):
            print(e["data"], end="")


def fetch(api, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    api.kernels_output(f"{_username()}/{RUNNER_SLUG}", str(dest))
    print(f"downloaded -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push-runner"); p.add_argument("--no-gpu", action="store_true")
    p = sub.add_parser("set-jobs"); p.add_argument("--jobs", required=True)
    sub.add_parser("status")
    sub.add_parser("logs")
    p = sub.add_parser("fetch"); p.add_argument("--to", default="runs/kaggle")
    sub.add_parser("push-code")
    args = ap.parse_args()

    api = _api()
    if args.cmd == "push-runner":
        push_runner(api, gpu=not args.no_gpu)
    elif args.cmd == "set-jobs":
        set_jobs(api, Path(args.jobs))
    elif args.cmd == "push-code":
        push_code(api)
    elif args.cmd == "status":
        status(api)
    elif args.cmd == "logs":
        logs(api)
    elif args.cmd == "fetch":
        fetch(api, Path(args.to))


if __name__ == "__main__":
    main()
