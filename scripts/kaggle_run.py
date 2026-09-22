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

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CODE_DIRS = ["rljevf", "scripts", "jobs", "state"]
CODE_FILES = ["pyproject.toml"]
RUNNER_SLUG = "rljevf-runner"
CODE_SLUG = "rljevf-code"
SECRETS_SLUG = "rljevf-secrets"
SECRETS_FILE = "rljevf_secrets.json"


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
# Raw: the kernel source contains \n escapes that must reach the generated
# file intact rather than being interpreted when this module is imported.
RUNNER_SOURCE = r'''# rljevf runner -- pushed ONCE and never again.
#
# Pushing a kernel version drops its Kaggle secret attachment, so this source
# must not change. Everything variable lives in jobs/active.json inside the
# mounted code dataset, and a dataset update does not touch the kernel.
#
# What it handles:
#   * the 12h session wall -- stops at the plan's session_seconds (10.5h) and
#     always reserves time to package artifacts, because Kaggle only commits
#     /kaggle/working when the kernel exits cleanly;
#   * resume -- jobs already completed in a previous session are skipped, and
#     training scripts find their own resume_state under /kaggle/input;
#   * a missing secret -- jobs that do not need the API still run;
#   * connectivity -- checked with retries before anything expensive starts.
import json, os, pathlib, shutil, subprocess, sys, time, urllib.request

T0 = time.time()
WORK = "/kaggle/working"
os.environ["RLJEVF_RUN_ROOT"] = WORK + "/runs"
os.environ["RLJEVF_CACHE_ROOT"] = WORK + "/cache"
os.environ["RLJEVF_WORK"] = WORK
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def elapsed():
    return time.time() - T0

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
# Kaggle kills the session at 12h; stop well before and keep time to package.
BUDGET_S = float(plan.get("session_seconds", 37800))
RESERVE_S = float(plan.get("reserve_seconds", 1200))
ALREADY_DONE = set(plan.get("done", []))
if ALREADY_DONE:
    print("already done in earlier sessions:", sorted(ALREADY_DONE), flush=True)

# -- connectivity ----------------------------------------------------------
def reachable(url, tries=5):
    for i in range(tries):
        try:
            urllib.request.urlopen(url, timeout=25)
            return True
        except Exception as e:
            print(f"  {url}: attempt {i+1}/{tries} failed ({type(e).__name__})", flush=True)
            time.sleep(min(30, 3 * (i + 1)))
    return False

net = {u: reachable(u) for u in ("https://openrouter.ai/api/v1/models",
                                 "https://huggingface.co/api/models?limit=1")}
print("connectivity:", net, flush=True)
if not all(net.values()):
    print("WARNING: some endpoints unreachable; jobs needing them will fail", flush=True)

# -- credentials -----------------------------------------------------------
# Two sources, in order of preference. A notebook secret is the better place
# for a key, but it cannot be attached over the API and a kernel push drops
# it, so unattended batches fall back to a private mounted dataset.
key = ""
source = None
try:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("OPEN_ROUTER_API_KEY")
    source = "kaggle secret"
except Exception as e:
    print("kaggle secret unavailable:", type(e).__name__, flush=True)

if not key:
    for cand in sorted(pathlib.Path("/kaggle/input").rglob("rljevf_secrets.json")):
        try:
            blob = json.loads(cand.read_text())
            key = blob.get("OPEN_ROUTER_API_KEY", "")
            if key:
                source = f"mounted dataset {cand.parent.name}"
                break
        except Exception as e:
            print("could not read", cand, type(e).__name__, flush=True)

if key:
    os.environ["OPEN_ROUTER_API_KEY"] = key
    print(f"credentials: loaded from {source} (len {len(key)})", flush=True)
else:
    print("NOTE: no OPEN_ROUTER_API_KEY from either source. "
          "Jobs marked needs_api will be skipped.", flush=True)

# -- deps ------------------------------------------------------------------
pips = plan.get("pip", [])
if pips:
    for attempt in range(3):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pips])
        if r.returncode == 0:
            break
        print(f"pip install failed (attempt {attempt+1}/3), retrying", flush=True)
        time.sleep(15)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      "| gpus", torch.cuda.device_count(), flush=True)

# -- run -------------------------------------------------------------------
state = {"plan": plan.get("name"), "session_budget_s": BUDGET_S,
         "completed": [], "failed": [], "skipped": [], "connectivity": net,
         "had_secret": bool(key)}
state_path = pathlib.Path(WORK) / "runner_state.json"

def save_state():
    state["elapsed_s"] = round(elapsed(), 1)
    state_path.write_text(json.dumps(state, indent=1))

save_state()

for job in plan["jobs"]:
    name = job["name"]
    if name in ALREADY_DONE:
        print(f"SKIP {name}: completed in an earlier session", flush=True)
        state["skipped"].append({"name": name, "why": "already done"})
        continue
    if job.get("needs_api", True) and not key:
        print(f"SKIP {name}: needs the API and no secret is attached", flush=True)
        state["skipped"].append({"name": name, "why": "no secret"})
        continue
    left = BUDGET_S - elapsed() - RESERVE_S
    need = float(job.get("est_seconds", 0))
    if left < max(300.0, need * 0.4):
        print(f"SKIP {name}: {left/60:.1f} min usable, needs ~{need/60:.1f} min",
              flush=True)
        state["skipped"].append({"name": name, "why": "out of session time"})
        continue

    print(f"\n=== {name} ({left/60:.0f} min usable of session) ===", flush=True)
    cmd = [sys.executable, "-u"] + job["command"].split()
    print(">>", " ".join(cmd), flush=True)
    t = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(CODE)).returncode
    except Exception as e:
        print("launch error:", type(e).__name__, e, flush=True)
        rc = 1
    dt = round(time.time() - t, 1)
    rec = {"name": name, "seconds": dt, "returncode": rc}
    (state["completed"] if rc == 0 else state["failed"]).append(rec)
    print(f"=== {name} -> rc={rc} in {dt/60:.1f} min ===", flush=True)
    save_state()

# -- package ---------------------------------------------------------------
# Always, even after failures: this is what makes the session downloadable.
print(f"\n=== packaging artifacts ({elapsed()/60:.0f} min elapsed) ===", flush=True)
try:
    subprocess.run([sys.executable, "-u", "scripts/package_artifacts.py"],
                   cwd=str(CODE), timeout=RESERVE_S)
except Exception as e:
    print("packaging error:", type(e).__name__, e, flush=True)

save_state()
print("\nSTATE:", json.dumps(state, indent=1), flush=True)
sys.exit(1 if state["failed"] else 0)
'''


def retrying(fn, *a, tries: int = 5, base: float = 3.0, what: str = "kaggle call", **kw):
    """Kaggle's API drops TLS connections often enough that an un-retried call
    is a liability in a long orchestration loop."""
    last = None
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
            if i == tries - 1:
                break
            wait = min(45.0, base * (2 ** i))
            print(f"  {what}: {type(e).__name__}, retry {i+1}/{tries-1} in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise last


def preflight() -> None:
    """Refuse to ship code that cannot even be compiled.

    A syntax error only shows up minutes into a Kaggle session, after the
    image has booted and pip has run, and costs a whole manual re-run. Cheap
    to check here; expensive to discover there.
    """
    bad = []
    for d in CODE_DIRS:
        root = PROJECT / d
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.py")):
            try:
                compile(f.read_text(), str(f), "exec")
            except SyntaxError as e:
                bad.append(f"{f.relative_to(PROJECT)}:{e.lineno}: {e.msg}")
    # the job plan must be loadable, name commands that exist, and write
    # somewhere writable -- the code dataset is mounted read-only in the kernel,
    # which cost a job when collect_results.py tried to write beside its source
    active = PROJECT / "jobs" / "active.json"
    if active.exists():
        try:
            plan = json.loads(active.read_text())
            for job in plan.get("jobs", []):
                parts = job["command"].split()
                scripts_named = [a for a in parts if a.endswith(".py")]
                for script in scripts_named:
                    if not (PROJECT / script).exists():
                        bad.append(f"jobs/active.json: {job['name']} -> missing {script}")
                if "--out" in parts:
                    out = parts[parts.index("--out") + 1]
                    if not out.startswith("/kaggle/working"):
                        bad.append(f"jobs/active.json: {job['name']} writes to "
                                   f"{out!r}, which is not under /kaggle/working "
                                   f"(the code dataset is read-only)")
        except Exception as e:  # noqa: BLE001
            bad.append(f"jobs/active.json: {type(e).__name__}: {e}")
    if bad:
        for b in bad:
            print("PREFLIGHT FAIL:", b)
        raise SystemExit(f"refusing to upload: {len(bad)} problem(s)")
    print("preflight: all shipped python compiles, job commands resolve")


def push_secrets(api) -> str:
    """Publish the API key as a PRIVATE Kaggle dataset the kernel can mount.

    Kaggle cannot attach a notebook secret over the API, and pushing a kernel
    version drops any attachment made in the UI, which makes unattended runs
    impossible. Mounting the key as a private dataset removes that dependency
    entirely: the kernel source can then be pushed freely, which is what lets
    a batch be started without a human clicking anything.

    The trade-off is deliberate and worth stating: the key now lives in Kaggle
    storage rather than in Kaggle's secret store. It is written to a dataset
    created with `public=False`, and this function verifies that the dataset
    really is private before returning. The key is never printed.
    """
    import rljevf  # noqa: F401  -- importing the package loads a local .env

    key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPEN_ROUTER_API_KEY not in the environment; nothing to upload")

    ref = f"{_username()}/{SECRETS_SLUG}"
    staging = PROJECT / ".kaggle_staging" / "secrets"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / SECRETS_FILE).write_text(json.dumps({"OPEN_ROUTER_API_KEY": key}))
    (staging / "dataset-metadata.json").write_text(json.dumps(
        {"title": SECRETS_SLUG, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, indent=1))

    try:
        api.dataset_create_version(str(staging), version_notes=f"key {int(time.time())}")
        print(f"updated private dataset {ref} (key length {len(key)})")
    except Exception as e:
        if any(k in str(e).lower() for k in ("not found", "404", "403", "forbidden")):
            api.dataset_create_new(str(staging), public=False)
            print(f"created private dataset {ref} (key length {len(key)})")
        else:
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)   # no key left on disk

    # Never take privacy on trust for a file holding a credential.
    # This API build has no dataset_view, so confirm two ways: the owned
    # listing must report it private, and a public search must not find it.
    for _ in range(6):
        time.sleep(5)
        try:
            owned = api.dataset_list(mine=True, search=SECRETS_SLUG)
            match = [d for d in owned if str(d.ref).endswith(SECRETS_SLUG)]
            if not match:
                continue
            private = getattr(match[0], "isPrivate", getattr(match[0], "is_private", None))
            public_hits = [d for d in api.dataset_list(search=SECRETS_SLUG)
                           if str(d.ref).endswith(SECRETS_SLUG)]
            if private is True and not public_hits:
                print(f"verified: {ref} is private and not publicly listed")
                return ref
            if private is False or public_hits:
                raise SystemExit(
                    f"REFUSING TO CONTINUE: {ref} appears PUBLIC. Delete it now at "
                    f"https://www.kaggle.com/datasets/{ref}/settings"
                )
        except SystemExit:
            raise
        except Exception:
            continue
    print(f"WARNING: could not confirm {ref} is private -- check "
          f"https://www.kaggle.com/datasets/{ref}/settings")
    return ref


def push_code(api) -> str:
    preflight()
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


def push_runner(api, gpu: bool, quiet: bool = False) -> str:
    code_ref = f"{_username()}/{CODE_SLUG}"
    if not quiet:
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
        "dataset_sources": [code_ref, f"{_username()}/{SECRETS_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }, indent=1))
    retrying(api.kernels_push, str(kdir), what="kernels_push")
    print(f"pushed kernel {ref} -- run started")
    if quiet:
        return ref
    print("\nIf you prefer the key in Kaggle's secret store instead of a "
          "mounted dataset:")
    print(f"  open https://www.kaggle.com/code/{_username()}/{RUNNER_SLUG}")
    print("  Add-ons -> Secrets -> attach OPEN_ROUTER_API_KEY")
    print("  then 'Save & Run All'. Do not push this kernel again.")
    return ref


def trigger(api, gpu: bool = True) -> str:
    """Start a batch. Re-pushing the kernel is what starts a run on Kaggle;
    with the key mounted as a dataset there is no secret attachment to lose,
    so this needs nobody at a keyboard."""
    push_code(api)
    print("waiting 25s for the dataset version to land...")
    time.sleep(25)
    return push_runner(api, gpu=gpu, quiet=True)


def wait_for(api, poll: float = 60, timeout_s: float = 12 * 3600) -> str:
    ref = f"{_username()}/{RUNNER_SLUG}"
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            st = retrying(api.kernels_status, ref, tries=4, what="kernels_status")
            s = getattr(st, "status", None)
            s = getattr(s, "name", str(s))
        except Exception as e:  # noqa: BLE001
            # a transient outage must not end the watch
            print(f"  status unavailable ({type(e).__name__}); continuing", flush=True)
            time.sleep(poll)
            continue
        if s != last:
            print(f"[{(time.time()-t0)/60:6.1f}m] {s}", flush=True)
            last = s
        if any(k in str(s).upper() for k in ("COMPLETE", "ERROR", "CANCEL")):
            return s
        time.sleep(poll)
    return "TIMEOUT"


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
    st = retrying(api.kernels_status, ref, what="kernels_status")
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
    retrying(api.kernels_output, ref, d, what="kernels_output")
    for f in Path(d).rglob("*.log"):
        for e in json.loads(f.read_text()):
            print(e["data"], end="")


def fetch(api, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    retrying(api.kernels_output, f"{_username()}/{RUNNER_SLUG}", str(dest),
             what="kernels_output")
    print(f"downloaded -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push-runner"); p.add_argument("--no-gpu", action="store_true")
    sub.add_parser("push-secrets")
    p = sub.add_parser("run")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--wait", action="store_true")
    p = sub.add_parser("set-jobs"); p.add_argument("--jobs", required=True)
    sub.add_parser("status")
    sub.add_parser("logs")
    p = sub.add_parser("fetch"); p.add_argument("--to", default="runs/kaggle")
    sub.add_parser("push-code")
    args = ap.parse_args()

    api = _api()
    if args.cmd == "push-runner":
        push_secrets(api)
        push_runner(api, gpu=not args.no_gpu)
    elif args.cmd == "push-secrets":
        push_secrets(api)
    elif args.cmd == "run":
        trigger(api, gpu=not args.no_gpu)
        if args.wait:
            print("final status:", wait_for(api))
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
