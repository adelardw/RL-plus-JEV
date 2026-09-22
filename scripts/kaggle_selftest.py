#!/usr/bin/env python
"""Cheap pre-flight for a Kaggle session: proves the environment can do
everything a real run needs before we spend GPU quota on it."""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from __future__ import annotations

import json
import os
import sys
import time

checks: dict[str, object] = {}


def check(name):
    def deco(fn):
        t = time.perf_counter()
        try:
            checks[name] = {"ok": True, "info": fn(), "s": round(time.perf_counter() - t, 2)}
        except Exception as e:  # noqa: BLE001
            checks[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "s": round(time.perf_counter() - t, 2)}
        print(f"{'PASS' if checks[name]['ok'] else 'FAIL'}  {name}: "
              f"{checks[name].get('info', checks[name].get('error'))}", flush=True)
        return fn
    return deco


@check("torch")
def _():
    import torch
    return {
        "version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "vram_gb": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)
                    for i in range(torch.cuda.device_count())],
    }


@check("libs")
def _():
    import datasets, transformers, trl
    return {"transformers": transformers.__version__, "trl": trl.__version__, "datasets": datasets.__version__}


@check("api_key")
def _():
    k = os.environ.get("OPEN_ROUTER_API_KEY", "")
    if not k:
        raise RuntimeError("OPEN_ROUTER_API_KEY missing")
    return {"present": True, "len": len(k)}


@check("jev_call")
def _():
    from rljevf.jevclient import JevClient
    from rljevf.rubric import GENERAL_RUBRIC, build_state
    c = JevClient(use_cache=False)
    r = c.ask(build_state("What is 2+2?", "4"), GENERAL_RUBRIC.jev_questions())
    return {"reward": round(GENERAL_RUBRIC.scalarise_jev(r["answers"]), 3),
            "cost": r["usage"]["cost"], "p50_s": c.meter.summary()["latency_p50_s"]}


@check("jev_throughput")
def _():
    from rljevf.jevclient import JevClient
    from rljevf.rubric import GENERAL_RUBRIC, build_state
    c = JevClient(use_cache=False, concurrency=48)
    states = [build_state(f"Question {i}: what is {i}+{i}?", f"The answer is {2*i}.") for i in range(64)]
    t = time.perf_counter()
    c.ask_many(states, GENERAL_RUBRIC.jev_questions())
    wall = time.perf_counter() - t
    s = c.meter.summary()
    return {"n": 64, "wall_s": round(wall, 2), "calls_per_s": round(64 / wall, 1),
            "p50_s": s["latency_p50_s"], "p95_s": s["latency_p95_s"], "cost": s["cost_usd"]}


@check("datasets")
def _():
    from rljevf.data import eval_prompts, fingerprint, gsm8k, rl_prompts
    a, b, g = rl_prompts(64), eval_prompts(16), gsm8k(8)
    return {"rl": fingerprint(a), "eval": fingerprint(b), "gsm8k": len(g)}


@check("policy_load_and_generate")
def _():
    import torch
    from rljevf.config import POLICY_MODEL
    from rljevf.evaluate.generate import load_policy, sample
    m, tok, dev = load_policy(POLICY_MODEL)
    out = sample(m, tok, ["Say hello in one word."], dev, max_new_tokens=8, greedy=True)
    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    return {"device": dev, "sample": out[0][:40], "peak_vram_gb": round(mem, 2)}


print(json.dumps(checks, indent=1))
sys.exit(0 if all(c["ok"] for c in checks.values()) else 1)
