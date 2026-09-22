"""Where experiments are allowed to run.

The laptop is for smoke tests: a few steps of a tiny model to prove the code
path works. Experiments run on Kaggle. Mixing the two produces results that
cannot be compared -- MPS and CUDA differ in throughput, in supported dtypes
and in kernel behaviour -- and it quietly occupies a machine someone is using.

This is enforced rather than remembered: an experiment script refuses to start
without CUDA unless it is explicitly told it is a smoke test.
"""
from __future__ import annotations

import os
import sys


class WrongMachine(SystemExit):
    pass


def require_experiment_host(name: str, allow_smoke: bool = True) -> str:
    """Call at the top of any script that produces a result.

    Returns the device. Raises unless one of:
      * CUDA is present (Kaggle, or any real GPU host);
      * `--smoke` was passed, or RLJEVF_SMOKE=1 is set.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"

    smoke = "--smoke" in sys.argv or os.environ.get("RLJEVF_SMOKE") == "1"
    if smoke and allow_smoke:
        print(f"[{name}] SMOKE MODE on "
              f"{'mps' if torch.backends.mps.is_available() else 'cpu'}: "
              f"results are for checking the code path, not for the paper.",
              flush=True)
        return "mps" if torch.backends.mps.is_available() else "cpu"

    raise WrongMachine(
        f"\n[{name}] refusing to run: no CUDA device.\n"
        f"  Experiments run on Kaggle, not on the laptop -- results from MPS "
        f"and CUDA are not comparable,\n"
        f"  and a long local job occupies a machine someone is using.\n"
        f"    to run it properly:  python scripts/kaggle_run.py set-jobs --jobs jobs/<plan>.json\n"
        f"                         python scripts/kaggle_run.py run\n"
        f"    to check the code path here instead, pass --smoke\n"
    )


def is_smoke() -> bool:
    return "--smoke" in sys.argv or os.environ.get("RLJEVF_SMOKE") == "1"


def pin_single_gpu(reason: str = "training") -> None:
    """Restrict training to one GPU.

    With two devices visible, the HuggingFace Trainer wraps the model in
    `nn.DataParallel`, which scatters the batch across both and then fails --
    "index is on cuda:1, different from other tensors on cuda:0" -- for models
    that were never set up for it. A 0.5B policy with LoRA does not need a
    second card, and leaving the second free is what lets a local judge sit
    beside the run rather than competing with it.

    Must be called before CUDA initialises, which is why it edits the
    environment rather than calling into torch.
    """
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[{reason}] pinned to one GPU: the Trainer would otherwise use "
          f"DataParallel across every visible device", flush=True)
