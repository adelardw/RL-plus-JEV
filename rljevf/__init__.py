"""RL fine-tuning with Jev as a reward source and as a critic."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# transformers 5.17 materialises weights on a thread pool; on macOS that
# segfaults when a second model is loaded into an already busy process (which
# is exactly what GRPOTrainer does when it builds the reference model).
# Linux/CUDA is unaffected, so only the local dev path pays the slower load.
if sys.platform == "darwin":
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


def _load_dotenv() -> None:
    """Local convenience: pick up a .env next to the project so scripts can be
    run without exporting anything. On Kaggle the key comes from secrets."""
    if os.environ.get("OPEN_ROUTER_API_KEY"):
        return
    p = Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

__all__ = ["config", "jevclient", "rubric", "sg2jev", "registry", "ppo", "data"]
