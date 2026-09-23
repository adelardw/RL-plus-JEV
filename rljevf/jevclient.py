"""Async client for Jev (TypeSafe System One) via OpenRouter's /alpha/decisions.

Responsibilities beyond "send HTTP":
  * a hard spend cap, so a bug cannot burn the study budget;
  * an on-disk cache, so re-running eval or resuming a pre-empted Kaggle
    session costs nothing and reproduces bit-for-bit;
  * per-call latency and cost accounting, which is itself a result (paper 8.4).
"""
from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Sequence

import httpx

from .config import (
    BUDGET_USD,
    CACHE_ROOT,
    JEV_ENDPOINT,
    JEV_MAX_CONCURRENCY,
    JEV_MODEL,
)


class BudgetExceeded(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        try:  # Kaggle
            from kaggle_secrets import UserSecretsClient  # type: ignore

            key = UserSecretsClient().get_secret("OPEN_ROUTER_API_KEY")
        except Exception:
            pass
    if not key:
        raise RuntimeError(
            "OPEN_ROUTER_API_KEY not found in env or Kaggle secrets."
        )
    return key


# --------------------------------------------------------------------------- #
#  Accounting
# --------------------------------------------------------------------------- #
@dataclass
class Meter:
    """Thread-safe spend/latency accounting, persisted so the cap survives
    a process restart (Kaggle sessions get pre-empted)."""

    name: str = "jev"
    path: Path | None = None
    budget_usd: float = BUDGET_USD
    cost_usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_flushed: float = 0.0

    flush_every: int = 20

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = CACHE_ROOT / f"spend_{self.name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The cap is only as good as the record of what has been spent, so make
        # sure a process that dies mid-run still leaves its spend on disk.
        atexit.register(self._flush_quiet)
        if self.path.exists():
            try:
                prior = json.loads(self.path.read_text())
                self.cost_usd = float(prior.get("cost_usd", 0.0))
                self._last_flushed = self.cost_usd
                self.calls = int(prior.get("calls", 0))
                self.input_tokens = int(prior.get("input_tokens", 0))
            except Exception:
                pass

    # The cap is on the *study*, not on one client. Each Meter keeps its own
    # file, so checking only its own total made the effective ceiling
    # budget x (number of meters) -- fourteen of them, and a $30 budget that
    # was really $420. The global total is the sum of every meter's file, which
    # stays correct without a shared ledger to race over.
    # ClassVar, not a field: these are shared cache state read through `cls`,
    # and a dataclass would otherwise give every instance its own stale copy in
    # __init__, __repr__ and __eq__.
    _global_total: ClassVar[float] = 0.0
    _global_checked_at: ClassVar[float] = 0.0
    _global_lock: ClassVar[threading.Lock] = threading.Lock()
    GLOBAL_REFRESH_S: ClassVar[float] = 10.0

    @classmethod
    def global_spend(cls, force: bool = False) -> float:
        now = time.time()
        with cls._global_lock:
            if not force and now - cls._global_checked_at < cls.GLOBAL_REFRESH_S:
                return cls._global_total
            total = 0.0
            try:
                for f in CACHE_ROOT.glob("spend_*.json"):
                    try:
                        total += float(json.loads(f.read_text()).get("cost_usd", 0.0))
                    except (OSError, ValueError, json.JSONDecodeError):
                        continue
            except OSError:
                pass
            cls._global_total = total
            cls._global_checked_at = now
            return total

    def check(self) -> None:
        # own unflushed spend plus everyone else's last flush
        other = max(0.0, self.global_spend() - self._last_flushed)
        total = other + self.cost_usd
        if total >= self.budget_usd:
            raise BudgetExceeded(
                f"study budget reached: ${total:.4f} of ${self.budget_usd:.2f} "
                f"across all meters (this one: {self.name}, ${self.cost_usd:.4f})"
            )

    def record(self, cost: float, latency: float, input_tokens: int = 0) -> None:
        with self._lock:
            self.cost_usd += cost
            self.calls += 1
            self.input_tokens += input_tokens
            self.latencies.append(latency)
            if self.calls % max(1, self.flush_every) == 0:
                self._flush()

    def hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.summary()))
        tmp.replace(self.path)
        self._last_flushed = self.cost_usd
        type(self)._global_checked_at = 0.0     # force a refresh on next check

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def _flush_quiet(self) -> None:
        try:
            with self._lock:
                self._flush()
        except Exception:  # noqa: BLE001 - never raise from an atexit hook
            pass

    def summary(self) -> dict[str, Any]:
        lat = sorted(self.latencies)

        def q(p: float) -> float:
            if not lat:
                return 0.0
            return lat[min(len(lat) - 1, int(len(lat) * p))]

        return {
            "name": self.name,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "latency_p50_s": round(q(0.50), 4),
            "latency_p95_s": round(q(0.95), 4),
            "latency_mean_s": round(sum(lat) / len(lat), 4) if lat else 0.0,
        }


# --------------------------------------------------------------------------- #
#  Cache
# --------------------------------------------------------------------------- #
class _Cache:
    """SQLite keyed by a hash of (model, state, questions). Concurrent-safe
    enough for our use: many readers, occasional writers, single process."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value))
            )
            self._conn.commit()


def cache_key(model: str, state: Any, questions: dict) -> str:
    blob = json.dumps(
        {"m": model, "s": state, "q": questions}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- #
#  Client
# --------------------------------------------------------------------------- #
class JevClient:
    def __init__(
        self,
        model: str = JEV_MODEL,
        concurrency: int = JEV_MAX_CONCURRENCY,
        meter: Meter | None = None,
        cache_path: Path | None = None,
        use_cache: bool = True,
        max_retries: int = 6,
    ):
        self.model = model
        self.concurrency = concurrency
        self.meter = meter or Meter(name="jev")
        self.max_retries = max_retries
        self.use_cache = use_cache
        self._cache = _Cache(cache_path or CACHE_ROOT / "jev_cache.sqlite") if use_cache else None
        self._key = _api_key()
        self._backoff_base = 2.0
        self._throttle_lock = threading.Lock()

    # -- single call --------------------------------------------------------
    async def _post(
        self, client: httpx.AsyncClient, state: Any, questions: dict
    ) -> dict:
        key = cache_key(self.model, state, questions)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self.meter.hit()
                return hit

        self.meter.check()
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = await client.post(
                    JEV_ENDPOINT, headers=headers, json=payload, timeout=90.0
                )
                dt = time.perf_counter() - t0
                # 403 here is not an auth failure -- the key tests fine
                # immediately afterwards. It is the gateway shedding load from
                # a burst, and it needs a longer pause than a 5xx does.
                if r.status_code in (403, 429, 529) or r.status_code >= 500:
                    self._throttle(r.status_code)
                    raise httpx.HTTPStatusError(
                        f"retryable {r.status_code}", request=r.request, response=r
                    )
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage", {}) or {}
                self.meter.record(
                    float(usage.get("cost", 0.0)), dt, int(usage.get("input_tokens", 0))
                )
                if self._cache is not None:
                    self._cache.put(key, data)
                return data
            except Exception as e:  # noqa: BLE001 - retry everything transient
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                base = self._backoff_base
                await asyncio.sleep(min(120.0, base * 2**attempt) * (0.5 + random.random()))
        raise RuntimeError(f"Jev call failed after {self.max_retries} attempts: {last_err}")

    def _throttle(self, status: int) -> None:
        """Narrow the pipe after the gateway pushes back.

        Six retries topping out at 30s covers a blip, not a rate-limit window;
        a burst of 48 concurrent calls that trips one will trip it again on
        every retry. Halving the in-flight limit and lengthening the backoff
        makes the client yield instead of insisting.
        """
        if status not in (403, 429):
            return
        with self._throttle_lock:
            new = max(4, self.concurrency // 2)
            if new < self.concurrency:
                print(f"[jev] {status} from the gateway: concurrency "
                      f"{self.concurrency} -> {new}", flush=True)
                self.concurrency = new
            self._backoff_base = min(20.0, self._backoff_base * 2)

    # -- batch --------------------------------------------------------------
    async def aask_many(
        self, states: Sequence[Any], questions: dict
    ) -> list[dict]:
        # read the current limit: a previous batch may have narrowed it
        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(
            max_connections=self.concurrency + 8,
            max_keepalive_connections=self.concurrency + 8,
        )
        async with httpx.AsyncClient(limits=limits) as client:

            async def one(s: Any) -> dict:
                async with sem:
                    return await self._post(client, s, questions)

            return await asyncio.gather(*(one(s) for s in states))

    def ask_many(self, states: Sequence[Any], questions: dict) -> list[dict]:
        """Blocking entry point -- TRL reward functions are synchronous."""
        return _run_sync(self.aask_many(states, questions))

    def ask(self, state: Any, questions: dict) -> dict:
        return self.ask_many([state], questions)[0]


def _run_sync(coro):
    """Run a coroutine even if an event loop is already running in this thread
    (notebooks), without nest_asyncio."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["v"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            result["e"] = e

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if "e" in result:
        raise result["e"]
    return result["v"]
