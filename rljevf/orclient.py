"""Minimal async OpenRouter chat client, used for (a) generating Jev question
sets and (b) the independent evaluation judge. Shares Meter/cache machinery
with the Jev client so every dollar in the paper is accounted for."""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from .config import CACHE_ROOT, OPENROUTER_CHAT_ENDPOINT
from .jevclient import Meter, _Cache, _api_key, _run_sync, cache_key


class ChatClient:
    def __init__(
        self,
        model: str,
        concurrency: int = 16,
        meter: Meter | None = None,
        cache_path: Path | None = None,
        use_cache: bool = True,
        max_retries: int = 6,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.concurrency = concurrency
        self.meter = meter or Meter(name=f"chat_{model.replace('/', '_')}")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._cache = (
            _Cache(cache_path or CACHE_ROOT / "chat_cache.sqlite") if use_cache else None
        )
        self._key = _api_key()

    async def _post(self, client: httpx.AsyncClient, messages: list[dict], **over) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": over.get("temperature", self.temperature),
            "max_tokens": over.get("max_tokens", self.max_tokens),
        }
        for k in ("response_format", "logprobs", "top_logprobs", "reasoning",
                  "provider", "seed"):
            if k in over:
                body[k] = over[k]
        key = cache_key(self.model, messages, body)
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self.meter.hit()
                return hit
        self.meter.check()
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = await client.post(
                    OPENROUTER_CHAT_ENDPOINT, headers=headers, json=body, timeout=180.0
                )
                dt = time.perf_counter() - t0
                if r.status_code in (429, 529) or r.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage", {}) or {}
                self.meter.record(
                    float(usage.get("cost", 0.0) or 0.0), dt, int(usage.get("prompt_tokens", 0) or 0)
                )
                if self._cache is not None:
                    self._cache.put(key, data)
                return data
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt == self.max_retries - 1:
                    break
                await asyncio.sleep(min(30.0, 2**attempt) * (0.5 + random.random()))
        raise RuntimeError(f"chat call failed: {last}")

    async def acomplete_raw(self, conversations: Sequence[list[dict]], **over) -> list[dict]:
        """Full response objects -- needed when the caller wants logprobs
        rather than text."""
        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency + 4)
        async with httpx.AsyncClient(limits=limits) as client:

            async def one(m: list[dict]) -> dict:
                async with sem:
                    return await self._post(client, m, **over)

            return await asyncio.gather(*(one(m) for m in conversations))

    async def acomplete_many(
        self, conversations: Sequence[list[dict]], **over
    ) -> list[str]:
        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency + 4)
        async with httpx.AsyncClient(limits=limits) as client:

            async def one(m: list[dict]) -> str:
                async with sem:
                    d = await self._post(client, m, **over)
                    return d["choices"][0]["message"]["content"] or ""

            return await asyncio.gather(*(one(m) for m in conversations))

    def complete_many(self, conversations: Sequence[list[dict]], **over) -> list[str]:
        return _run_sync(self.acomplete_many(conversations, **over))

    def complete(self, messages: list[dict], **over) -> str:
        return self.complete_many([messages], **over)[0]


def extract_json(text: str) -> Any:
    """Models wrap JSON in prose or fences more often than they should."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for op, cl in (("{", "}"), ("[", "]")):
        i, j = t.find(op), t.rfind(cl)
        if i != -1 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in: {text[:200]!r}")
