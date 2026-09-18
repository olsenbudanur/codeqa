"""Small helpers for the bulk Haiku/Sonnet calls datagen makes: profiles, a bounded async runner, and a JSONL cache.

Every call has a hard timeout (rule 6) and every result is cached by key so reruns are free.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from codeqa.clients.anthropic import AnthropicClient
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Message

HAIKU = EndpointProfile(name="haiku-datagen", kind="anthropic", model="claude-haiku-4-5", max_generation_tokens=400, thinking=False)
SONNET = EndpointProfile(name="sonnet-datagen", kind="anthropic", model="claude-sonnet-5", max_generation_tokens=1200, thinking=False)
CALL_TIMEOUT = 60.0
CACHE_DIR = paths.DATA / "cache" / "rewrites"


class JsonlCache:
    """key -> dict, appended as it goes; safe to reread after a crash."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]
        self._f = self.path.open("a")

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.data[key] = value
        self._f.write(json.dumps({"key": key, "value": value}) + "\n")
        self._f.flush()


async def ask(client: AnthropicClient, system: str, user: str, max_tokens: int | None = None) -> str:
    msgs = [Message(role="system", content=system), Message(role="user", content=user)]
    reply = await asyncio.wait_for(client.chat(msgs, max_tokens=max_tokens), CALL_TIMEOUT)
    return reply.content


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The first {...} object in a reply, tolerant of fences and prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


async def run_bounded(items: list[Any], fn: Callable[[Any], Awaitable[Any]], concurrency: int = 8, label: str = "llm") -> list[Any]:
    """Run fn over items with bounded concurrency; exceptions become None; progress every 25 items."""
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.time()
    results: list[Any] = [None] * len(items)

    async def one(i: int, item: Any) -> None:
        nonlocal done
        async with sem:
            try:
                results[i] = await fn(item)
            except Exception as e:  # timeouts, API errors: skip the item, keep going
                print(f"  [{label}] item {i} failed: {type(e).__name__}: {str(e)[:120]}", flush=True, file=sys.stderr)
            done += 1
            if done % 25 == 0 or done == len(items):
                print(f"  [{label}] {done}/{len(items)}  {time.time()-t0:.0f}s", flush=True, file=sys.stderr)

    await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    return results
