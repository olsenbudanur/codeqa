"""Hugging Face datasets-server: rows without downloading whole datasets."""
from __future__ import annotations

import os
import time
from typing import Any, Iterator

import httpx
from dotenv import load_dotenv

load_dotenv()
BASE = "https://datasets-server.huggingface.co"


def _headers() -> dict[str, str]:
    h = {"User-Agent": "codeqa"}
    if os.environ.get("HF_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    return h


def _get(path: str, params: dict[str, Any], retries: int = 6) -> dict[str, Any]:
    err: Exception | None = None
    for i in range(retries):
        try:
            r = httpx.get(f"{BASE}/{path}", params=params, headers=_headers(), timeout=60)
            if r.status_code == 429:  # rate limited: honour Retry-After (datasets-server sends it), else 20s
                wait = min(float(r.headers.get("retry-after", 20)), 55)
                print(f"  datasets-server 429; sleeping {wait:.0f}s", flush=True)
                time.sleep(wait)
                err = RuntimeError("429 rate limited")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # 502s are common; back off
            err = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"datasets-server {path} failed: {err}")


def splits(dataset: str) -> list[tuple[str, str]]:
    return [(s["config"], s["split"]) for s in _get("splits", {"dataset": dataset})["splits"]]


def size(dataset: str, config: str, split: str) -> int:
    for s in _get("size", {"dataset": dataset, "config": config, "split": split})["size"]["splits"]:
        if s["config"] == config and s["split"] == split:
            return int(s["num_rows"])
    raise KeyError((config, split))


def rows(dataset: str, config: str, split: str, offset: int = 0, length: int = 100) -> list[dict[str, Any]]:
    data = _get("rows", {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": min(length, 100)})
    return [r["row"] for r in data["rows"]]


def iter_rows(dataset: str, config: str, split: str, limit: int | None = None, page: int = 100) -> Iterator[dict[str, Any]]:
    total = size(dataset, config, split)
    n = 0
    for off in range(0, total, page):
        for r in rows(dataset, config, split, off, page):
            yield r
            n += 1
            if limit is not None and n >= limit:
                return
