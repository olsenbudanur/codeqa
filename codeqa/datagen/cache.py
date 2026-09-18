"""Page cache for Hugging Face datasets-server rows -> data/cache/hf/<dataset>__<config>__<split>.jsonl.

datasets-server serves 100 rows per call and 502s now and then, so every full pull is done once and kept.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from codeqa.clients import hf
from codeqa.shared import paths

CACHE = paths.DATA / "cache" / "hf"


def cache_path(dataset: str, config: str, split: str) -> Path:
    return CACHE / f"{dataset.replace('/', '__')}__{config}__{split}.jsonl"


def fetch_all(dataset: str, config: str, split: str, force: bool = False, page: int = 100, pause: float = 0.6) -> Path:
    """Pull every row of one split into the cache (idempotent, resumable). Prints one line per page, unbuffered.
    `pause` between pages keeps us under the datasets-server rate limit (~1 page/s tripped a 429 after 46 pages)."""
    out = cache_path(dataset, config, split)
    if out.exists() and not force:
        return out
    total = hf.size(dataset, config, split)
    tmp = out.with_suffix(".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # resume: pages are written whole, so the line count of a .part file is the next offset
    n = sum(1 for _ in tmp.open()) if tmp.exists() else 0
    n -= n % page
    t0 = time.time()
    with tmp.open("r+" if n else "w") as f:
        if n:
            for _ in range(n):
                f.readline()
            f.truncate()
        for off in range(n, total, page):
            for r in hf.rows(dataset, config, split, off, page):
                f.write(json.dumps(r) + "\n")
                n += 1
            f.flush()
            print(f"  [{dataset}/{split}] {n}/{total} rows  {time.time()-t0:.0f}s", flush=True, file=sys.stderr)
            time.sleep(pause)
    tmp.rename(out)
    return out


def iter_cached(dataset: str, config: str, split: str) -> Iterator[dict[str, Any]]:
    p = fetch_all(dataset, config, split)
    with p.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_cached(dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    return list(iter_cached(dataset, config, split))
