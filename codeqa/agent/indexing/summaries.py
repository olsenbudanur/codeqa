"""Semantic layer: one paragraph per directory, one line per large file, from Haiku, once. (C2 summaries.json)

Input per call is the structural index only (names, signatures, line counts, README head), never whole files,
so a repo costs ~one call per directory. Resumable: existing entries in summaries.json are kept.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, IndexSymbol, Manifest, Message

HAIKU = "claude-haiku-4-5-20251001"
LARGE_FILE_LINES = 300
MAX_DIRS = 400
CONCURRENCY = 8

DIR_PROMPT = """You write one-paragraph summaries of source directories for a code navigation index.
Given the listing below, write ONE paragraph (at most 60 words, plain text, no markdown, no preamble) saying what the
directory contains, what it is for, and its main entry points, naming concrete classes or functions from the listing.
If it is documentation, tests, or configuration, say so briefly.

Directory: {path}
{listing}"""

FILE_PROMPT = """You write one-line summaries of source files for a code navigation index.
Given the file's symbols below, write ONE line (at most 25 words, plain text, no markdown) saying what the file
implements and its main entry points, naming concrete symbols.

File: {path} ({lines} lines)
{listing}"""


def summaries_path(repo_id: str) -> Path:
    return paths.index_dir(repo_id) / "summaries.json"


def load_summaries(repo_id: str) -> dict[str, str]:
    p = summaries_path(repo_id)
    return json.loads(p.read_text()) if p.exists() else {}


def _dir_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else "."


def _listing_for_dir(d: str, manifest: Manifest, by_file: dict[str, list[IndexSymbol]], root: Path) -> str:
    lines: list[str] = []
    subdirs = sorted({_dir_of(f.path) for f in manifest.files if _dir_of(f.path) != d and _dir_of(_dir_of(f.path)) == d})
    for sd in subdirs[:30]:
        lines.append(f"{sd.rsplit('/', 1)[-1]}/")
    files = [f for f in manifest.files if _dir_of(f.path) == d]
    for f in files[:40]:
        syms = by_file.get(f.path, [])
        top = [s.name for s in syms if s.parent is None][:12]
        lines.append(f"{f.path.rsplit('/', 1)[-1]}  ({f.lines} lines)" + (f"  symbols: {', '.join(top)}" if top else ""))
    if len(files) > 40:
        lines.append(f"(+{len(files) - 40} more files)")
    for f in files:
        if f.path.rsplit("/", 1)[-1].lower().startswith("readme"):
            try:
                head = (root / f.path).read_text(errors="replace").splitlines()[:20]
                lines.append("README head:\n" + "\n".join(head))
            except OSError:
                pass
            break
    return "\n".join(lines)


def _listing_for_file(f_path: str, by_file: dict[str, list[IndexSymbol]]) -> str:
    syms = by_file.get(f_path, [])
    out = []
    for s in syms[:60]:
        indent = "  " if s.parent else ""
        out.append(f"{indent}{s.signature}  (L{s.start}-L{s.end})")
    return "\n".join(out) or "(no symbols)"


async def summarize(manifest: Manifest, symbols: list[IndexSymbol], *, model: str = HAIKU,
                    concurrency: int = CONCURRENCY, log=print) -> dict[str, str]:
    from codeqa.clients.anthropic import AnthropicClient
    repo_id = manifest.repo_id
    root = paths.repo_dir(repo_id)
    existing = load_summaries(repo_id)
    by_file: dict[str, list[IndexSymbol]] = defaultdict(list)
    for s in symbols:
        by_file[s.path].append(s)

    dirs = sorted({_dir_of(f.path) for f in manifest.files})
    # code directories first so a capped run still covers what matters
    has_code = {d for f in manifest.files if f.lang in ("python", "javascript", "typescript", "tsx", "go", "rust", "java", "c", "cpp") for d in [_dir_of(f.path)]}
    dirs = sorted(dirs, key=lambda d: (d not in has_code, d.startswith("."), d.count("/"), d))[:MAX_DIRS]
    large = [f for f in manifest.files if f.lines >= LARGE_FILE_LINES and by_file.get(f.path)]

    jobs: list[tuple[str, str]] = []
    for d in dirs:
        if d not in existing:
            jobs.append((d, DIR_PROMPT.format(path=d, listing=_listing_for_dir(d, manifest, by_file, root))))
    for f in large:
        if f.path not in existing:
            jobs.append((f.path, FILE_PROMPT.format(path=f.path, lines=f.lines, listing=_listing_for_file(f.path, by_file))))
    log(f"[summaries] {repo_id}: {len(dirs)} dirs, {len(large)} large files, {len(jobs)} to do, {len(existing)} cached")
    if not jobs:
        return existing

    client = AnthropicClient(EndpointProfile(name="haiku", kind="anthropic", model=model, max_generation_tokens=200))
    sem = asyncio.Semaphore(concurrency)
    out = dict(existing)
    failures: list[str] = []

    async def one(key: str, prompt: str) -> None:
        async with sem:
            for attempt in range(2):
                try:
                    msg = await asyncio.wait_for(client.chat([Message(role="user", content=prompt)], max_tokens=200), timeout=60)
                    out[key] = " ".join(msg.content.split())
                    return
                except Exception as e:  # noqa: BLE001
                    if attempt == 1:
                        failures.append(f"{key}: {type(e).__name__}: {e}")

    await asyncio.gather(*(one(k, p) for k, p in jobs))
    p = summaries_path(repo_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, sort_keys=True))
    log(f"[summaries] {repo_id}: wrote {len(out)} entries ({len(failures)} failed) -> {p}")
    for f in failures[:5]:
        log("  fail:", f)
    return out
