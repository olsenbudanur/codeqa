"""Repo listing and on-demand indexing jobs for the product API (C1, gap_specs §8).

`list_repos` reads every `data/repos/*/manifest.json` that has a `map.txt` (the `__nodoc` variants are training-only
and hidden). `IndexJobs` runs snapshot → index → map, marks the repo ready, then fills in summaries and rebuilds the
map in the background (`fast` mode). Progress is reported per stage for the web stepper.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from codeqa.agent.indexing.index import build_index, load_symbols
from codeqa.agent.indexing.repomap import build_map
from codeqa.agent.indexing.snapshot import load_manifest, snapshot
from codeqa.agent.indexing.summaries import load_summaries, summarize
from codeqa.shared import paths

GITHUB_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s#?]+)")

_symbol_counts: dict[str, tuple[float, int]] = {}  # repo_id -> (symbols.json mtime, count)


def _symbol_count(repo_id: str) -> int | None:
    p = paths.index_dir(repo_id) / "symbols.json"
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    hit = _symbol_counts.get(repo_id)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        n = len(json.loads(p.read_text())["symbols"])
    except (OSError, ValueError, KeyError):
        return None
    _symbol_counts[repo_id] = (mtime, n)
    return n


def repo_summary(repo_id: str) -> dict[str, Any] | None:
    """One `GET /repos` row, or None when the repo has no manifest or no map yet."""
    mpath = paths.repo_dir(repo_id) / "manifest.json"
    if not mpath.exists() or not (paths.index_dir(repo_id) / "map.txt").exists():
        return None
    m = json.loads(mpath.read_text())
    files = m.get("files", [])
    return {
        "repo_id": repo_id,
        "url": m.get("url", ""),
        "sha": m.get("sha", ""),
        "files": len(files),
        "lines": sum(int(f.get("lines", 0)) for f in files),
        "symbols": _symbol_count(repo_id),
        "stage": "ready",
        "summaries": (paths.index_dir(repo_id) / "summaries.json").exists(),
    }


def list_repos(include_nodoc: bool = False) -> list[dict[str, Any]]:
    out = []
    if not paths.REPOS.exists():
        return out
    for mpath in sorted(paths.REPOS.glob("*/manifest.json")):
        rid = mpath.parent.name
        if rid.endswith("__nodoc") and not include_nodoc:
            continue
        row = repo_summary(rid)
        if row:
            out.append(row)
    return out


def parse_github_url(url: str) -> tuple[str, str]:
    m = GITHUB_URL_RE.search(url.strip())
    if not m:
        raise ValueError("Paste a GitHub URL like https://github.com/owner/repo")
    return m.group(1), m.group(2).removesuffix(".git")


STAGE_PROGRESS = {"snapshot": 0.15, "index": 0.55, "summaries": 0.8, "ready": 1.0, "error": 0.0}


@dataclass
class IndexJob:
    job_id: str
    owner: str
    repo: str
    ref: str
    fast: bool
    repo_id: str | None = None
    stage: str = "snapshot"
    message: str | None = None
    started: float = field(default_factory=time.time)
    task: asyncio.Task | None = None

    def status(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "repo_id": self.repo_id or f"{self.owner}__{self.repo}__{self.ref[:7]}",
            "stage": self.stage,
            "progress": STAGE_PROGRESS.get(self.stage, 0.0),
            "seconds": round(time.time() - self.started, 1),
            "message": self.message,
        }


class IndexJobs:
    def __init__(self, log=print) -> None:
        self.jobs: dict[str, IndexJob] = {}
        self.log = log

    def get(self, job_id: str) -> IndexJob | None:
        return self.jobs.get(job_id)

    def in_progress_rows(self) -> list[dict[str, Any]]:
        """Placeholder `GET /repos` rows for jobs that are not ready yet."""
        rows = []
        for j in self.jobs.values():
            if j.stage in ("ready", "error"):
                continue
            rows.append({"repo_id": j.status()["repo_id"], "url": f"https://github.com/{j.owner}/{j.repo}", "sha": j.ref,
                         "files": 0, "lines": 0, "symbols": None, "stage": j.stage, "summaries": False})
        return rows

    def start(self, url: str, sha: str | None = None, fast: bool = True) -> IndexJob:
        owner, repo = parse_github_url(url)
        job = IndexJob(job_id=f"job_{uuid.uuid4().hex[:10]}", owner=owner, repo=repo, ref=sha or "HEAD", fast=fast)
        job.task = asyncio.create_task(self._run(job))  # needs a running loop: call from an async endpoint
        self.jobs[job.job_id] = job
        return job

    async def _run(self, job: IndexJob) -> None:
        t0 = time.time()
        try:
            job.stage = "snapshot"
            manifest = await asyncio.to_thread(snapshot, job.owner, job.repo, job.ref)
            job.repo_id = manifest.repo_id
            self.log(f"[api] {job.job_id} snapshot {manifest.repo_id}: {len(manifest.files)} files ({time.time()-t0:.1f}s)")

            job.stage = "index"
            symbols = await asyncio.to_thread(build_index, manifest)
            summaries = load_summaries(manifest.repo_id)
            await asyncio.to_thread(build_map, manifest, symbols, summaries)
            self.log(f"[api] {job.job_id} index {len(symbols)} symbols, map written ({time.time()-t0:.1f}s)")

            if job.fast:
                job.stage = "ready"  # askable now; summaries fill in below
            else:
                job.stage = "summaries"

            if not summaries:
                try:
                    summaries = await summarize(manifest, symbols, log=lambda *a: None)
                    await asyncio.to_thread(build_map, manifest, symbols, summaries)
                    self.log(f"[api] {job.job_id} summaries {len(summaries)} dirs, map rebuilt ({time.time()-t0:.1f}s)")
                except Exception as e:  # noqa: BLE001 — summaries are optional; the repo stays askable
                    self.log(f"[api] {job.job_id} summaries failed: {type(e).__name__}: {e}")
                    if not job.fast:
                        job.message = f"Indexed without summaries: {type(e).__name__}"
            job.stage = "ready"
        except Exception as e:  # noqa: BLE001
            job.stage = "error"
            job.message = f"{type(e).__name__}: {e}"[:300]
            self.log(f"[api] {job.job_id} failed: {job.message}")
            traceback.print_exc()


def read_file(repo_id: str, rel_path: str, start: int | None = None, end: int | None = None) -> str:
    """File text from the snapshot; `start`/`end` are 1-based inclusive lines. Refuses paths outside the repo."""
    root = paths.repo_dir(repo_id).resolve()
    target = (root / rel_path).resolve()
    if root not in target.parents or not target.is_file():
        raise FileNotFoundError(rel_path)
    text = target.read_text(errors="replace")
    if start is None and end is None:
        return text
    lines = text.splitlines(keepends=True)
    s = max(1, start or 1)
    e = min(len(lines), end or len(lines))
    return "".join(lines[s - 1 : e])


__all__ = ["IndexJobs", "IndexJob", "list_repos", "repo_summary", "read_file", "parse_github_url", "load_manifest", "load_symbols"]


# ---------------------------------------------------------------------------
# Suggested questions, from the index: one locate, one trace, one explain.
# ---------------------------------------------------------------------------

_SKIP_DIRS = ("test", "tests", "testing", "docs", "doc", "examples", "example", "benchmarks", "scripts", "conftest")


def _is_core_path(path: str) -> bool:
    parts = path.lower().split("/")
    return not any(p in _SKIP_DIRS or p.startswith("test_") or p.endswith("_test.py") for p in parts)


_DEFAULT_ARG_RE = re.compile(r"(\w+)\s*(?::[^=,)]+)?=\s*([^,)]+)")


def suggest_items(repo_id: str) -> list[dict[str, str]]:
    """One question per task type (locate, value, enumerate, trace, explain), from the index."""
    owner_repo = repo_id.split("__")
    name = owner_repo[1] if len(owner_repo) >= 2 else repo_id
    generic = [
        {"type": "locate", "question": f"Where is the main entry point of {name} defined?"},
        {"type": "value", "question": f"What version string does {name} declare, and where?"},
        {"type": "enumerate", "question": f"Which modules make up the public API of {name}?"},
        {"type": "trace", "question": f"Trace what happens when {name} handles an error."},
        {"type": "explain", "question": f"How does {name} decide what to do on startup, and why?"},
    ]
    try:
        symbols = load_symbols(repo_id)
    except (OSError, ValueError, KeyError):
        return generic
    core = [s for s in symbols if _is_core_path(s.path) and not s.name.startswith("_")]
    methods_per_class: dict[tuple[str, str], int] = {}
    for s in core:
        if s.kind == "method" and s.parent:
            methods_per_class[(s.path, s.parent)] = methods_per_class.get((s.path, s.parent), 0) + 1
    classes = sorted((s for s in core if s.kind == "class"),
                     key=lambda s: (-methods_per_class.get((s.path, s.name), 0), s.path.count("/"), s.path))
    functions = sorted((s for s in core if s.kind == "function" and "/" in s.path and not s.path.endswith(("setup.py", "version.py", "_version.py", "__main__.py"))),
                       key=lambda s: (-(s.end - s.start), s.path.count("/"), s.path))
    out: dict[str, str] = {}
    if classes:
        out["locate"] = f"Where is the `{classes[0].name}` class defined, and what does it inherit from?"
    # value: a parameter with a default in a function or method signature
    for s in core:
        sig = getattr(s, "signature", "") or ""
        if s.kind in ("function", "method") and "(" in sig:
            m = _DEFAULT_ARG_RE.search(sig[sig.index("(") + 1:])
            if m and m.group(2).strip() not in ("None", "", "...", "self"):
                owner = f"`{s.parent}.{s.name}`" if s.parent else f"`{s.name}`"
                out["value"] = f"What is the default value of `{m.group(1)}` in {owner}?"
                break
    # enumerate: a class with a handful of methods
    mid = [c for c in classes if 3 <= methods_per_class.get((c.path, c.name), 0) <= 12]
    if mid:
        out["enumerate"] = f"Which methods does the `{mid[0].name}` class define?"
    if functions:
        out["trace"] = f"Trace what happens when `{functions[0].name}` is called."
    if len(classes) > 1:
        out["explain"] = f"What is `{classes[1].name}` responsible for, and where is it used?"
    return [{"type": g["type"], "question": out.get(g["type"], g["question"])} for g in generic]


def suggest_questions(repo_id: str, n: int = 5) -> list[str]:
    return [i["question"] for i in suggest_items(repo_id)][:n]
