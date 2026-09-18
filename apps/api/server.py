"""FastAPI backend for the product (D2): repos, on-demand indexing, profiles, files, and `/ask` as a C9 SSE stream.

Runs lane A's `RepoEnv.from_question` + `run_episode` with a cached `ModelClient` per profile. Citation badges come
from the grader's `check_citations` (C7) on the final answer; the driver's lighter `citations` event is dropped.
Streams are plain `text/event-stream` frames (`data: {json}\n\n`), one per C9 event, flattened `{type, ...payload}`.

    uv run uvicorn apps.api.server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from apps.api.repos import IndexJobs, list_repos, read_file, repo_summary, suggest_questions
from apps.api.workshop import router as workshop_router
from codeqa.agent.driver import run_episode, save_trace
from codeqa.agent.env import RepoEnv
from codeqa.clients.base import ModelClient, make_client
from codeqa.grader.citations import check_citations
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, SSEEvent, TaskType
from codeqa.shared.profiles import get_profile, load_profiles

TRACE_RUN = os.environ.get("CODEQA_TRACE_RUN", "product")
ASK_TIMEOUT = float(os.environ.get("CODEQA_ASK_TIMEOUT", "600"))


def log(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Model clients: one per profile, created on first use and kept alive.
# Tinker sampling clients take seconds to warm, so the lifespan warms them.
# ---------------------------------------------------------------------------

_clients: dict[str, ModelClient] = {}
_client_locks: dict[str, asyncio.Lock] = {}


async def client_for(name: str) -> ModelClient:
    if name in _clients:
        return _clients[name]
    lock = _client_locks.setdefault(name, asyncio.Lock())
    async with lock:
        if name not in _clients:
            profile = get_profile(name)
            t0 = time.time()
            _clients[name] = await asyncio.to_thread(make_client, profile)
            log(f"[api] client {name} ({profile.kind}) ready in {time.time()-t0:.1f}s")
    return _clients[name]


async def _warm_clients() -> None:
    for name, p in load_profiles().items():
        if p.kind != "tinker":
            continue
        try:
            await asyncio.wait_for(client_for(name), timeout=60)
        except Exception as e:  # noqa: BLE001 — warm-up is best effort
            log(f"[api] warm-up {name} failed: {type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    paths.ensure_dirs()
    if os.environ.get("CODEQA_WARM_CLIENTS", "1") == "1":
        asyncio.create_task(_warm_clients())
    yield


app = FastAPI(title="codeqa api", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs = IndexJobs(log=log)
app.include_router(workshop_router, tags=["workshop"])


# ---------------------------------------------------------------------------
# Repos
# ---------------------------------------------------------------------------

class AddRepo(BaseModel):
    url: str
    sha: str | None = None
    fast: bool = True


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "repos": len(list_repos()), "clients": sorted(_clients)}


@app.get("/repos")
def get_repos() -> list[dict[str, Any]]:
    ready = list_repos()
    known = {r["repo_id"] for r in ready}
    return ready + [r for r in jobs.in_progress_rows() if r["repo_id"] not in known]


@app.post("/repos")
async def post_repo(body: AddRepo) -> dict[str, Any]:
    try:
        job = jobs.start(body.url, body.sha, fast=body.fast)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"job_id": job.job_id, "repo_id": job.status()["repo_id"]}


@app.get("/repos/{job_id}/status")
def job_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")
    return job.status()


@app.get("/repos/{repo_id}/suggestions")
def repo_suggestions(repo_id: str) -> dict[str, Any]:
    if repo_summary(repo_id) is None:
        raise HTTPException(404, f"unknown or unindexed repo {repo_id}")
    return {"repo_id": repo_id, "questions": suggest_questions(repo_id)}


@app.get("/file", response_class=PlainTextResponse)
def get_file(repo_id: str, path: str, start: int | None = Query(None, ge=1), end: int | None = Query(None, ge=1)) -> str:
    if repo_summary(repo_id) is None and not (paths.repo_dir(repo_id) / "manifest.json").exists():
        raise HTTPException(404, f"unknown repo {repo_id}")
    try:
        return read_file(repo_id, path, start, end)
    except FileNotFoundError as e:
        raise HTTPException(404, f"no such file in {repo_id}: {path}") from e


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"^qwen4b-(?P<run>.+)-step(?P<step>\d+|final)$")


def profile_row(p: EndpointProfile) -> dict[str, Any]:
    label, note = p.model, ""
    if p.kind == "anthropic":
        label = {"claude-sonnet-5": "Claude Sonnet 5", "claude-haiku-4-5-20251001": "Claude Haiku 4.5"}.get(p.model, p.model)
        note = "teacher" if "sonnet" in p.model else "judge"
    elif p.kind == "tinker":
        m = _STEP_RE.match(p.name)
        if m:
            label, note = "Qwen3.5-4B, trained", f"{m['run']}, step {m['step']}"
        elif p.model.startswith("Qwen/"):
            label, note = "Qwen3.5-4B, untrained", "step 0"
    elif p.kind == "openai":
        label, note = "Qwen3.5-4B, served", "vLLM on Modal"
    return {"name": p.name, "kind": p.kind, "model": p.model, "label": label, "note": note}


@app.get("/profiles")
def get_profiles() -> list[dict[str, Any]]:
    return [profile_row(p) for p in load_profiles().values()]


# ---------------------------------------------------------------------------
# Ask: one episode, streamed as C9 events
# ---------------------------------------------------------------------------

class Ask(BaseModel):
    repo_id: str
    question: str = Field(min_length=1, max_length=4000)
    profile: str
    task_type: TaskType = "explain"
    temperature: float = Field(0.7, ge=0.0, le=2.0)


def _frame(ev: SSEEvent) -> str:
    return "data: " + json.dumps({"type": ev.type, **ev.payload}, ensure_ascii=False) + "\n\n"


async def episode_events(req: Ask) -> AsyncIterator[SSEEvent]:
    """Runs one episode and yields C9 events as they happen. The driver's own `citations` and `done` are replaced by
    the grader's citation check followed by `done`, so the UI's badges are the authoritative ones."""
    queue: asyncio.Queue[SSEEvent | None] = asyncio.Queue()

    async def on_event(ev: SSEEvent) -> None:
        if ev.type in ("citations", "done"):
            return
        await queue.put(ev)

    async def run() -> None:
        try:
            profile = get_profile(req.profile)
            env = RepoEnv.from_question(req.repo_id, req.question, profile, task_type=req.task_type)
            client = await client_for(req.profile)
            trace = await asyncio.wait_for(run_episode(env, client, on_event, temperature=req.temperature), timeout=ASK_TIMEOUT)
            if trace.answer:
                report = await asyncio.to_thread(check_citations, trace.answer, trace.stats.files_read, req.repo_id)
                items = [{"path": c.path, "start": c.start, "end": c.end, "exists": c.exists, "verified": c.exists and c.grounded}
                         for c in report.citations]
                await queue.put(SSEEvent(type="citations", payload={"items": items}))
            elif trace.stats.stop_reason != "error":
                await queue.put(SSEEvent(type="error", payload={"message": f"The agent stopped without answering ({trace.stats.stop_reason})."}))
            out = save_trace(trace, run=TRACE_RUN)
            log(f"[api] ask {req.profile} {req.repo_id}: {trace.stats.tool_calls} calls, {trace.stats.seconds}s, stop={trace.stats.stop_reason} -> {out}")
        except FileNotFoundError as e:
            await queue.put(SSEEvent(type="error", payload={"message": f"Repository is not indexed: {e}"}))
        except KeyError as e:
            await queue.put(SSEEvent(type="error", payload={"message": f"Unknown profile {req.profile}: {e}"}))
        except asyncio.TimeoutError:
            await queue.put(SSEEvent(type="error", payload={"message": f"Timed out after {ASK_TIMEOUT:.0f} s."}))
        except Exception as e:  # noqa: BLE001
            log(f"[api] ask failed: {type(e).__name__}: {e}")
            await queue.put(SSEEvent(type="error", payload={"message": f"{type(e).__name__}: {e}"[:300]}))
        finally:
            await queue.put(SSEEvent(type="done"))
            await queue.put(None)

    task = asyncio.create_task(run())
    try:
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield ev
    finally:
        if not task.done():
            task.cancel()


@app.post("/ask")
async def post_ask(req: Ask) -> StreamingResponse:
    if repo_summary(req.repo_id) is None:
        raise HTTPException(404, f"unknown or unindexed repo {req.repo_id}")
    if req.profile not in load_profiles():
        raise HTTPException(404, f"unknown profile {req.profile}")

    async def body() -> AsyncIterator[str]:
        async for ev in episode_events(req):
            yield _frame(ev)

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache", "x-accel-buffering": "no"})
