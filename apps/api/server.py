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

import hmac

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from apps.api.repos import IndexJobs, list_repos, read_file, repo_summary, suggest_items, suggest_questions
from apps.api.workshop import router as workshop_router
from apps.api.judge import router as judge_router
from codeqa.agent.driver import run_episode, save_trace
from codeqa.agent.env import RepoEnv
from codeqa.clients.base import ModelClient, make_client
from codeqa.grader.citations import check_citations
from codeqa.grader.gates import citations_parse_gate, format_gate
from apps.api.workshop import read_json, read_jsonl, run_titles
from codeqa.shared import paths
from codeqa.shared.contracts import DEFAULT_BUDGETS, UNLIMITED_CALLS, EndpointProfile, SSEEvent, TaskType
from codeqa.shared.profiles import load_profiles

TRACE_RUN = os.environ.get("CODEQA_TRACE_RUN", "product")
# Shared secret for the demo. One password for everyone; not real auth. Override with CODEQA_PASSWORD.
PASSWORD = os.environ.get("CODEQA_PASSWORD", "Action!")
PUBLIC_PATHS = {"/health", "/auth/login"}
ASK_TIMEOUT = float(os.environ.get("CODEQA_ASK_TIMEOUT", "600"))


def log(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Model clients: one per profile, created on first use and kept alive.
# Tinker sampling clients take seconds to warm, so the lifespan warms them.
# ---------------------------------------------------------------------------

_clients: dict[str, ModelClient] = {}
_client_locks: dict[str, asyncio.Lock] = {}


_CKPT_RE = re.compile(r"^qwen4b-(?P<run>[A-Za-z0-9_.\-]+)-step(?P<step>\d+|final)$")


SECOND_ORG_RUN_PREFIXES = ("p4_", "p5_", "p6_")      # runs trained in the second Tinker org (their samplers need TINKER_API_KEY_NEW)
DEFAULT_PROFILE = os.environ.get("CODEQA_DEFAULT_PROFILE", "qwen4b-p6_bash_v3-step24")   # best held-out so far (0.587 / 87 %, bash harness)   # what the product's model picker opens on


def checkpoint_profiles() -> dict[str, EndpointProfile]:
    """One profile per sampler checkpoint under data/logs/<run>/checkpoints.jsonl, named qwen4b-<run>-step<N> (lane A's
    convention), unless profiles.yaml already names that sampler path. Read-only: nothing is written to profiles.yaml."""
    yaml_profiles = load_profiles()
    known = {p.model for p in yaml_profiles.values()}
    out: dict[str, EndpointProfile] = {}
    if not paths.LOGS.exists():
        return out
    for run in sorted(paths.LOGS.iterdir()):
        rows = read_jsonl(run / "checkpoints.jsonl")
        if not rows:
            continue
        cfg = read_json(run / "config.json", {}) or {}
        base = cfg.get("model_name") or "Qwen/Qwen3.5-4B"
        for r in rows:
            sp = r.get("sampler_path")
            if not sp or sp in known:
                continue
            step = str(r.get("batch", r.get("name", "?")))
            name = f"qwen4b-{run.name}-step{step}"
            if name in out or name in yaml_profiles:  # `000003` and `final` rows share a batch; profiles.yaml wins
                continue
            key_env = "TINKER_API_KEY_NEW" if run.name.startswith(SECOND_ORG_RUN_PREFIXES) and os.environ.get("TINKER_API_KEY_NEW") else None
            out[name] = EndpointProfile(name=name, kind="tinker", model=sp, base_model=base, renderer=cfg.get("renderer_name") or "qwen3_5",
                                        max_context=32768, max_generation_tokens=int(cfg.get("max_tokens") or 2048), api_key_env=key_env,
                                        variant=run_variant(run.name, cfg))
    return out


def run_variant(run: str, cfg: dict[str, Any]) -> str | None:
    """The agent a run trained with: `runs.json` (the lead's titles file) first, then config.json. A checkpoint must be
    served with the tools, prompt and budget it trained on; None = the default variant."""
    meta = run_titles().get(run)
    v = (meta.get("variant") if isinstance(meta, dict) else None) or cfg.get("variant")   # runs.json may hold a plain string note
    return v if v and v != "none" else None


from apps.api.harness import V3_COMMANDS, V3_CONTEXT_TOKENS, V3_MESSAGES, V3_VARIANTS, apply_harness_knobs, harness_budget, harness_row, is_v3  # noqa: E402,F401


def all_profiles() -> dict[str, EndpointProfile]:
    return {**checkpoint_profiles(), **load_profiles()}


def resolve_profile(name: str) -> EndpointProfile:
    p = all_profiles().get(name)
    if p is None:
        raise KeyError(name)
    return p


async def client_for(name: str) -> ModelClient:
    if name in _clients:
        return _clients[name]
    lock = _client_locks.setdefault(name, asyncio.Lock())
    async with lock:
        if name not in _clients:
            profile = resolve_profile(name)
            t0 = time.time()
            _clients[name] = await asyncio.to_thread(make_client, profile)
            log(f"[api] client {name} ({profile.kind}) ready in {time.time()-t0:.1f}s")
    return _clients[name]


# Tinker's 400 once the server has ended our session: "This ServiceClient was cancelled and cannot run further
# operations", or "... has finished (interrupted) and cannot run further operations" (seen after a congested hour).
DEAD_SESSION_MARK = "cannot run further operations"


def is_dead_session(message: str) -> bool:
    return DEAD_SESSION_MARK in message


def reset_tinker_session() -> None:
    """Drop the process-wide Tinker session and every client built on it. The next ask opens one fresh session
    (the product keeps exactly one; training jobs have their own). Called when Tinker answers 400 "cancelled":
    the server ends a session whose heartbeats stopped, and a cached client would fail every call forever."""
    from codeqa.clients import tinker as tk
    for name in [n for n, c in _clients.items() if c.profile.kind == "tinker"]:
        _clients.pop(name, None)
    tk.sampling_client.cache_clear()
    tk.service_client.cache_clear()
    log("[api] tinker session was ended by the server; dropped it, the next ask opens a new one")


# A sampling client that lived through a congested hour can wedge inside the SDK's per-client futures poller (every
# abandoned request, from a Stop or a timeout, stays in its poll batch): every sample then hangs until CHAT_TIMEOUT
# although a fresh client answers in a second (seen 23:24-23:47 on 2026-09-18; the 9B client in the same session was
# fine). Before a Tinker ask, if that profile has not completed a call recently, send a 1-token probe with a short
# timeout; on a hang rebuild the sampling clients and keep the session.
PROBE_AFTER_IDLE = float(os.environ.get("CODEQA_TINKER_PROBE_IDLE", "120"))
PROBE_TIMEOUT = float(os.environ.get("CODEQA_TINKER_PROBE_TIMEOUT", "20"))
_tinker_last_ok: dict[str, float] = {}


def mark_tinker_ok(name: str) -> None:
    _tinker_last_ok[name] = time.time()


async def probe_tinker(client: ModelClient) -> bool:
    """One-token sample on the client's cached sampling session. False = hung or failed."""
    import tinker
    try:
        await asyncio.wait_for(
            client._sc.sample_async(prompt=tinker.ModelInput.from_ints([client.tokenizer.eos_token_id or 0]), num_samples=1,  # type: ignore[attr-defined]
                                    sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0)),
            timeout=PROBE_TIMEOUT)
        return True
    except Exception as e:  # noqa: BLE001 — any failure means rebuild
        log(f"[api] tinker probe failed for {client.profile.name}: {type(e).__name__}: {str(e)[:120]}")
        return False


def rebuild_sampling_client(name: str) -> None:
    """Drop this profile's client and the SDK's cached sampling clients; the session (and its heartbeat) stay."""
    from codeqa.clients import tinker as tk
    _clients.pop(name, None)
    tk.sampling_client.cache_clear()
    log(f"[api] rebuilt the sampling client for {name}")


async def ensure_tinker_alive(name: str, profile: EndpointProfile) -> ModelClient:
    """The client for `name`, on a sampling client that answered a probe if the profile has been idle for PROBE_AFTER_IDLE."""
    client = await client_for(name)
    if profile.kind != "tinker" or time.time() - _tinker_last_ok.get(name, 0.0) < PROBE_AFTER_IDLE:
        return client
    if await probe_tinker(client):
        mark_tinker_ok(name)
        return client
    rebuild_sampling_client(name)
    return await client_for(name)


def record_adhoc(task_id: str, repo_id: str, question: str, task_type: str, profile: str) -> None:
    """Product traces carry only a hashed task id; this sidecar lets the Workshop find the repo (and so check citations)."""
    p = paths.TRACES / TRACE_RUN / "adhoc.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"task_id": task_id, "repo_id": repo_id, "question": question, "task_type": task_type, "profile": profile}) + "\n")


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


def _password_ok(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    supplied = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-password", "")
    return hmac.compare_digest(supplied, PASSWORD)


@app.middleware("http")
async def require_password(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS or _password_ok(request):
        return await call_next(request)
    return JSONResponse({"detail": "password required"}, status_code=401, headers={"www-authenticate": "Bearer"})


class Login(BaseModel):
    password: str


@app.post("/auth/login")
def login(body: Login) -> dict[str, bool]:
    if not hmac.compare_digest(body.password, PASSWORD):
        raise HTTPException(401, "wrong password")
    return {"ok": True}
app.include_router(workshop_router, tags=["workshop"])
app.include_router(judge_router, tags=["judge"])


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
    items = suggest_items(repo_id)
    return {"repo_id": repo_id, "questions": [i["question"] for i in items], "items": items}


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


# Named product models: a trained checkpoint promoted out of the checkpoint list with a name, like a release.
# "Scholia": the marginal notes of ancient scholars, each one citing the line of the text it comments on.
NAMED_MODELS: dict[str, tuple[str, str]] = {
    "scholia-bash-v3": ("Scholia 4B (bash_v3)", "bash agent, rounds harness, trained 24 steps"),
}


def profile_row(p: EndpointProfile, from_checkpoints: bool = False) -> dict[str, Any]:
    label, note = p.model, ""
    if p.name in NAMED_MODELS:
        label, note = NAMED_MODELS[p.name]
    elif p.kind == "anthropic":
        label = {"claude-sonnet-5": "Claude Sonnet 5", "claude-haiku-4-5-20251001": "Claude Haiku 4.5"}.get(p.model, p.model)
        note = "teacher" if "sonnet" in p.model else "judge"
    elif p.kind == "tinker":
        m = _STEP_RE.match(p.name)
        if m:
            label, note = "Qwen3.5-4B, trained", f"{m['run']}, step {m['step']}"
        elif p.model.startswith("Qwen/"):
            label, note = f"{p.model.split('/', 1)[1]}, untrained", "step 0"
    elif p.kind == "openai":
        label, note = "Qwen3.5-4B, served", "vLLM on Modal"
    if from_checkpoints:
        note = f"{note} (checkpoint)" if note else "checkpoint"
    if p.kind == "tinker":
        row_extra = harness_row(p)
    else:
        row_extra = {}
    return {"name": p.name, "kind": p.kind, "model": p.model, "label": label, "note": note, "source": "checkpoints" if from_checkpoints else "profiles.yaml",
            "default": p.name == DEFAULT_PROFILE, **row_extra}


@app.get("/profiles")
def get_profiles() -> list[dict[str, Any]]:
    rows = [profile_row(p) for p in load_profiles().values()]
    rows += [profile_row(p, from_checkpoints=True) for p in checkpoint_profiles().values()]
    return rows


# ---------------------------------------------------------------------------
# Ask: one episode, streamed as C9 events
# ---------------------------------------------------------------------------

class Ask(BaseModel):
    repo_id: str
    question: str = Field(min_length=1, max_length=4000)
    profile: str
    task_type: TaskType = "explain"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    # Run this ask under a given agent harness (tools, prompt, budget) instead of the profile's own, e.g. every column
    # of a comparison on Scholia's bash_v3 harness. None = the profile's variant.
    variant: str | None = None


def _frame(ev: SSEEvent) -> str:
    return "data: " + json.dumps({"type": ev.type, **ev.payload}, ensure_ascii=False) + "\n\n"


async def episode_events(req: Ask) -> AsyncIterator[SSEEvent]:
    """Runs one episode and yields C9 events as they happen. The driver's own `citations` and `done` are replaced by
    the grader's citation check followed by `done`, so the UI's badges are the authoritative ones."""
    queue: asyncio.Queue[SSEEvent | None] = asyncio.Queue()
    clock = {"t0": time.time()}  # reset when the episode starts, so `t` matches the driver's stats.seconds
    timing = {"model": 0.0, "tools": 0.0, "mark": 0.0, "open_call": None}  # seconds; `mark` = when the model last got control

    state = {"dead_session": False, "rerun": False}

    async def on_event(ev: SSEEvent) -> None:
        if ev.type in ("citations", "done"):
            return
        if ev.type == "error" and is_dead_session(str(ev.payload.get("message", ""))) and not state["rerun"]:
            state["dead_session"] = True   # swallowed, with the rest of this attempt: the episode is re-run on a fresh session
        if state["dead_session"] and not state["rerun"]:
            return
        now = time.time() - clock["t0"]
        # Model time runs from the last tool result (or the start) to the first event the model produces next.
        if ev.type in ("thinking", "tool_call", "answer") and timing["mark"] is not None:
            timing["model"] += now - timing["mark"]
            timing["mark"] = None
        if ev.type == "tool_call":
            timing["open_call"] = now
        if ev.type == "tool_result":
            if timing["open_call"] is not None:
                timing["tools"] += now - timing["open_call"]
                timing["open_call"] = None
            timing["mark"] = now
        if ev.type == "stats":
            ev = SSEEvent(type="stats", payload={**ev.payload, "model_seconds": round(timing["model"], 2), "tool_seconds": round(timing["tools"], 2)})
        await queue.put(SSEEvent(type=ev.type, payload={**ev.payload, "t": round(now, 3)}))

    async def run() -> None:
        try:
            profile = resolve_profile(req.profile)
            apply_harness_knobs(profile, req.variant)
            env = RepoEnv.from_question(req.repo_id, req.question, profile, task_type=req.task_type,
                                        budget=harness_budget(profile, req.task_type, req.variant), variant=req.variant)
            client = await ensure_tinker_alive(req.profile, profile)
            clock["t0"] = time.time()
            timing["mark"] = 0.0
            trace = await asyncio.wait_for(run_episode(env, client, on_event, temperature=req.temperature), timeout=ASK_TIMEOUT)
            if state["dead_session"]:
                # Tinker ended the session under us (heartbeats missed, or ended from the console). Nothing was
                # answered on the old one, so open a fresh session and run the same episode once more.
                reset_tinker_session()
                env = RepoEnv.from_question(req.repo_id, req.question, profile, task_type=req.task_type,
                                            budget=harness_budget(profile, req.task_type, req.variant), variant=req.variant)
                client = await client_for(req.profile)
                clock["t0"] = time.time()
                timing.update(model=0.0, tools=0.0, mark=0.0, open_call=None)
                state["rerun"] = True
                trace = await asyncio.wait_for(run_episode(env, client, on_event, temperature=req.temperature), timeout=ASK_TIMEOUT)
            # Format verdict, the same gates the grader applies first.
            fmt_ok, fmt_why = format_gate(trace.answer, trace, env.budget)
            cit_ok, cit_why = citations_parse_gate(trace.answer) if trace.answer else (False, "no final answer")
            if trace.answer:
                report = await asyncio.to_thread(check_citations, trace.answer, trace.stats.files_read, req.repo_id)
                items = [{"path": c.path, "start": c.start, "end": c.end, "exists": c.exists, "verified": c.exists and c.grounded}
                         for c in report.citations]
                await queue.put(SSEEvent(type="citations", payload={"items": items, "format_ok": fmt_ok and cit_ok,
                                                                    "format_reason": (fmt_why or cit_why) if not (fmt_ok and cit_ok) else "",
                                                                    "t": round(time.time() - clock["t0"], 3)}))
            elif trace.stats.stop_reason != "error":
                await queue.put(SSEEvent(type="error", payload={"message": f"The agent stopped without answering ({trace.stats.stop_reason})."}))
            if profile.kind == "tinker" and trace.stats.stop_reason != "error":
                mark_tinker_ok(req.profile)
            out = save_trace(trace, run=TRACE_RUN)
            record_adhoc(trace.task_id, req.repo_id, req.question, req.task_type, req.profile)
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
    if req.profile not in all_profiles():
        raise HTTPException(404, f"unknown profile {req.profile}")
    if req.variant:
        from codeqa.agent.variants import VARIANTS
        if req.variant not in VARIANTS:
            raise HTTPException(400, f"unknown variant {req.variant!r}; one of {', '.join(VARIANTS)}")

    async def body() -> AsyncIterator[str]:
        async for ev in episode_events(req):
            yield _frame(ev)

    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache", "x-accel-buffering": "no"})
