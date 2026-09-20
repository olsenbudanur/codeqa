"""Read-only workshop API (lane D6): training runs, checkpoints, data, traces. Parses files on request with a 5 s
cache keyed by path + mtime. Nothing here launches, grades with a judge, or writes.

Trace ids (URL path segments):
  trace/<run>/<file-stem>                    data/traces/<run>/<stem>.json
  eval/<profile>/<set>/<task_id>             data/evals/<profile>/<set>/traces/<task_id>.json (+ per_task row)
  rollout/<run>/<iteration>/<group>/<traj>   data/logs/<run>/iteration_NNNNNN/train_rollout_summaries.jsonl row
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, Query

from codeqa.grader.citations import check_citations
from codeqa.shared import paths
from codeqa.shared.contracts import Task, Trace
from codeqa.shared.profiles import load_profiles

router = APIRouter()

CACHE_TTL = 5.0
_cache: dict[str, tuple[float, float, Any]] = {}  # key -> (checked_at, mtime, value)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def cached(key: str, p: Path, build):
    """Rebuild when the file changed; re-check the mtime at most every CACHE_TTL seconds."""
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[2]
    m = _mtime(p)
    if hit and hit[1] == m:
        _cache[key] = (now, m, hit[2])
        return hit[2]
    v = build()
    _cache[key] = (now, m, v)
    return v


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def read_json(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

METRIC_PREFIXES = ("env/", "eval/", "optim/", "progress/", "kl_ref/", "loss/", "time/train_step", "time/total",
                   "time/compute_group_rewards:total", "time/policy_sample:total", "time/env_step:total", "time/run_evaluations_parallel")
LIVE_SECONDS = 1200.0  # a step can take 3–12 min when several jobs share Tinker, plus a 2-min sync and a 30 s commit; 330 s blinked runs off the Live board between steps (2026-09-19)
_growth: dict[str, tuple[int, float]] = {}  # run -> (rows last seen, time the row count last grew)


def _is_live(name: str, n_rows: int, mtime: float, planned: int | None) -> bool:
    """Live = new metric rows keep arriving. mtime alone lies here: the volume sync rewrites every file it pulls."""
    now = time.time()
    seen = _growth.get(name)
    if seen is None:
        # first sight this process: trust a fresh mtime once
        _growth[name] = (n_rows, mtime if now - mtime < LIVE_SECONDS else 0.0)
    elif n_rows != seen[0]:   # any change is activity (a count can drop when the volume sync replaces a file, e.g. after a duplicate job is stopped)
        _growth[name] = (n_rows, now)
    last_growth = _growth[name][1]
    if planned and n_rows >= planned:
        return False
    return now - last_growth < LIVE_SECONDS


def run_titles() -> dict[str, dict[str, Any]]:
    """Readable names and hypotheses per run, from data/logs/runs.json (lane C, LOG 2026-09-20)."""
    p = paths.LOGS / "runs.json"
    return cached("runs.json", p, lambda: read_json(p, {}) or {})


def run_dir(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
        raise HTTPException(400, "bad run name")
    return paths.LOGS / name


def _config_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    ds = cfg.get("dataset_builder") or {}
    return {
        "learning_rate": cfg.get("learning_rate"),
        "model_name": cfg.get("model_name"),
        "group_size": ds.get("group_size"),
        "groups_per_batch": ds.get("groups_per_batch"),
        "tasks_path": ds.get("tasks_path"),
        "profile_name": ds.get("profile_name"),
        "epochs": ds.get("epochs"),
        "variant": ds.get("variant"),
        "judge_model": ds.get("judge_model"),
        "eval_every": cfg.get("eval_every"),
        "save_every": cfg.get("save_every"),
        "max_tokens": cfg.get("max_tokens"),
        "lora_rank": cfg.get("lora_rank"),
    }


def _final_eval(name: str) -> tuple[Path | None, dict[str, Any]]:
    """The arm's final temperature-0.2 eval of the last checkpoint (`data/evals/<profile>-<run>-step<N>/fast_t02/results.json`).
    The in-loop evaluator never fires on the last step, so this is the held-out point for step N."""
    hits = sorted(paths.EVALS.glob(f"*-{name}-step*/fast_t02/results.json"))
    if not hits:
        return None, {}
    d = read_json(hits[-1], {}) or {}
    return hits[-1], (d.get("summary") or d)


def _own_metrics(name: str) -> list[dict[str, Any]]:
    p = run_dir(name) / "metrics.jsonl"
    fe_path, fe = _final_eval(name)

    def build():
        rows = []
        for r in read_jsonl(p):
            keep = {k: v for k, v in r.items() if k == "step" or k.startswith(METRIC_PREFIXES)}
            keep = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in keep.items()}
            rows.append(keep)
        # a preempted-and-resumed run re-appends the steps it redid (Modal restarts the function; the trainer resumes from the
        # last checkpoint): keep the LAST row per (step, kind) so the chart shows each step once (2026-09-20 17:10, p6_bash_v4)
        seen: dict[tuple[int, bool], int] = {}
        for i, r in enumerate(rows):
            seen[(int(r.get("step", i)), any(k.startswith("env/") for k in r))] = i
        rows = [r for i, r in enumerate(rows) if seen[(int(r.get("step", i)), any(k.startswith("env/") for k in r))] == i]
        # async training writes a held-out as its own row next to that step's training row: merge them so one x has one row
        # (two rows at one x broke the chart's hover on the held-out point, 2026-09-20)
        by_step: dict[int, dict[str, Any]] = {}
        merged: list[dict[str, Any]] = []
        for r in rows:
            st = int(r.get("step", 0))
            if any(k.startswith("env/") for k in r):
                if st in by_step:
                    by_step[st].update({k: v for k, v in r.items() if k not in by_step[st] or k.startswith("env/")})
                else:
                    by_step[st] = r; merged.append(r)
            elif st in by_step:
                by_step[st].update({k: v for k, v in r.items() if k != "step"})
            else:
                by_step[st] = r; merged.append(r)
        rows = sorted(merged, key=lambda r: int(r.get("step", 0)))
        if rows and fe:     # the final eval measures the checkpoint AFTER the last step: x = steps completed, like the in-loop points (0, 8, ...)
            extra: dict[str, Any] = {"step": int(rows[-1].get("step", len(rows) - 1)) + 1, "eval/fast/final_t02": 1.0}
            for src, dst in (("reward", "reward"), ("correct_rate", "correct"), ("tool_calls", "tool_calls"), ("prompt_tokens", "prompt_tokens"),
                             ("completion_tokens", "completion_tokens"), ("citations_grounded", "citations_grounded"), ("correctness", "correctness")):
                if isinstance(fe.get(src), (int, float)):
                    extra[f"eval/fast/env/all/{dst}"] = float(fe[src])
            rows.append(extra)
        return rows

    return cached(f"metrics:{name}:{_mtime(fe_path) if fe_path else 0}", p, build)


def _run_metrics(name: str) -> list[dict[str, Any]]:
    """A fork (runs.json `fork_of`) draws on its parent's graph: the parent's rows first (its final-eval point kept, since the
    fork's own step 0 held-out re-measures the same weights), then the fork's rows shifted by the parent's step count."""
    own = _own_metrics(name)
    meta = run_titles().get(name, {})
    parent = meta.get("fork_of") if isinstance(meta, dict) else None
    if not parent or not (run_dir(parent) / "metrics.jsonl").exists():
        return own
    prows = _own_metrics(parent)
    offset = sum(1 for r in prows if any(k.startswith("env/") for k in r))
    shifted = []
    for r in own:
        q = {k: v for k, v in r.items() if not (int(r.get("step", 0)) == 0 and k.startswith("eval/"))}   # the fork's step-0 held-out re-measures
        if len(q) <= 1:                                                                                  # the parent's final weights: drop it
            continue
        q["step"] = int(q.get("step", 0)) + offset; q["fork"] = 1.0
        shifted.append(q)
    parent_rows = [dict(r, parent=1.0) for r in prows]
    final = next((r for r in parent_rows if not any(k.startswith("env/") for k in r) and int(r.get("step", -1)) == offset), None)
    if final is not None and shifted and int(shifted[0]["step"]) == offset:      # one row at x = offset: the parent's final held-out + the fork's first step
        shifted[0].update({k: v for k, v in final.items() if k != "step"})
        parent_rows = [r for r in parent_rows if r is not final]
    rows_out = parent_rows + shifted
    until = meta.get("show_until") if isinstance(meta, dict) else None       # display cutoff (runs.json): rows past it stay on disk, off the chart
    if isinstance(until, int):
        rows_out = [r for r in rows_out if int(r.get("step", 0)) <= until]
    return rows_out


def run_row(name: str) -> dict[str, Any] | None:
    d = run_dir(name)
    cfg_p, met_p = d / "config.json", d / "metrics.jsonl"
    if not d.is_dir() or not (cfg_p.exists() or met_p.exists()):
        return None
    rows = _run_metrics(name)
    last = _mtime(met_p)
    iterations = sorted(int(p.name.split("_")[1]) for p in d.glob("iteration_*") if p.is_dir())
    meta = run_titles().get(name, {})
    if not isinstance(meta, dict):   # runs.json may hold a plain string note for a run
        meta = {"hypothesis": str(meta)}
    planned = meta.get("steps") if isinstance(meta.get("steps"), int) else None
    return {
        "name": name,
        "title": meta.get("title") or name,
        "hypothesis": meta.get("hypothesis"),
        "variant": meta.get("variant"),
        "planned_steps": planned,
        "reward_setting": meta.get("reward"),
        "steps": sum(1 for r in rows if any(k.startswith("env/") for k in r)),    # training rows only (a final-eval row carries no env/ keys); a fork counts its parent's too
        "fork_of": meta.get("fork_of"),
        "own_steps": sum(1 for r in _own_metrics(name) if any(k.startswith("env/") for k in r)),
        "started": _mtime(cfg_p) if cfg_p.exists() else None,
        "updated": last if last > 0 else None,
        "live": last > 0 and _is_live(name, len(rows), last, planned),
        "config": _config_summary(read_json(cfg_p, {}) or {}),
        "iterations": iterations,
        "last_reward": next((r.get("env/all/reward/total") for r in reversed(rows) if r.get("env/all/reward/total") is not None), None),
    }


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    if not paths.LOGS.exists():
        return []
    rows = [run_row(p.name) for p in sorted(paths.LOGS.iterdir()) if p.is_dir()]
    rows = [r for r in rows if r]
    rows.sort(key=lambda r: r["updated"] or 0, reverse=True)
    return rows


@router.get("/runs/{name}")
def get_run(name: str) -> dict[str, Any]:
    row = run_row(name)
    if row is None:
        raise HTTPException(404, f"unknown run {name}")
    cfg = read_json(run_dir(name) / "config.json", {}) or {}
    checkpoints = read_jsonl(run_dir(name) / "checkpoints.jsonl")
    metrics = _run_metrics(name)
    return {**row, "config_full": cfg, "metrics": metrics, "checkpoints": checkpoints, "warnings": run_warnings(metrics)}


def run_warnings(rows: list[dict[str, Any]]) -> list[str]:
    """The same collapse checks the CLI monitor prints (`codeqa.evals.monitor.checks`), so the two never disagree.
    Import-rule note: apps/api reaches into evals for this one pure function; a move to codeqa/shared is requested in LOG."""
    try:
        from codeqa.evals.monitor import checks
    except Exception:  # noqa: BLE001 — monitor is optional
        return []
    try:
        return list(checks(rows))
    except Exception as e:  # noqa: BLE001
        return [f"health checks failed: {type(e).__name__}: {e}"]


def _iteration_path(name: str, n: int) -> Path:
    return run_dir(name) / f"iteration_{n:06d}" / "train_rollout_summaries.jsonl"


def _tool_sequence(rollout: dict[str, Any]) -> list[str]:
    seq = []
    for st in rollout.get("steps", []):
        logs = st.get("logs", {})
        for k in sorted(k for k in logs if k.startswith("tool_call_")):
            m = re.match(r"(\w+)\(", str(logs[k]))
            seq.append(m.group(1) if m else "?")
    return seq


def _answer_excerpt(rollout: dict[str, Any], n: int = 240) -> str:
    steps = rollout.get("steps", [])
    if not steps:
        return ""
    text = str(steps[-1].get("logs", {}).get("assistant_content", "")).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


@router.get("/runs/{name}/iterations/{n}")
def get_iteration(name: str, n: int) -> dict[str, Any]:
    p = _iteration_path(name, n)
    if not p.exists():
        raise HTTPException(404, f"no iteration {n} for run {name}")
    rollouts = cached(f"iter:{name}:{n}", p, lambda: read_jsonl(p))
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rollouts:
        tm = r.get("trajectory_metrics", {})
        groups[int(r.get("group_idx", 0))].append({
            "id": f"rollout/{name}/{n}/{r.get('group_idx', 0)}/{r.get('traj_idx', 0)}",
            "traj_idx": r.get("traj_idx"),
            "tags": r.get("tags", []),
            "reward": r.get("total_reward"),
            "metrics": {k: tm.get(k) for k in ("format_ok", "citations_grounded", "correctness", "efficiency", "tool_calls", "prompt_tokens", "turns")},
            "stop": next((k[5:] for k in ("stop_answer", "stop_max_turns", "stop_budget", "stop_overflow", "stop_parse_error", "stop_error") if tm.get(k)), None),
            "tool_sequence": _tool_sequence(r),
            "answer_excerpt": _answer_excerpt(r),
        })
    return {"run": name, "iteration": n, "groups": [{"group_idx": g, "trajectories": ts} for g, ts in sorted(groups.items())]}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

EVAL_KEYS = ("n", "reward", "correct_rate", "format_ok", "citation_valid", "citations_grounded", "tool_calls", "tool_calls_per_correct",
             "prompt_tokens", "answer_tokens", "stop_answer", "sweqa_total", "sweqa_correctness")


def eval_summaries(profile: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    d = paths.EVALS / profile
    if not d.is_dir():
        return out
    for s in sorted(d.iterdir()):
        rp = s / "results.json"
        if not rp.exists():
            continue
        res = cached(f"eval:{profile}/{s.name}", rp, lambda rp=rp: read_json(rp, {}) or {})
        summ = res.get("summary", {})
        row = {k: summ.get(k) for k in EVAL_KEYS if k in summ}
        judge = s / "sweqa_judge.json"
        if judge.exists():
            j = cached(f"judge:{profile}/{s.name}", judge, lambda judge=judge: read_json(judge, {}) or {})
            for k in ("total", "correctness"):
                v = j.get(k) if isinstance(j, dict) else None
                if v is not None:
                    row[f"sweqa_{k}"] = v
        row["seconds"] = res.get("seconds")
        row["temperature"] = res.get("temperature")
        out[s.name] = row
    return out


@router.get("/checkpoints")
def list_checkpoints() -> dict[str, Any]:
    manifest = cached("manifest", paths.MODELS_MANIFEST, lambda: read_json(paths.MODELS_MANIFEST, []) or [])
    profiles = load_profiles()
    rows = []
    for rec in manifest:
        prof = profiles.get(rec.get("profile") or "")
        rows.append({
            **{k: rec.get(k) for k in ("name", "run", "step", "created_at", "tinker_path", "merged_path", "modal_volume", "profile", "is_final", "notes")},
            "profile_kind": prof.kind if prof else None,
            "servable": bool(rec.get("merged_path") or rec.get("modal_volume")),
            "evals": {**(rec.get("evals") or {}), **eval_summaries(rec.get("profile") or "")},
        })
    baselines = []
    for name in ("qwen4b-base", "claude"):
        p = profiles.get(name)
        baselines.append({"name": name, "profile": name, "kind": p.kind if p else None, "model": p.model if p else None, "evals": eval_summaries(name)})
    sets = sorted({s for r in rows + baselines for s in r["evals"]})
    return {"checkpoints": rows, "baselines": baselines, "sets": sets}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

TASK_FILES = {
    "raw": ["codescout", "deepcodebench", "structural", "teacher"],
    "train": ["all"],
    "eval": ["deepcodebench_test", "sweqa", "fast"],
}


def _tasks(split: str, name: str) -> list[dict[str, Any]]:
    p = paths.TASKS / split / f"{name}.jsonl"
    return cached(f"tasks:{split}/{name}", p, lambda: read_jsonl(p))


def _passrate() -> dict[str, dict[str, Any]]:
    p = paths.TASKS / "reports" / "passrate.jsonl"
    return cached("passrate", p, lambda: {r["task_id"]: r for r in read_jsonl(p) if "task_id" in r})


def difficulty(row: dict[str, Any] | None) -> float | None:
    """Lane B's difficulty score: content-only correctness over answered samples when >= 2 answered, else found-gold rate."""
    if not row:
        return None
    if (row.get("n_answered") or 0) >= 2 and row.get("correct_lenient") is not None:
        return float(row["correct_lenient"])
    v = row.get("found")
    return float(v) if v is not None else None


def bucket(d: float | None, lo: float = 0.1, hi: float = 0.9) -> str | None:
    if d is None:
        return None
    return "too_hard" if d < lo else "too_easy" if d > hi else "kept"


@router.get("/data/summary")
def data_summary() -> dict[str, Any]:
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for split, names in TASK_FILES.items():
        for name in names:
            c: dict[str, Counter] = defaultdict(Counter)
            for t in _tasks(split, name):
                c[t.get("source", "?")][t.get("task_type", "?")] += 1
            counts[f"{split}/{name}"] = {s: dict(v) for s, v in c.items()}
    repos = []
    if paths.REPOS.exists():
        for mp in sorted(paths.REPOS.glob("*/manifest.json")):
            rid = mp.parent.name
            m = cached(f"manifest:{rid}", mp, lambda mp=mp: read_json(mp, {}) or {})
            files = m.get("files", [])
            idx = paths.index_dir(rid)
            sym = idx / "symbols.json"
            nsym = cached(f"nsym:{rid}", sym, lambda sym=sym: len((read_json(sym, {}) or {}).get("symbols", []))) if sym.exists() else None
            repos.append({"repo_id": rid, "url": m.get("url"), "files": len(files), "lines": sum(int(f.get("lines", 0)) for f in files),
                          "symbols": nsym, "summaries": (idx / "summaries.json").exists(), "map": (idx / "map.txt").exists(),
                          "nodoc": rid.endswith("__nodoc")})
    readme = paths.TASKS / "README.md"
    return {"counts": counts, "repos": repos, "readme": readme.read_text() if readme.exists() else ""}


@router.get("/data/passrate")
def data_passrate(bins: int = Query(10, ge=2, le=50)) -> dict[str, Any]:
    pr = _passrate()
    hist = [0] * bins
    per_source: dict[str, dict[str, float]] = {}
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_scored = 0
    for r in pr.values():
        d = difficulty(r)
        if d is not None:
            n_scored += 1
            hist[min(bins - 1, int(d * bins))] += 1
        src = r.get("source", "?")
        for k in ("answered", "format_ok", "found", "correct_lenient_all", "reward", "tool_calls"):
            v = r.get(k)
            if isinstance(v, (int, float)):
                acc[src][k].append(float(v))
    for src, ks in acc.items():
        per_source[src] = {k: sum(v) / len(v) for k, v in ks.items() if v}
        per_source[src]["n"] = len(ks.get("answered", []))
    buckets = Counter(bucket(difficulty(r)) for r in pr.values())
    summary = read_json(paths.TASKS / "reports" / "passrate_summary.json", {})
    return {"bins": bins, "histogram": hist, "n": len(pr), "n_scored": n_scored, "window": [0.1, 0.9],
            "buckets": {k or "unscored": v for k, v in buckets.items()}, "per_source": per_source, "summary": summary}


def _trace_index_for_task() -> dict[str, list[str]]:
    """task_id -> trace ids, over data/traces/** and data/evals/*/*/traces."""
    def build():
        idx: dict[str, list[str]] = defaultdict(list)
        for run in sorted(paths.TRACES.iterdir()) if paths.TRACES.exists() else []:
            if run.is_dir():
                for f in run.glob("*.json"):
                    tid = f.stem.split("__", 1)[0]
                    idx[tid].append(f"trace/{run.name}/{f.stem}")
        for prof in sorted(paths.EVALS.iterdir()) if paths.EVALS.exists() else []:
            for s in prof.iterdir() if prof.is_dir() else []:
                td = s / "traces"
                if td.is_dir():
                    for f in td.glob("*.json"):
                        idx[f.stem].append(f"eval/{prof.name}/{s.name}/{f.stem}")
        return dict(idx)
    # keyed on the traces dir mtime; new runs/sets bump it, new files inside a run are picked up within the TTL
    return cached("trace-index", paths.TRACES, build)


@router.get("/data/tasks")
def data_tasks(q: str | None = None, source: str | None = None, type: str | None = None, split: str = "train",
               repo: str | None = None, bucket_: str | None = Query(None, alias="bucket"), limit: int = Query(30, ge=1, le=200),
               offset: int = Query(0, ge=0), random_: bool = Query(False, alias="random")) -> dict[str, Any]:
    if split == "train":
        rows = _tasks("train", "all")
    elif split == "eval":
        rows = [t for n in TASK_FILES["eval"] for t in _tasks("eval", n)]
    elif split == "raw":
        rows = [t for n in TASK_FILES["raw"] for t in _tasks("raw", n)]
    else:
        raise HTTPException(400, "split must be train, eval or raw")
    pr = _passrate()
    ql = (q or "").lower()
    out = []
    for t in rows:
        if source and t.get("source") != source:
            continue
        if type and t.get("task_type") != type:
            continue
        if repo and t.get("repo_id") != repo:
            continue
        if ql and ql not in t.get("question", "").lower() and ql not in t.get("task_id", "").lower():
            continue
        p = pr.get(t["task_id"])
        b = bucket(difficulty(p))
        if bucket_ and b != bucket_:
            continue
        out.append((t, p, b))
    total = len(out)
    if random_:
        random.shuffle(out)
    else:
        out = out[offset: offset + limit]
    out = out[:limit]
    traces = _trace_index_for_task()
    cards = []
    for t, p, b in out:
        cards.append({**t, "passrate": {k: p.get(k) for k in ("n", "answered", "format_ok", "found", "correct_lenient", "correct_lenient_all", "reward", "tool_calls", "stops")} if p else None,
                      "difficulty": difficulty(p), "bucket": b, "example_traces": traces.get(t["task_id"], [])})
    return {"total": total, "offset": offset, "limit": limit, "tasks": cards}


@router.get("/data/repos")
def data_repos() -> list[dict[str, Any]]:
    return data_summary()["repos"]


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

def _task_index() -> dict[str, dict[str, Any]]:
    def build():
        idx: dict[str, dict[str, Any]] = {}
        for split, names in TASK_FILES.items():
            for n in names:
                for t in _tasks(split, n):
                    idx.setdefault(t["task_id"], t)
        for extra in ("smoke_sweqa_flask", "sweqa_100"):
            for t in _tasks("eval", extra):
                idx.setdefault(t["task_id"], t)
        for extra in ("smoke_graphiti", "run1", "run1_verifiable"):
            for t in _tasks("train", extra):
                idx.setdefault(t["task_id"], t)
        for extra in ("reserve_easy", "reserve_hard", "smoke_codescout", "smoke_deepcodebench"):
            for t in _tasks("raw", extra):
                idx.setdefault(t["task_id"], t)
        return idx
    return cached("task-index", paths.TASKS, build)


def trace_events(trace: Trace) -> list[dict[str, Any]]:
    """C9 events reconstructed from a C6 trace, so the UI can replay it through the same reducer."""
    from codeqa.agent.rounds import FORCED_MARK
    ev: list[dict[str, Any]] = []
    turn = 0
    for m in trace.messages:
        if m.role == "assistant":
            turn += 1
            if m.thinking:
                ev.append({"type": "thinking", "text": m.thinking})
            if m.parse_error:
                ev.append({"type": "error", "message": m.parse_error[:300]})
            for tc in m.tool_calls:
                ev.append({"type": "tool_call", "name": tc.name, "args": tc.args, "turn": turn})   # turn: rounds (bash_v3)
        elif m.role == "tool" and m.name not in (None, "budget"):
            text = m.content or ""
            ev.append({"type": "tool_result", "name": m.name, "summary": text.splitlines()[0][:160] if text else "", "chars": len(text), "text": text[:4000]})
        elif m.role == "user" and (m.content or "").startswith(FORCED_MARK):
            ev.append({"type": "notice", "kind": "forced_answer", "text": m.content})
    if trace.answer:
        ev.append({"type": "answer", "markdown": trace.answer})
    st = trace.stats
    ev.append({"type": "stats", "tool_calls": st.tool_calls, "tool_errors": st.tool_errors, "prompt_tokens": st.prompt_tokens,
               "completion_tokens": st.completion_tokens, "seconds": st.seconds, "stop_reason": st.stop_reason, "turns": st.turns,
               "forced_answer": bool(getattr(st, "forced_answer", False))})
    ev.append({"type": "done"})
    return ev


def rollout_events(r: dict[str, Any]) -> list[dict[str, Any]]:
    ev: list[dict[str, Any]] = []
    answer = ""
    for st in r.get("steps", []):
        logs = st.get("logs", {})
        content = str(logs.get("assistant_content", "")).strip()
        calls = sorted(k for k in logs if k.startswith("tool_call_"))
        if content and calls:
            ev.append({"type": "thinking", "text": content})
        for k in calls:
            m = re.match(r"(\w+)\((.*)\)\s*$", str(logs[k]), re.S)
            name, args = (m.group(1), m.group(2)) if m else (str(logs[k]), "{}")
            try:
                parsed = json.loads(args)
            except ValueError:
                parsed = {"raw": args}
            ev.append({"type": "tool_call", "name": name, "args": parsed, "turn": int(st.get("step_idx", 0)) + 1})
            res = str(logs.get("tool_result_" + k.split("_")[-1], ""))
            ev.append({"type": "tool_result", "name": name, "summary": res.splitlines()[0][:160] if res else "", "chars": len(res), "text": res[:4000]})
        if content and not calls:
            answer = content
    if answer:
        ev.append({"type": "answer", "markdown": answer})
    tm = r.get("trajectory_metrics", {})
    ev.append({"type": "stats", "tool_calls": tm.get("tool_calls"), "tool_errors": tm.get("tool_errors"), "prompt_tokens": tm.get("prompt_tokens"),
               "completion_tokens": tm.get("completion_tokens"), "seconds": None, "turns": tm.get("turns"),
               "stop_reason": next((k[5:] for k in ("stop_answer", "stop_max_turns", "stop_budget", "stop_overflow", "stop_parse_error", "stop_error") if tm.get(k)), None)})
    ev.append({"type": "done"})
    return ev


def rollout_messages(r: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st in r.get("steps", []):
        logs = st.get("logs", {})
        calls = []
        for k in sorted(k for k in logs if k.startswith("tool_call_")):
            m = re.match(r"(\w+)\((.*)\)\s*$", str(logs[k]), re.S)
            name, args = (m.group(1), m.group(2)) if m else (str(logs[k]), "{}")
            try:
                parsed = json.loads(args)
            except ValueError:
                parsed = {"raw": args}
            calls.append({"name": name, "args": parsed, "call_id": k})
        out.append({"role": "assistant", "content": str(logs.get("assistant_content", "")), "thinking": None, "tool_calls": calls,
                    "usage": {"prompt_tokens": st.get("ob_len"), "completion_tokens": st.get("ac_len")}, "reward": st.get("reward")})
        for k in sorted(k for k in logs if k.startswith("tool_result_")):
            idx = k.split("_")[-1]
            call = next((c for c in calls if c["call_id"] == f"tool_call_{idx}"), None)
            out.append({"role": "tool", "name": call["name"] if call else "?", "content": str(logs[k]), "call_id": f"tool_call_{idx}"})
    return out


def _grade_from_metrics(m: dict[str, Any], notes: str = "", gate: str | None = None) -> dict[str, Any]:
    comps = {k: m.get(k) for k in ("format_ok", "citations_parse", "citations_exist", "citations_grounded", "identifier_grounded", "correctness", "efficiency")}
    if gate is None:
        gate = next((g for g in ("format", "citations", "grounding", "budget", "judge_error") if m.get(f"gate_{g}")), None)
    return {"reward": m.get("reward"), "components": comps, "gate_failed": gate, "notes": notes, "source": "graded"}


def _citation_table(answer: str, files_read, repo_id: str) -> list[dict[str, Any]]:
    if not answer or not (paths.repo_dir(repo_id) / "manifest.json").exists():
        return []
    try:
        rep = check_citations(answer, files_read, repo_id)
    except Exception:  # noqa: BLE001
        return []
    return [{"path": c.path, "start": c.start, "end": c.end, "exists": c.exists, "grounded": c.grounded, "verified": c.exists and c.grounded} for c in rep.citations]


def _list_row(id_: str, task_id: str, profile: str, run: str, stats: dict[str, Any], task: dict[str, Any] | None,
              reward: float | None, has_citations: bool, mtime: float | None) -> dict[str, Any]:
    return {
        "id": id_, "task_id": task_id, "profile": profile, "run": run,
        "repo_id": task.get("repo_id") if task else None,
        "source": task.get("source") if task else None, "task_type": task.get("task_type") if task else None,
        "question": (task.get("question") if task else None) or "",
        "stop_reason": stats.get("stop_reason"), "turns": stats.get("turns"), "tool_calls": stats.get("tool_calls"),
        "prompt_tokens": stats.get("prompt_tokens"), "reward": reward, "has_citations": has_citations, "mtime": mtime,
    }


def _per_task_rows(profile: str, set_: str) -> dict[str, dict[str, Any]]:
    p = paths.EVALS / profile / set_ / "per_task.jsonl"
    return cached(f"per_task:{profile}/{set_}", p, lambda: {r["task_id"]: r for r in read_jsonl(p)})


def _light_trace_stats(p: Path) -> dict[str, Any]:
    """Stats + flags without validating the whole trace."""
    d = read_json(p, {}) or {}
    st = d.get("stats", {}) or {}
    return {**st, "has_citations": "[" in (d.get("answer") or "") and ":L" in (d.get("answer") or ""), "task_id": d.get("task_id"), "profile": d.get("profile")}


def _groups_map(run: str, n: int) -> dict[int, dict[str, Any]]:
    """iteration_N/groups.json (written by the trainer since 2026-09-20): group_idx -> task fields. Empty for older runs."""
    p = run_dir(run) / f"iteration_{n:06d}" / "groups.json"
    if not p.exists():
        return {}
    try:
        rows = cached(f"groups:{run}:{n}", p, lambda: json.loads(p.read_text()))
        return {int(r["group_idx"]): r for r in rows}
    except Exception:  # noqa: BLE001
        return {}


def all_trace_rows() -> list[dict[str, Any]]:
    tasks = _task_index()
    rows: list[dict[str, Any]] = []
    if paths.TRACES.exists():
        for run in sorted(paths.TRACES.iterdir()):
            if not run.is_dir():
                continue
            for f in sorted(run.glob("*.json")):
                st = cached(f"lt:{f}", f, lambda f=f: _light_trace_stats(f))
                tid = st.get("task_id") or f.stem.split("__", 1)[0]
                prof = st.get("profile") or (f.stem.split("__", 1)[1] if "__" in f.stem else "?")
                rows.append(_list_row(f"trace/{run.name}/{f.stem}", tid, prof, run.name, st, tasks.get(tid), None, st["has_citations"], _mtime(f)))
    if paths.EVALS.exists():
        for prof in sorted(paths.EVALS.iterdir()):
            if not prof.is_dir():
                continue
            for s in sorted(prof.iterdir()):
                td = s / "traces"
                if not td.is_dir():
                    continue
                per = _per_task_rows(prof.name, s.name)
                for f in sorted(td.glob("*.json")):
                    st = cached(f"lt:{f}", f, lambda f=f: _light_trace_stats(f))
                    pt = per.get(f.stem, {})
                    rows.append(_list_row(f"eval/{prof.name}/{s.name}/{f.stem}", f.stem, prof.name, f"eval:{s.name}", st, tasks.get(f.stem),
                                          pt.get("reward"), st["has_citations"], _mtime(f)))
    if paths.LOGS.exists():
        for run in sorted(paths.LOGS.iterdir()):
            for p in sorted(run.glob("iteration_*/train_rollout_summaries.jsonl")):
                n = int(p.parent.name.split("_")[1])
                rollouts = cached(f"iter:{run.name}:{n}", p, lambda p=p: read_jsonl(p))
                gmap = _groups_map(run.name, n)
                for r in rollouts:
                    tm = r.get("trajectory_metrics", {})
                    tags = r.get("tags", [])
                    ans = _answer_excerpt(r, 100000)
                    ginfo = gmap.get(int(r.get("group_idx", -1)), {})
                    task = tasks.get(ginfo.get("task_id", "")) or {"source": tags[0] if tags else None, "task_type": tags[1] if len(tags) > 1 else None, "question": ginfo.get("question", "")}
                    stats = {"stop_reason": next((k[5:] for k in ("stop_answer", "stop_max_turns", "stop_budget", "stop_overflow", "stop_parse_error", "stop_error") if tm.get(k)), None),
                             "turns": tm.get("turns"), "tool_calls": tm.get("tool_calls"), "prompt_tokens": tm.get("prompt_tokens")}
                    rows.append(_list_row(f"rollout/{run.name}/{n}/{r.get('group_idx', 0)}/{r.get('traj_idx', 0)}", ginfo.get("task_id") or f"group {r.get('group_idx', 0)}",
                                          f"train:{run.name}@{r.get('sampling_client_step', n)}", f"train:{run.name}", stats, task, r.get("total_reward"),
                                          "[" in ans and ":L" in ans, _mtime(p)))
    return rows


@router.get("/traces")
def list_traces(profile: str | None = None, run: str | None = None, source: str | None = None, type: str | None = None,
                stop: str | None = None, has_citations: bool | None = None, task_id: str | None = None, q: str | None = None,
                limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)) -> dict[str, Any]:
    rows = all_trace_rows()
    facets = {"profile": sorted({r["profile"] for r in rows}), "run": sorted({r["run"] for r in rows}),
              "source": sorted({r["source"] for r in rows if r["source"]}), "task_type": sorted({r["task_type"] for r in rows if r["task_type"]}),
              "stop_reason": sorted({r["stop_reason"] for r in rows if r["stop_reason"]})}
    ql = (q or "").lower()
    out = [r for r in rows
           if (not profile or r["profile"] == profile) and (not run or r["run"] == run) and (not source or r["source"] == source)
           and (not type or r["task_type"] == type) and (not stop or r["stop_reason"] == stop)
           and (has_citations is None or r["has_citations"] == has_citations) and (not task_id or r["task_id"] == task_id)
           and (not ql or ql in r["question"].lower() or ql in r["task_id"].lower())]
    out.sort(key=lambda r: r["mtime"] or 0, reverse=True)
    return {"total": len(out), "facets": facets, "traces": out[offset: offset + limit]}


def trace_detail(trace_id: str) -> dict[str, Any]:
    parts = trace_id.strip("/").split("/")
    kind = parts[0] if parts else ""
    tasks = _task_index()
    if kind == "trace" and len(parts) == 3:
        p = paths.TRACES / parts[1] / f"{parts[2]}.json"
        if not p.exists():
            raise HTTPException(404, f"no trace {trace_id}")
        trace = Trace.model_validate_json(p.read_text())
        task = tasks.get(trace.task_id)
        cits = _citation_table(trace.answer, trace.stats.files_read, task["repo_id"] if task else _repo_from_task_id(trace.task_id))
        grade = {"reward": None, "components": {"citations_parse": float(bool(cits)) if trace.answer else None,
                                                "citations_exist": _all(cits, "exists"), "citations_grounded": _all(cits, "grounded")},
                 "gate_failed": None, "notes": "not graded: citation check only", "source": "check_citations"}
        return {"id": trace_id, "kind": "trace", "run": parts[1], "task_id": trace.task_id, "profile": trace.profile, "task": task,
                "repo_id": task["repo_id"] if task else _repo_from_task_id(trace.task_id), "question": _question(trace, task),
                "events": trace_events(trace), "stats": trace.stats.model_dump(), "answer": trace.answer, "grade": grade, "citations": cits,
                "messages": [m.model_dump() for m in trace.messages]}
    if kind == "eval" and len(parts) == 4:
        prof, set_, tid = parts[1], parts[2], parts[3]
        p = paths.EVALS / prof / set_ / "traces" / f"{tid}.json"
        if not p.exists():
            raise HTTPException(404, f"no trace {trace_id}")
        trace = Trace.model_validate_json(p.read_text())
        task = tasks.get(tid)
        pt = _per_task_rows(prof, set_).get(tid)
        repo_id = task["repo_id"] if task else (pt or {}).get("repo_id") or _repo_from_task_id(tid)
        cits = _citation_table(trace.answer, trace.stats.files_read, repo_id)
        if pt:
            grade = {"reward": pt.get("reward"), "components": pt.get("components", {}), "gate_failed": pt.get("gate_failed"),
                     "notes": pt.get("notes", ""), "source": f"evals/{prof}/{set_}"}
        else:
            grade = {"reward": None, "components": {}, "gate_failed": None, "notes": "not graded", "source": "none"}
        return {"id": trace_id, "kind": "eval", "run": f"eval:{set_}", "task_id": tid, "profile": prof, "task": task, "repo_id": repo_id,
                "question": _question(trace, task), "events": trace_events(trace), "stats": trace.stats.model_dump(), "answer": trace.answer,
                "grade": grade, "citations": cits, "messages": [m.model_dump() for m in trace.messages]}
    if kind == "rollout" and len(parts) == 5:
        run, n, g, t = parts[1], int(parts[2]), int(parts[3]), int(parts[4])
        p = _iteration_path(run, n)
        if not p.exists():
            raise HTTPException(404, f"no trace {trace_id}")
        rollouts = cached(f"iter:{run}:{n}", p, lambda: read_jsonl(p))
        r = next((r for r in rollouts if int(r.get("group_idx", -1)) == g and int(r.get("traj_idx", -1)) == t), None)
        if r is None:
            raise HTTPException(404, f"no trace {trace_id}")
        tm = r.get("trajectory_metrics", {})
        events = rollout_events(r)
        answer = next((e["markdown"] for e in events if e["type"] == "answer"), "")
        tags = r.get("tags", [])
        # The rollout summary does not carry task_id or repo; the question is the first user message if logged.
        stats = next(e for e in events if e["type"] == "stats")
        grade = _grade_from_metrics(tm, notes=f"training rollout, sampled at step {r.get('sampling_client_step', n)}")
        ginfo = _groups_map(run, n).get(g, {})
        task_full = _task_index().get(ginfo.get("task_id", ""))
        return {"id": trace_id, "kind": "rollout", "run": f"train:{run}", "task_id": ginfo.get("task_id") or f"group {g}", "profile": f"train:{run}@{r.get('sampling_client_step', n)}",
                "task": task_full or {"source": tags[0] if tags else None, "task_type": tags[1] if len(tags) > 1 else None}, "repo_id": ginfo.get("repo_id"),
                "question": ginfo.get("question", ""), "events": events, "stats": stats, "answer": answer, "grade": grade, "citations": [],
                "messages": rollout_messages(r), "messages_note": "Training rollout summaries log each turn's generation and tool results, not the system and user prompts; ob_len is the size of the prompt the model saw at that turn."}
    raise HTTPException(404, f"bad trace id {trace_id}")


def _all(cits: list[dict[str, Any]], key: str) -> float | None:
    if not cits:
        return None
    return float(all(c[key] for c in cits))


def _question(trace: Trace, task: dict[str, Any] | None) -> str:
    if task and task.get("question"):
        return task["question"]
    for m in trace.messages:
        if m.role == "user":
            txt = m.content
            i = txt.rfind("Question:")
            return txt[i + len("Question:"):].strip() if i >= 0 else txt[-500:]
    return ""


@lru_cache(maxsize=1)
def _repo_ids() -> list[str]:
    return sorted(p.name for p in paths.REPOS.iterdir() if p.is_dir()) if paths.REPOS.exists() else []


def _adhoc_index() -> dict[str, dict[str, Any]]:
    """Product asks recorded by the API (`data/traces/product/adhoc.jsonl`): task id -> repo, question, type."""
    p = paths.TRACES / "product" / "adhoc.jsonl"
    return cached("adhoc.jsonl", p, lambda: {r["task_id"]: r for r in read_jsonl(p)})


def _repo_from_task_id(task_id: str) -> str:
    rec = _adhoc_index().get(task_id)
    if rec:
        return rec["repo_id"]
    # older adhoc traces without a sidecar row: best effort by repo name in the id
    for rid in _repo_ids():
        if rid.split("__")[1].lower() in task_id.lower():
            return rid
    return ""


@router.get("/traces/compare")
def compare_traces(a: str, b: str) -> dict[str, Any]:
    return {"a": trace_detail(a), "b": trace_detail(b)}


@router.get("/traces/{trace_id:path}")
def get_trace(trace_id: str) -> dict[str, Any]:
    return trace_detail(trace_id)


# ---------------------------------------------------------------------------
# Repo page: what the agent sees, and a console to call its tools by hand.
# ---------------------------------------------------------------------------

from codeqa.agent.tools import RepoTools  # noqa: E402  (agent is core; api may import it)

_tools_cache: dict[str, tuple[float, RepoTools]] = {}


def repo_tools(repo_id: str) -> RepoTools:
    """One RepoTools per repo, rebuilt when its index changes. Call counting is reset per request (see run_tool)."""
    idx = paths.index_dir(repo_id) / "symbols.json"
    m = _mtime(idx)
    hit = _tools_cache.get(repo_id)
    if hit and hit[0] == m:
        return hit[1]
    t = RepoTools(repo_id)
    _tools_cache[repo_id] = (m, t)
    return t


def _tree_map(repo_id: str) -> str:
    """Lean-variant map (no summaries needed); '' when the repo is not indexed yet."""
    try:
        from codeqa.agent.indexing.repomap import tree_map
        return tree_map(repo_id)
    except Exception:  # noqa: BLE001 - overview must not 500 on a half-indexed repo
        return ""


@router.get("/repos/{repo_id}/overview")
def repo_overview(repo_id: str) -> dict[str, Any]:
    mp = paths.repo_dir(repo_id) / "manifest.json"
    if not mp.exists():
        raise HTTPException(404, f"unknown repo {repo_id}")
    m = read_json(mp, {}) or {}
    files = m.get("files", [])
    idx = paths.index_dir(repo_id)
    map_p = idx / "map.txt"
    summ = read_json(idx / "summaries.json", {}) if (idx / "summaries.json").exists() else {}
    langs = Counter(f.get("lang") or "other" for f in files)
    top_dirs = Counter((f["path"].split("/")[0] if "/" in f["path"] else ".") for f in files)
    nsym = len((read_json(idx / "symbols.json", {}) or {}).get("symbols", [])) if (idx / "symbols.json").exists() else None
    return {
        "repo_id": repo_id,
        "url": m.get("url"),
        "sha": m.get("sha"),
        "files": [{"path": f["path"], "lines": f.get("lines", 0), "lang": f.get("lang")} for f in files],
        "n_files": len(files),
        "lines": sum(int(f.get("lines", 0)) for f in files),
        "symbols": nsym,
        "dropped": m.get("dropped", {}),
        "languages": dict(langs.most_common(12)),
        "top_dirs": dict(top_dirs.most_common(20)),
        "map": map_p.read_text() if map_p.exists() else _tree_map(repo_id),
        "map_lines": len(map_p.read_text().splitlines()) if map_p.exists() else 0,
        "n_summaries": len(summ) if isinstance(summ, dict) else 0,
        "nodoc": repo_id.endswith("__nodoc"),
        "has_nodoc_variant": (paths.repo_dir(repo_id + "__nodoc") / "manifest.json").exists(),
    }


@router.get("/repos/{repo_id}/tools")
def repo_tool_specs(repo_id: str, variant: str | None = None) -> dict[str, Any]:
    """The tools of one agent variant (default: the default variant), so the console can drive the bash harness too."""
    if not (paths.repo_dir(repo_id) / "manifest.json").exists():
        raise HTTPException(404, f"unknown repo {repo_id}")
    t = repo_tools(repo_id)
    caps = t.caps
    from codeqa.agent.variants import VARIANTS, resolve as resolve_variant
    try:
        v = resolve_variant(variant or None)
    except KeyError:
        raise HTTPException(400, f"unknown variant {variant!r}; one of {', '.join(VARIANTS)}")
    return {"tools": t.specs(v.tools), "variant": v.name, "variants": list(VARIANTS),
            "caps": {k: getattr(caps, k) for k in dir(caps) if not k.startswith("_") and isinstance(getattr(caps, k), (int, float))}}


from pydantic import BaseModel as _BaseModel  # noqa: E402


class ToolCallBody(_BaseModel):
    name: str
    args: dict[str, Any] = {}
    variant: str | None = None   # which agent's tool set the name is resolved in (bash lives only in the bash variants)


@router.post("/repos/{repo_id}/tool")
async def run_tool(repo_id: str, body: ToolCallBody) -> dict[str, Any]:
    """Run one agent tool exactly as the driver would (same caps, same error text) and return what the agent would see."""
    from tinker_cookbook.tool_use.types import ToolInput

    if not (paths.repo_dir(repo_id) / "manifest.json").exists():
        raise HTTPException(404, f"unknown repo {repo_id}")
    t = repo_tools(repo_id)
    from codeqa.agent.variants import resolve as resolve_variant
    tools = {x.name: x for x in t.tools(resolve_variant(body.variant or None).tools)}
    if body.name not in tools:
        raise HTTPException(400, f"unknown tool {body.name!r}; one of {', '.join(tools)}")
    # The console is stateless: no call budget, and files_read is per call.
    t.max_tool_calls = None
    before = len(t.files_read)
    t0 = time.time()
    result = await tools[body.name].run(ToolInput(arguments=body.args, call_id=f"console_{int(t0)}"))
    content = result.messages[0]["content"]
    text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content) if isinstance(content, list) else str(content)
    spans = [s.model_dump() for s in t.files_read[before:]]
    return {"name": body.name, "args": body.args, "output": text, "chars": len(text), "seconds": round(time.time() - t0, 3),
            "error": text.startswith("ERROR"), "files_read": spans}
