"""B5 runner: seeds -> Sonnet authors -> Haiku answers blind through the real RepoEnv -> judge agreement -> keep.

Writes data/tasks/raw/teacher.jsonl and data/tasks/reports/teacher_attempts.jsonl (one line per attempt with cost,
tool calls, blind score and the reason if rejected). Resumable: attempts already logged are skipped.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from codeqa.agent.driver import run_episode
from codeqa.agent.env import RepoEnv
from codeqa.clients.anthropic import AnthropicClient
from codeqa.datagen import generate, llm
from codeqa.datagen.repos import ensure_repo
from codeqa.datagen.sources import teacher
from codeqa.datagen.sources.structural import _is_skipped
from codeqa.grader.judge import judge
from codeqa.shared import paths
from codeqa.shared.contracts import Grading, IndexSymbol, Manifest, Task
from codeqa.shared.jsonl import write

REPORTS = paths.TASKS / "reports"
ATTEMPTS = REPORTS / "teacher_attempts.jsonl"
KEEP_THRESHOLD = 0.5          # blind Haiku must satisfy at least half the rubric for the task to count as well-posed
PRICES = {"claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0)}   # $/M in, out


def _log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def pick_seeds(manifest: Manifest, symbols: list[IndexSymbol], n: int, rng: random.Random) -> list[tuple[str, list[str]]]:
    """Source files (no tests/docs) with >= 3 symbols, spread across directories, biased to mid-sized files."""
    by_path: dict[str, list[IndexSymbol]] = defaultdict(list)
    for s in symbols:
        by_path[s.path].append(s)
    lines = {f.path: f.lines for f in manifest.files}
    cands = [(p, syms) for p, syms in by_path.items() if not _is_skipped(p) and len(syms) >= 3 and 80 <= lines.get(p, 0) <= 2500]
    rng.shuffle(cands)
    by_dir: dict[str, list] = defaultdict(list)
    for p, syms in cands:
        by_dir[p.rsplit("/", 1)[0] if "/" in p else "."].append((p, syms))
    out: list[tuple[str, list[str]]] = []
    dirs = list(by_dir)
    rng.shuffle(dirs)
    while len(out) < n and dirs:
        for d in list(dirs):
            if not by_dir[d]:
                dirs.remove(d); continue
            p, syms = by_dir[d].pop()
            names = [f"{s.parent}.{s.name}" if s.parent else s.name for s in syms if not s.name.startswith("_")]
            out.append((p, names))
            if len(out) >= n:
                break
    return out


def _attempt_key(repo_id: str, seed: str) -> str:
    return hashlib.sha1(f"{repo_id}|{seed}".encode()).hexdigest()[:12]


def _cost(model: str, ptoks: int, ctoks: int) -> float:
    i, o = PRICES.get(model, (2.0, 10.0))
    return round((ptoks * i + ctoks * o) / 1e6, 4)


async def one_attempt(repo_id: str, seed: str, seed_syms: list[str], author_client: AnthropicClient,
                      blind_client: AnthropicClient, judge_client) -> tuple[Task | None, dict[str, Any]]:
    t0 = time.time()
    rec: dict[str, Any] = {"key": _attempt_key(repo_id, seed), "repo_id": repo_id, "seed": seed}
    try:
        authored, reason = await teacher.author(author_client, repo_id, seed, seed_syms)
    except Exception as e:  # noqa: BLE001
        authored, reason = None, f"author_error:{type(e).__name__}"
    rec["author_reason"] = reason
    if authored is None:
        rec.update(status="rejected", seconds=round(time.time() - t0, 1))
        return None, rec
    rec.update(author_tool_calls=authored.tool_calls, author_cost=_cost(author_client.profile.model, authored.prompt_tokens, authored.completion_tokens),
               question=authored.question, task_type=authored.task_type)
    task = Task(task_id=f"tch-{rec['key'][:8]}", repo_id=repo_id, split="train", question=authored.question, task_type=authored.task_type,  # type: ignore[arg-type]
                source="teacher", source_id=f"seed:{seed}",
                grading=Grading(expected_paths=sorted({e.path for e in authored.evidence}), reference_answer=authored.answer,
                                rubric=authored.rubric, required_citations=authored.evidence))
    # blind answer through the real environment (same prompt, tools, caps the student gets)
    try:
        env = RepoEnv(task, blind_client.profile)
        trace = await run_episode(env, blind_client)
        rec.update(blind_stop=trace.stats.stop_reason, blind_tool_calls=trace.stats.tool_calls,
                   blind_cost=_cost(blind_client.profile.model, trace.stats.prompt_tokens, trace.stats.completion_tokens))
        if not trace.answer:
            rec.update(status="rejected", blind_score=None, reject="blind_no_answer", seconds=round(time.time() - t0, 1))
            return None, rec
        verdict = await judge(task.question, trace.answer, task.grading.rubric, None, task.effective_budget().max_answer_tokens, client=judge_client)
    except Exception as e:  # noqa: BLE001
        rec.update(status="rejected", reject=f"blind_error:{type(e).__name__}", seconds=round(time.time() - t0, 1))
        return None, rec
    score = verdict.score
    rec["blind_score"] = None if score != score else round(score, 3)
    rec["seconds"] = round(time.time() - t0, 1)
    if score != score:
        rec.update(status="rejected", reject="judge_failed")
        return None, rec
    if score < KEEP_THRESHOLD:
        rec.update(status="rejected", reject="blind_disagrees")
        return None, rec
    rec["status"] = "kept"
    rec["task"] = task.model_dump(mode="json")
    return task, rec


async def teach_all(seeds_per_repo: int = 30, concurrency: int = 6, only: list[str] | None = None, limit: int | None = None,
                    max_cost_usd: float | None = None) -> dict[str, Any]:
    t0 = time.time()
    repos = generate.training_repos()
    if only:
        repos = [r for r in repos if r[1] in only or f"{r[0]}/{r[1]}" in only]
    author_client = AnthropicClient(llm.SONNET)
    blind_client = AnthropicClient(llm.HAIKU.model_copy(update={"name": "haiku-blind", "max_generation_tokens": 1500}))
    judge_client = AnthropicClient(llm.HAIKU.model_copy(update={"name": "haiku-judge", "max_generation_tokens": 1024}))
    REPORTS.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if ATTEMPTS.exists():
        for line in ATTEMPTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["key"]] = r
    jobs: list[tuple[str, str, list[str]]] = []
    for owner, repo, sha in repos:
        m, syms = ensure_repo(owner, repo, sha)
        if not (paths.index_dir(m.repo_id) / "map.txt").exists():
            _log(f"  teacher: skipping {m.repo_id}: no map.txt yet"); continue
        for seed, names in pick_seeds(m, syms, seeds_per_repo, random.Random(m.repo_id)):
            if _attempt_key(m.repo_id, seed) not in done:
                jobs.append((m.repo_id, seed, names))
    if limit:
        jobs = jobs[:limit]
    _log(f"  teacher: {len(jobs)} attempts to run ({len(done)} already logged), concurrency {concurrency}")
    sem = asyncio.Semaphore(concurrency)
    f = ATTEMPTS.open("a")
    n_done = 0
    spent = 0.0
    skipped_for_cost = 0

    async def run(job):
        nonlocal n_done, spent, skipped_for_cost
        async with sem:
            if max_cost_usd is not None and spent >= max_cost_usd:
                skipped_for_cost += 1
                return None
            task, rec = await one_attempt(*job, author_client, blind_client, judge_client)
            spent += (rec.get("author_cost") or 0) + (rec.get("blind_cost") or 0) + 0.005   # + judge estimate
            f.write(json.dumps(rec) + "\n"); f.flush()
            done[rec["key"]] = rec
            n_done += 1
            if n_done % 5 == 0 or n_done == len(jobs):
                kept = sum(1 for r in done.values() if r.get("status") == "kept")
                _log(f"  teacher: {n_done}/{len(jobs)} attempts, {kept} kept total, ${spent:.2f} this run, {time.time()-t0:.0f}s")
            return task

    await asyncio.gather(*(run(j) for j in jobs))
    f.close()
    # rebuild the task file from every kept attempt (this run and earlier ones)
    tasks = rebuild_tasks(done)
    out = paths.TASKS_RAW / "teacher.jsonl"
    n = write(out, tasks)
    recs = list(done.values())
    report = {"source": "teacher", "records": n, "file": str(out.relative_to(paths.ROOT)),
              "attempts": len(recs), "kept": sum(1 for r in recs if r.get("status") == "kept"),
              "rejections": dict(Counter((r.get("reject") or r.get("author_reason")) for r in recs if r.get("status") != "kept")),
              "types": dict(Counter(t.task_type for t in tasks)),
              "cost_usd": round(sum((r.get("author_cost") or 0) + (r.get("blind_cost") or 0) for r in recs), 2),
              "cost_per_kept_usd": round(sum((r.get("author_cost") or 0) + (r.get("blind_cost") or 0) for r in recs) / max(n, 1), 3),
              "mean_blind_score_kept": round(sum(r["blind_score"] for r in recs if r.get("status") == "kept") / max(n, 1), 3),
              "per_repo": dict(Counter(t.repo_id for t in tasks)), "seconds": round(time.time() - t0, 1),
              "this_run_cost_usd": round(spent, 2), "skipped_for_cost": skipped_for_cost}
    (REPORTS / "teacher.json").write_text(json.dumps(report, indent=2))
    _log(f"  teacher: {n} tasks from {len(recs)} attempts; rejections {report['rejections']}; ${report['cost_usd']} total, ${report['cost_per_kept_usd']}/kept")
    return report


def rebuild_tasks(done: dict[str, dict[str, Any]]) -> list[Task]:
    """Tasks are persisted inside the attempt log (full record) so the JSONL can be rebuilt without re-running."""
    tasks = []
    for r in done.values():
        if r.get("status") == "kept" and r.get("task"):
            tasks.append(Task.model_validate(r["task"]))
    return tasks


def main(seeds_per_repo: int = 30, concurrency: int = 6, only: list[str] | None = None, limit: int | None = None,
         max_cost_usd: float | None = None) -> dict[str, Any]:
    return asyncio.run(teach_all(seeds_per_repo, concurrency, only, limit, max_cost_usd))
