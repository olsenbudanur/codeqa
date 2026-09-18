"""B6a: dedupe + base pass-rate measurement.

For every raw training task, sample the untrained student N times through the real RepoEnv and record, per task:
format_ok rate, citations_grounded rate, ungated correctness (verifier or judge on the answer text, gates ignored),
correctness given format, mean reward, tool calls. Written incrementally to data/tasks/reports/passrate.jsonl so the
run is resumable and the split step can be re-run without sampling again.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from codeqa.agent.driver import run_episode
from codeqa.agent.env import RepoEnv
from codeqa.clients.base import make_client
from codeqa.datagen import llm
from codeqa.grader import gates
from codeqa.grader.citations import check_citations
from codeqa.grader.grade import grade
from codeqa.grader.judge import judge
from codeqa.grader.repo import load_repo
from codeqa.grader.verifiers import uses_judge, verify
from codeqa.shared import paths
from codeqa.shared.contracts import Task
from codeqa.shared.jsonl import read_all
from codeqa.shared.profiles import get_profile

REPORTS = paths.TASKS / "reports"
PASSRATE = REPORTS / "passrate.jsonl"
RAW_SOURCES = ("deepcodebench", "codescout", "structural", "teacher")
EPISODE_TIMEOUT = 240.0


def _log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


# ------------------------------------------------------------------------------------------------------ dedupe
_WORD = re.compile(r"[a-z0-9_]{3,}")
_CODE = re.compile(r"`([^`]+)`|\b([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+)\b")


def _tokens(q: str) -> set[str]:
    """Words plus every backticked / dotted identifier kept whole, so templated questions that differ only in the
    identifier (`pkg.a` vs `pkg.b`) are not near-duplicates."""
    toks = set(_WORD.findall(q.lower()))
    for m in _CODE.finditer(q):
        toks.add(("`" + (m.group(1) or m.group(2))).lower())
    return toks


def dedupe(tasks: list[Task], threshold: float = 0.8) -> tuple[list[Task], list[tuple[str, str]]]:
    """Drop a task whose question token-set Jaccard with an earlier task in the same repo is >= threshold.
    Returns (kept, [(dropped_id, kept_id)])."""
    kept: list[Task] = []
    dropped: list[tuple[str, str]] = []
    by_repo: dict[str, list[tuple[Task, set[str]]]] = defaultdict(list)
    for t in tasks:
        toks = _tokens(t.question)
        dup = None
        for other, otoks in by_repo[t.repo_id]:
            inter = len(toks & otoks)
            if inter and inter / len(toks | otoks) >= threshold:
                dup = other; break
        if dup is not None:
            dropped.append((t.task_id, dup.task_id))
        else:
            kept.append(t); by_repo[t.repo_id].append((t, toks))
    return kept, dropped


def load_raw(sources: tuple[str, ...] = RAW_SOURCES) -> list[Task]:
    out: list[Task] = []
    for s in sources:
        p = paths.TASKS_RAW / f"{s}.jsonl"
        if p.exists():
            out.extend(read_all(p, Task))
    return out


# ------------------------------------------------------------------------------------------------------ sampling
async def sample_task(task: Task, client, judge_client, n: int) -> dict[str, Any]:
    repo = load_repo(task.repo_id)
    budget = task.effective_budget()
    rows: list[dict[str, Any]] = []
    for i in range(n):
        env = RepoEnv(task, client.profile)
        try:
            trace = await asyncio.wait_for(run_episode(env, client, temperature=1.0), EPISODE_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            rows.append({"error": f"{type(e).__name__}: {str(e)[:100]}"}); continue
        answer = gates.extract_answer(trace)
        answered = bool(answer.strip())
        fmt_ok, _ = gates.format_gate(answer, trace, budget)
        report = check_citations(answer, trace.stats.files_read, task.repo_id, task.grading.expected_symbols, repo=repo)
        grounded = bool(report.citations) and report.all_exist and report.all_grounded
        found = found_gold(task, trace.stats.files_read)
        judged = uses_judge(task)
        gate_ok = fmt_ok and bool(report.citations) and report.all_exist and report.all_grounded and trace.stats.tool_calls <= budget.max_tool_calls
        if not answered:
            strict = lenient = 0.0
            reward, gate = 0.0, "format"
        elif judged:
            # one judge call per sample (grade() would call it again): reward = verdict when every gate passes
            v = await judge(task.question, answer, task.grading.rubric, task.grading.reference_answer, budget.max_answer_tokens, client=judge_client)
            strict = lenient = v.score
            reward, gate = (v.score if gate_ok else 0.0), (None if gate_ok else "gates")
        else:
            strict, _ = verify(task, answer, report, repo)
            lenient = lenient_correctness(task, answer)
            full = await grade(task, trace, judge_client=judge_client, repo=repo)
            reward, gate = full.reward, full.gate_failed
        rows.append({"answered": answered, "format_ok": fmt_ok, "grounded": grounded, "found": found, "correct": strict, "lenient": lenient,
                     "reward": reward, "gate": gate, "tool_calls": trace.stats.tool_calls, "stop": trace.stats.stop_reason,
                     "answer_chars": len(answer)})
    ok = [r for r in rows if "error" not in r]
    def mean(key, subset=None):
        vals = [r[key] for r in (subset if subset is not None else ok) if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return round(sum(vals) / len(vals), 3) if vals else None
    answered_rows = [r for r in ok if r["answered"]]
    fmt_rows = [r for r in ok if r["format_ok"]]
    return {
        "task_id": task.task_id, "source": task.source, "task_type": task.task_type, "repo_id": task.repo_id, "n": len(ok),
        "errors": len(rows) - len(ok),
        "answered": mean("answered"), "n_answered": len(answered_rows),
        "format_ok": mean("format_ok"), "n_format_ok": len(fmt_rows),
        "grounded": mean("grounded"), "found": mean("found"),
        "correct": mean("correct"), "correct_given_format": mean("correct", fmt_rows),
        "correct_lenient": mean("lenient", answered_rows), "correct_lenient_all": mean("lenient"),
        "reward": mean("reward"), "tool_calls": mean("tool_calls"),
        "stops": dict(Counter(r["stop"] for r in ok)),
    }


def lenient_correctness(task: Task, answer: str) -> float:
    """Content-only correctness: ignores citations entirely. Literal match; gold symbol names mentioned; gold paths
    (or basenames) mentioned; judged types are handled by the judge (already text-only). Any-of for a single gold."""
    g = task.grading
    a = answer or ""
    if g.expected_literal is not None:
        from codeqa.grader.verifiers import literal_score
        return literal_score(a, g.expected_literal)
    if g.expected_symbols:
        names = [q.split(":", 1)[-1].split(".")[-1] for q in g.expected_symbols]
        hit = [bool(re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w])", a)) for n in names]
        return float(any(hit)) if len(hit) == 1 else sum(hit) / len(hit)
    if g.expected_paths:
        hit = [(p in a) or (p.rsplit("/", 1)[-1] in a) for p in g.expected_paths]
        return float(any(hit)) if len(hit) == 1 else sum(hit) / len(hit)
    return 0.0


def found_gold(task: Task, files_read) -> bool:
    """Did the episode read a required span (or, without spans, a gold file)? Retrievability regardless of answering."""
    g = task.grading
    if g.required_citations:
        for c in g.required_citations:
            for s in files_read:
                if s.path == c.path and s.start <= c.end and c.start <= s.end:
                    return True
        return False
    return any(s.path in g.expected_paths for s in files_read)


def difficulty_score(rec: dict[str, Any]) -> float | None:
    """Lenient correctness over answered samples when >= 2 answered; otherwise the found-gold rate. The untrained
    student often finds the right lines and then runs out of turns without answering; that is a behaviour RL fixes
    in a few steps, not a property of the task, so it must not push every task into the hard reserve."""
    if rec.get("n_answered", 0) >= 2 and rec.get("correct_lenient") is not None:
        return rec["correct_lenient"]
    return rec.get("found")


def merge_records(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Combine two measurements of the same task (sample-weighted means; counts added)."""
    na, nb = a.get("n", 0), b.get("n", 0)
    if not na:
        return b
    if not nb:
        return a
    out = dict(a)
    out["n"] = na + nb
    out["errors"] = a.get("errors", 0) + b.get("errors", 0)
    for k in ("answered", "format_ok", "grounded", "found", "correct", "correct_lenient_all", "reward", "tool_calls"):
        va, vb = a.get(k), b.get(k)
        out[k] = round((va * na + vb * nb) / (na + nb), 3) if va is not None and vb is not None else (va if va is not None else vb)
    for k, nk in (("correct_given_format", "n_format_ok"), ("correct_lenient", "n_answered")):
        wa, wb = a.get(nk, 0), b.get(nk, 0)
        va, vb = a.get(k), b.get(k)
        out[nk] = wa + wb
        out[k] = round((va * wa + vb * wb) / (wa + wb), 3) if wa and wb and va is not None and vb is not None else (va if wa else vb)
    st = Counter(a.get("stops", {})); st.update(b.get("stops", {}))
    out["stops"] = dict(st)
    return out


def load_passrate() -> dict[str, dict[str, Any]]:
    """Latest record per task; several processes append, and refine passes append merged records."""
    done: dict[str, dict[str, Any]] = {}
    if PASSRATE.exists():
        for line in PASSRATE.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["task_id"]] = r
    return done


async def measure(profile_name: str = "qwen4b-base", samples: int = 4, concurrency: int = 16, limit: int | None = None,
                  sources: tuple[str, ...] = RAW_SOURCES, judge_model: str = "claude-haiku-4-5", refine: bool = False,
                  target_n: int = 4, shard: tuple[int, int] | None = None) -> dict[str, Any]:
    t0 = time.time()
    raw = load_raw(sources)
    tasks, dropped = dedupe(raw)
    _log(f"  filter: {len(raw)} raw tasks, {len(dropped)} near-duplicates dropped, {len(tasks)} to measure")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "dedupe.json").write_text(json.dumps({"dropped": dropped}, indent=1))
    done = load_passrate()
    if refine:
        # second pass on the extremes only: a two-sample 0 or 1 is too coarse to place a task; bring them to target_n
        def extreme(r):
            d = difficulty_score(r)
            return r.get("n", 0) < target_n and (d is None or d <= 0.0 or d >= 1.0)
        todo = [t for t in tasks if t.task_id in done and extreme(done[t.task_id])]
        samples = max(1, min(samples, target_n - min(done[t.task_id].get("n", 0) for t in todo))) if todo else samples
        _log(f"  filter(refine): {len(done)} measured, {len(todo)} at an extreme with n < {target_n}; adding {samples} samples each")
    else:
        todo = [t for t in tasks if t.task_id not in done]
        _log(f"  filter: {len(done)} already measured, {len(todo)} to run x {samples} samples, concurrency {concurrency}")
    if shard:
        i, n = shard
        todo = todo[i::n]                      # several processes = several Tinker sessions; each is latency-bound
        _log(f"  filter: shard {i}/{n} -> {len(todo)} tasks")
    if limit:
        todo = todo[:limit]
    client = make_client(get_profile(profile_name))
    judge_client = llm.AnthropicClient(llm.HAIKU.model_copy(update={"name": "haiku-judge", "model": judge_model, "max_generation_tokens": 1024}))
    sem = asyncio.Semaphore(concurrency)
    f = PASSRATE.open("a")
    n_done = 0

    async def one(task: Task):
        nonlocal n_done
        async with sem:
            try:
                rec = await sample_task(task, client, judge_client, samples)
            except Exception as e:  # noqa: BLE001
                rec = {"task_id": task.task_id, "source": task.source, "task_type": task.task_type, "repo_id": task.repo_id, "n": 0,
                       "errors": samples, "fatal": f"{type(e).__name__}: {str(e)[:120]}"}
            if refine and task.task_id in done:
                rec = merge_records(done[task.task_id], rec)
            f.write(json.dumps(rec) + "\n"); f.flush()
            done[task.task_id] = rec
            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                measured = [r for r in done.values() if r.get("n")]
                def m(key):
                    vals = [r[key] for r in measured if r.get(key) is not None]
                    return sum(vals) / len(vals) if vals else 0.0
                _log(f"  filter: {n_done}/{len(todo)} tasks, {time.time()-t0:.0f}s ({(time.time()-t0)/max(n_done,1):.1f}s/task); "
                     f"running means answered={m('answered'):.2f} format_ok={m('format_ok'):.2f} found={m('found'):.2f} "
                     f"lenient={m('correct_lenient_all'):.2f} strict={m('correct'):.2f}")

    await asyncio.gather(*(one(t) for t in todo))
    f.close()
    return summarize(done, time.time() - t0)


def summarize(done: dict[str, dict[str, Any]], seconds: float = 0.0) -> dict[str, Any]:
    measured = [r for r in done.values() if r.get("n")]
    by_src: dict[str, dict[str, float]] = {}
    for src in sorted({r["source"] for r in measured}):
        rs = [r for r in measured if r["source"] == src]
        def m(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None
        by_src[src] = {"tasks": len(rs), "answered": m("answered"), "format_ok": m("format_ok"), "grounded": m("grounded"),
                       "found": m("found"), "correct_strict": m("correct"), "correct_lenient_all": m("correct_lenient_all"),
                       "reward": m("reward"), "tool_calls": m("tool_calls")}
    rep = {"measured": len(measured), "by_source": by_src, "seconds": round(seconds, 1)}
    (REPORTS / "passrate_summary.json").write_text(json.dumps(rep, indent=2))
    return rep


def main(**kw) -> dict[str, Any]:
    return asyncio.run(measure(**kw))
