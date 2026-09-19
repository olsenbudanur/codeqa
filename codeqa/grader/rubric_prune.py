"""Rubric hygiene: drop rubric items the question does not ask for (decisions.md 2026-09-19 "prune").

DeepCodeBench facts were copied verbatim into rubrics, so 10 % of items are true facts of the source file that no
correct answer needs to state (frontier audit, LOG 23:10). Two modes:
  --apply-empirical FILES --traces DIRS   keep an item iff some frontier answer (Sonnet/Opus traces) states it. This is the rule
                                          that reproduces the audit (both-miss items); tasks without a frontier answer are untouched.
  --apply FILES                           one Sonnet call per task marks items required/incidental from the question alone.
                                          Validated poorly (47 % of noise caught, 32 % of real items dropped): kept for reference only.
A task keeps at least MIN_KEEP items (if fewer would remain, nothing is pruned and the task is flagged).

  uv run python -m codeqa.grader.rubric_prune --validate                      # score the rule against reports/rubric_audit_frontier.json
  uv run python -m codeqa.grader.rubric_prune --apply data/tasks/eval/fast.jsonl data/tasks/train/all.jsonl ...
Originals are copied to data/tasks/backup/<name>.<timestamp>.jsonl; a report goes to data/tasks/reports/rubric_prune.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from codeqa.clients.anthropic import AnthropicClient
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Message, Task
from codeqa.shared.jsonl import read_all, write

MIN_KEEP = 2
MODEL = "claude-sonnet-5"

PROMPT = """You are cleaning a grading rubric for a question about a code repository.

The rubric items below were extracted from the source file. Some are facts a correct and complete answer to THIS question must state. Others are true facts about the code that the question does not ask for (implementation details, unrelated defaults, where a helper lives, what a test is named) and should not count against an answer that omits them.

Question:
{question}
{reference}
Rubric items:
{items}

For each item decide: "required" if any correct, complete answer to the question would have to state it (or something equivalent), or "incidental" if a fully correct answer could reasonably omit it. Be strict about "required": when in doubt, the item is incidental.
Return JSON only: {{"items": [{{"id": 1, "verdict": "required"}}, ...]}} with one entry per item, in order."""


def _client() -> AnthropicClient:
    return AnthropicClient(EndpointProfile(name="rubric-prune", kind="anthropic", model=MODEL, max_generation_tokens=1024, thinking=False))


async def classify(task: Task, client: AnthropicClient, retries: int = 3) -> list[bool] | None:
    """True = required, per rubric item. None on persistent failure."""
    items = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(task.grading.rubric))
    ref = f"\nReference answer (for context):\n{task.grading.reference_answer}\n" if task.grading.reference_answer else ""
    msg = Message(role="user", content=PROMPT.format(question=task.question, reference=ref, items=items))
    for attempt in range(retries):
        try:
            reply = await asyncio.wait_for(client.chat([msg], max_tokens=1024), timeout=60)
            m = re.search(r"\{.*\}", reply.content, re.DOTALL)
            data = json.loads(m.group(0))
            verdicts = [str(it.get("verdict", "")).lower().startswith("req") for it in data["items"]]
            if len(verdicts) == len(task.grading.rubric):
                return verdicts
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(2 ** attempt)
    return None


async def classify_many(tasks: list[Task], concurrency: int = 8) -> dict[str, list[bool] | None]:
    client = _client()
    sem = asyncio.Semaphore(concurrency)

    async def one(t: Task):
        async with sem:
            return t.task_id, await classify(t, client)
    return dict(await asyncio.gather(*(one(t) for t in tasks)))


def prune_task(task: Task, required: list[bool]) -> tuple[Task, list[str]]:
    kept = [it for it, r in zip(task.grading.rubric, required) if r]
    dropped = [it for it, r in zip(task.grading.rubric, required) if not r]
    if len(kept) < MIN_KEEP:
        return task, []                      # too aggressive for this task: leave it alone, flag in the report
    t = task.model_copy(deep=True)
    t.grading.rubric = kept
    return t, dropped


async def validate(audit_path: Path, tasks_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text())
    tasks = {t.task_id: t for t in read_all(tasks_path, Task)}
    ids = sorted({tid for key in ("both_fail", "opus_only", "sonnet_only") for tid, _ in audit[key]} | {tid for tid in tasks if tasks[tid].grading.rubric})
    ids = [i for i in ids if i in tasks and tasks[i].grading.rubric]
    verdicts = await classify_many([tasks[i] for i in ids])
    both_fail = {(tid, it) for tid, it in audit["both_fail"]}
    stated = set()
    for tid in ids:
        for it in tasks[tid].grading.rubric:
            if (tid, it) not in both_fail and not any((tid, it) == (a, b) for a, b in audit["opus_only"]) and not any((tid, it) == (a, b) for a, b in audit["sonnet_only"]):
                stated.add((tid, it))
    pruned_noise = sum(1 for tid, it in both_fail if verdicts.get(tid) and not verdicts[tid][tasks[tid].grading.rubric.index(it)])
    kept_signal = sum(1 for tid, it in stated if verdicts.get(tid) and verdicts[tid][tasks[tid].grading.rubric.index(it)])
    out = {"tasks": len(ids), "noise_items": len(both_fail), "noise_pruned": pruned_noise, "signal_items": len(stated), "signal_kept": kept_signal,
           "failed_calls": sum(1 for v in verdicts.values() if v is None)}
    print(f"validation: noise items pruned {pruned_noise}/{len(both_fail)} ({pruned_noise / max(len(both_fail), 1):.0%}); "
          f"items both models state kept {kept_signal}/{len(stated)} ({kept_signal / max(len(stated), 1):.0%}); failed calls {out['failed_calls']}", flush=True)
    return out


async def apply(files: list[Path], concurrency: int) -> dict[str, Any]:
    report: dict[str, Any] = {"files": {}, "model": MODEL, "min_keep": MIN_KEEP, "when": time.strftime("%Y-%m-%d %H:%M")}
    cache: dict[str, list[bool] | None] = {}
    for f in files:
        tasks = read_all(f, Task)
        todo = [t for t in tasks if t.grading.rubric and t.task_id not in cache]
        print(f"{f}: {len(tasks)} tasks, {len(todo)} judged tasks to classify", flush=True)
        cache.update(await classify_many(todo, concurrency))
        out, dropped_total, items_before, flagged, failed = [], 0, 0, [], 0
        for t in tasks:
            if not t.grading.rubric:
                out.append(t); continue
            items_before += len(t.grading.rubric)
            v = cache.get(t.task_id)
            if v is None:
                failed += 1; out.append(t); continue
            nt, dropped = prune_task(t, v)
            if not dropped and sum(1 for x in v if not x) > 0:
                flagged.append(t.task_id)
            dropped_total += len(dropped)
            out.append(nt)
        backup = paths.TASKS / "backup" / f"{f.stem}.{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, backup)
        write(f, out)
        report["files"][str(f)] = {"tasks": len(tasks), "rubric_items_before": items_before, "items_dropped": dropped_total,
                                   "tasks_left_unpruned_min_keep": flagged, "classify_failures": failed, "backup": str(backup)}
        print(f"  dropped {dropped_total}/{items_before} items ({dropped_total / max(items_before, 1):.0%}); {len(flagged)} tasks left unpruned (min_keep); backup {backup}", flush=True)
    rp = paths.TASKS / "reports" / "rubric_prune.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    return report


async def empirical_verdicts(tasks: list[Task], trace_dirs: list[Path], concurrency: int = 8) -> dict[str, list[bool] | None]:
    """The rule that matched the frontier audit: an item is kept if ANY available frontier answer states it (judge, rubric
    mode); an item stated by no answer is incidental. Tasks with no answered trace get None (left alone)."""
    from codeqa.grader.gates import extract_answer
    from codeqa.grader.judge import default_client, judge
    from codeqa.shared.contracts import Trace
    client = default_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(t: Task):
        answers = []
        for d in trace_dirs:
            f = d / f"{t.task_id}.json"
            if f.exists():
                a = extract_answer(Trace.model_validate_json(f.read_text()))
                if a:
                    answers.append(a)
        if not answers:
            return t.task_id, None
        stated = [False] * len(t.grading.rubric)
        for a in answers:
            async with sem:
                v = await judge(t.question, a, t.grading.rubric, None, 100000, client=client)
            if v.failed:
                continue
            stated = [x or y for x, y in zip(stated, v.satisfied)]
        return t.task_id, stated
    return dict(await asyncio.gather(*(one(t) for t in tasks)))


async def apply_empirical(files: list[Path], trace_dirs: list[Path], concurrency: int) -> dict[str, Any]:
    report: dict[str, Any] = {"files": {}, "rule": "keep items stated by any frontier answer", "traces": [str(d) for d in trace_dirs],
                              "min_keep": MIN_KEEP, "when": time.strftime("%Y-%m-%d %H:%M")}
    for f in files:
        tasks = read_all(f, Task)
        judged = [t for t in tasks if t.grading.rubric]
        verdicts = await empirical_verdicts(judged, trace_dirs, concurrency)
        out, dropped_total, items_before, flagged, skipped = [], 0, 0, [], 0
        for t in tasks:
            if not t.grading.rubric:
                out.append(t); continue
            items_before += len(t.grading.rubric)
            v = verdicts.get(t.task_id)
            if v is None:
                skipped += 1; out.append(t); continue
            nt, dropped = prune_task(t, v)
            if not dropped and any(not x for x in v):
                flagged.append(t.task_id)
            dropped_total += len(dropped)
            out.append(nt)
        backup = paths.TASKS / "backup" / f"{f.stem}.{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, backup)
        write(f, out)
        report["files"][str(f)] = {"tasks": len(tasks), "judged": len(judged), "no_trace": skipped, "rubric_items_before": items_before,
                                   "items_dropped": dropped_total, "tasks_left_unpruned_min_keep": flagged, "backup": str(backup)}
        print(f"{f}: judged {len(judged)}, without a frontier answer {skipped}; dropped {dropped_total}/{items_before} items "
              f"({dropped_total / max(items_before, 1):.0%}); {len(flagged)} tasks left unpruned (min_keep); backup {backup}", flush=True)
    rp = paths.TASKS / "reports" / "rubric_prune.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(rp.read_text()) if rp.exists() else {}
    prev.setdefault("runs", []).append(report)
    rp.write_text(json.dumps(prev, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true", help="score the rule against data/tasks/reports/rubric_audit_frontier.json (no writes)")
    ap.add_argument("--audit", default=str(paths.TASKS / "reports" / "rubric_audit_frontier.json"))
    ap.add_argument("--audit-tasks", default=str(paths.TASKS_EVAL / "sonnet_hard.jsonl"))
    ap.add_argument("--apply", nargs="*", default=None, help="task files to prune in place (backups under data/tasks/backup/)")
    ap.add_argument("--apply-empirical", nargs="*", default=None, help="task files to prune using frontier traces (--traces)")
    ap.add_argument("--traces", nargs="*", default=[], help="trace dirs with frontier answers (data/evals/<profile>/<set>/traces)")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args(argv)
    if a.apply_empirical:
        asyncio.run(apply_empirical([Path(p) for p in a.apply_empirical], [Path(d) for d in a.traces], a.concurrency))
    if a.validate:
        asyncio.run(validate(Path(a.audit), Path(a.audit_tasks)))
    if a.apply:
        asyncio.run(apply([Path(p) for p in a.apply], a.concurrency))
    return 0


if __name__ == "__main__":
    sys.exit(main())
