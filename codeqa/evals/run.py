"""Offline evaluation: profile x task file -> data/evals/<profile>/<set>/{per_task.jsonl, results.json} (C3).

  uv run python -u -m codeqa.evals.run --profile claude --tasks data/tasks/eval/smoke_sweqa_flask.jsonl
  uv run python -u -m codeqa.evals.run --profile qwen4b-run1-step40 --tasks data/tasks/eval/fast.jsonl --concurrency 8

Runs lane A's `run_episode` with the profile's client (Tinker, Anthropic, or vLLM), grades every trace with the
real grader, writes one row per task, aggregates by source x task_type, and records the summary in
`data/models/manifest.json` when a CheckpointRecord carries this profile name.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

from codeqa.agent.driver import run_episode
from codeqa.agent.env import RepoEnv
from codeqa.clients.base import make_client
from codeqa.grader.efficiency import VARIANTS
from codeqa.grader.grade import grade, metrics as grade_metrics
from codeqa.grader.judge import JudgeClient, KeywordJudge, default_client
from codeqa.shared import paths
from codeqa.shared.contracts import CheckpointRecord, Task, Trace
from codeqa.shared.jsonl import read_all
from codeqa.shared.profiles import get_profile

logger = logging.getLogger(__name__)
EPISODE_TIMEOUT = 240.0

SUMMARY_KEYS = ("reward", "correct", "stalled", "correctness", "format_ok", "citations_parse", "citations_exist", "citations_grounded", "identifier_grounded",
                "efficiency", "judge_error", "tool_calls", "tool_errors", "prompt_tokens", "completion_tokens", "answer_tokens", "turns", "seconds")


def eval_dir(profile: str, set_name: str) -> Path:
    return paths.EVALS / profile / set_name


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Means over rows, plus the talk's headline ratios and the stop-reason distribution."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    out: dict[str, float] = {"n": float(n)}
    for k in SUMMARY_KEYS:
        vals = [r["metrics"].get(k, 0.0) for r in rows]
        out[k] = sum(vals) / n
    correct = [r for r in rows if r["metrics"].get("reward", 0.0) > 0]
    out["correct_rate"] = len(correct) / n
    out["citation_valid"] = out["citations_grounded"]
    out["tool_calls_per_correct"] = (sum(r["metrics"]["tool_calls"] for r in rows) / len(correct)) if correct else float("inf")
    out["prompt_tokens_per_correct"] = (sum(r["metrics"]["prompt_tokens"] for r in rows) / len(correct)) if correct else float("inf")
    for reason in ("answer", "max_turns", "budget", "overflow", "parse_error", "error"):
        out[f"stop_{reason}"] = sum(1 for r in rows if r["stop_reason"] == reason) / n
    for gate in ("format", "citations", "grounding", "budget", "judge_error"):
        out[f"gate_{gate}"] = sum(1 for r in rows if r["gate_failed"] == gate) / n
    return out


def by_group(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(f"{r['source']}/{r['task_type']}", []).append(r)
        groups.setdefault(f"source:{r['source']}", []).append(r)
        groups.setdefault(f"type:{r['task_type']}", []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


async def eval_task(task: Task, profile_name: str, judge: JudgeClient | None, variant: str, temperature: float,
                    sem: asyncio.Semaphore, traces_dir: Path) -> dict[str, Any]:
    profile = get_profile(profile_name)
    errors: list[str] = []

    async def on_event(e):
        if e.type == "error":
            errors.append(str(e.payload.get("message", ""))[:300])

    async with sem:
        t0 = time.time()
        try:
            env = RepoEnv(task, profile)
            client = make_client(profile)
            trace = await asyncio.wait_for(run_episode(env, client, on_event=on_event, temperature=temperature), timeout=EPISODE_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - one bad task must not sink the set
            errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            trace = Trace(task_id=task.task_id, profile=profile_name, messages=[], answer="")
            trace.stats.stop_reason = "error"
            trace.stats.seconds = time.time() - t0
        if errors:
            logger.warning("%s: episode error: %s", task.task_id, errors[-1])
        (traces_dir / f"{task.task_id}.json").write_text(trace.model_dump_json(indent=1))
        result = await grade(task, trace, variant=variant, judge_client=judge)
    m = grade_metrics(result, trace, task)
    m["seconds"] = trace.stats.seconds
    row = {"task_id": task.task_id, "repo_id": task.repo_id, "source": task.source, "task_type": task.task_type, "profile": profile_name,
           "reward": None if math.isnan(result.reward) else result.reward, "gate_failed": result.gate_failed, "notes": result.notes,
           "components": result.components.model_dump(), "stop_reason": trace.stats.stop_reason, "metrics": m,
           "answer": trace.answer[:2000], "episode_error": errors[-1] if errors else None}
    print(f"  {task.task_id:26s} {task.source:13s} {task.task_type:9s} reward={m['reward']:.2f} gate={result.gate_failed} "
          f"calls={trace.stats.tool_calls} stop={trace.stats.stop_reason} {trace.stats.seconds:.0f}s", flush=True)
    return row


def update_manifest(profile_name: str, set_name: str, summary: dict[str, float]) -> bool:
    """Fill CheckpointRecord.evals[set_name] for the record whose profile matches. Returns True if a record was updated."""
    p = paths.MODELS_MANIFEST
    if not p.exists():
        return False
    records = [CheckpointRecord.model_validate(r) for r in json.loads(p.read_text())]
    hit = False
    keep = {k: v for k, v in summary.items() if isinstance(v, (int, float)) and math.isfinite(v)}
    for r in records:
        if r.profile == profile_name:
            r.evals[set_name] = keep
            hit = True
    if hit:
        p.write_text(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
    return hit


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks = read_all(Path(args.tasks), Task)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    set_name = args.set or Path(args.tasks).stem
    out_dir = eval_dir(args.profile, set_name)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    judge = KeywordJudge() if args.judge == "keyword" else (default_client(args.judge_model) if args.judge_model else default_client())
    sem = asyncio.Semaphore(args.concurrency)
    print(f"eval profile={args.profile} set={set_name} tasks={len(tasks)} concurrency={args.concurrency} variant={args.variant} -> {out_dir}", flush=True)
    t0 = time.time()
    rows = await asyncio.gather(*(eval_task(t, args.profile, judge, args.variant, args.temperature, sem, traces_dir) for t in tasks))
    rows = sorted(rows, key=lambda r: r["task_id"])
    with (out_dir / "per_task.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    results = {"profile": args.profile, "set": set_name, "tasks_file": args.tasks, "variant": args.variant, "temperature": args.temperature,
               "seconds": round(time.time() - t0, 1), "summary": summarize(rows), "by_group": by_group(rows)}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    s = results["summary"]
    print(f"done in {results['seconds']}s: n={int(s['n'])} reward={s['reward']:.3f} correct_rate={s['correct_rate']:.3f} "
          f"format_ok={s['format_ok']:.3f} citation_valid={s['citation_valid']:.3f} tool_calls/correct={s['tool_calls_per_correct']:.2f}", flush=True)
    if update_manifest(args.profile, set_name, s):
        print(f"updated {paths.MODELS_MANIFEST} evals[{set_name}] for profile {args.profile}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout, force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--set", default=None, help="eval set name (default: task file stem)")
    ap.add_argument("--variant", default="none", choices=VARIANTS)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--judge", default="haiku", choices=["haiku", "keyword"])
    ap.add_argument("--judge-model", default=None)
    asyncio.run(run(ap.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
