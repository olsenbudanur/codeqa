"""Grade traces against tasks from the command line.

  uv run python -m codeqa.grader --tasks tests/fixtures/tasks.jsonl --traces tests/fixtures/traces --judge keyword
  uv run python -m codeqa.grader --tasks data/tasks/eval/x.jsonl --traces data/traces/run1 --variant multiplicative --out grades.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from codeqa.grader.efficiency import VARIANTS
from codeqa.grader.grade import grade, metrics
from codeqa.grader.judge import KeywordJudge, default_client
from codeqa.shared.contracts import Task, Trace
from codeqa.shared.jsonl import read_all


def load_traces(p: Path) -> list[Trace]:
    if p.is_dir():
        return [Trace.model_validate_json(f.read_text()) for f in sorted(p.glob("*.json"))]
    if p.suffix == ".jsonl":
        return read_all(p, Trace)
    return [Trace.model_validate_json(p.read_text())]


async def run(args: argparse.Namespace) -> int:
    tasks = {t.task_id: t for t in read_all(Path(args.tasks), Task)}
    traces = load_traces(Path(args.traces))
    client = {"keyword": KeywordJudge(), "haiku": None, "none": KeywordJudge(fail=True)}[args.judge]
    if args.judge == "haiku":
        client = default_client()
    rows = []
    print(f"{'trace':28s} {'task':14s} {'reward':>7s} {'gate':12s} {'corr':>5s} {'eff':>5s}  notes", flush=True)
    for tr in traces:
        task = tasks.get(tr.task_id)
        if task is None:
            print(f"{tr.profile:28s} {tr.task_id:14s}  (no task record)", flush=True)
            continue
        r = await grade(task, tr, variant=args.variant, judge_client=client)
        m = metrics(r, tr, task)
        rows.append({"task_id": tr.task_id, "profile": tr.profile, **r.model_dump(), "metrics": m})
        rw = "NaN" if math.isnan(r.reward) else f"{r.reward:.2f}"
        print(f"{tr.profile[:28]:28s} {tr.task_id[:14]:14s} {rw:>7s} {str(r.gate_failed):12s} "
              f"{r.components.correctness:5.2f} {r.components.efficiency:5.2f}  {r.notes[:70]}", flush=True)
    if rows:
        keys = ["reward", "format_ok", "citations_grounded", "correctness", "efficiency", "tool_calls", "judge_error"]
        print("mean  " + "  ".join(f"{k}={sum(x['metrics'][k] for x in rows) / len(rows):.3f}" for k in keys), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for x in rows:
                f.write(json.dumps(x, default=float) + "\n")
        print(f"wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="C5 JSONL task file")
    ap.add_argument("--traces", required=True, help="C6 trace JSON file, JSONL file, or a directory of *.json")
    ap.add_argument("--variant", default="none", choices=VARIANTS)
    ap.add_argument("--judge", default="keyword", choices=["keyword", "haiku", "none"],
                    help="keyword = offline stand-in; haiku = real judge; none = simulate judge outage")
    ap.add_argument("--out", default=None, help="write per-trace grade rows as JSONL")
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
