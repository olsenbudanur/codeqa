"""Compare eval sets side by side (agent variants, or checkpoints) from data/evals/<profile>/<set>/per_task.jsonl.

  uv run python -m scripts.compare_evals qwen4b-base fast_default fast_bash fast_noindex
  uv run python -m scripts.compare_evals qwen4b-base:fast_default qwen4b-smoke1-step3:fast claude:fast
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter

from codeqa.shared import paths
from codeqa.shared.contracts import CITATION_RE


def load(profile: str, set_name: str) -> list[dict]:
    p = paths.EVALS / profile / set_name / "per_task.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def gold_paths(task: dict) -> set[str]:
    g = task["grading"]
    paths_ = set(g.get("expected_paths") or [])
    for s in g.get("expected_symbols") or []:
        paths_.add(s.split(":", 1)[0])
    return paths_


def summarize(rows: list[dict], tasks: dict[str, dict]) -> dict[str, float]:
    n = len(rows)
    def mean(key, sub=None):
        vals = [r["metrics"].get(key) for r in (sub or rows) if isinstance(r.get("metrics"), dict) and r["metrics"].get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")
    gates = Counter(r.get("gate_failed") for r in rows)
    passed = [r for r in rows if r.get("gate_failed") is None]
    stops = Counter(r.get("stop_reason") for r in rows)
    # lenient "found it": for tasks with gold paths, does the answer mention any gold path (cited or in prose)?
    found, n_gold = 0, 0
    for r in rows:
        gp = gold_paths(tasks[r["task_id"]])
        if not gp:
            continue
        n_gold += 1
        ans = r.get("answer") or ""
        if any(p in ans or p.rsplit("/", 1)[-1] in ans for p in gp):
            found += 1
    return {
        "n": n,
        "reward": sum(r["reward"] for r in rows) / n,
        "correct_rate": sum(1 for r in passed if r["components"]["correctness"] >= 0.99) / n,
        "gates_passed": len(passed) / n,
        "correct_given_pass": (sum(r["components"]["correctness"] for r in passed) / len(passed)) if passed else float("nan"),
        "answered": 1 - (gates.get("format", 0) / n) if True else 0,
        "format_fail": gates.get("format", 0) / n,
        "cite_fail": (gates.get("citations", 0) + gates.get("grounding", 0)) / n,
        "has_citation": sum(1 for r in rows if CITATION_RE.search(r.get("answer") or "")) / n,
        "found_gold_path": found / n_gold if n_gold else float("nan"),
        "tool_calls": mean("tool_calls"),
        "prompt_tokens": mean("prompt_tokens"),
        "stop_answer": stops.get("answer", 0) / n,
        "stop_budget": stops.get("budget", 0) / n,
        "stop_max_turns": stops.get("max_turns", 0) / n,
    }


def main(argv: list[str]) -> None:
    profile_default = argv[0] if ":" not in argv[0] else None
    specs = argv[1:] if profile_default else argv
    cols = []
    for s in specs:
        prof, set_name = s.split(":", 1) if ":" in s else (profile_default, s)
        cols.append((f"{prof}:{set_name}", load(prof, set_name)))
    task_file = paths.TASKS_EVAL / "fast.jsonl"
    tasks = {json.loads(l)["task_id"]: json.loads(l) for l in task_file.read_text().splitlines() if l.strip()}
    sums = [(name, summarize(rows, tasks)) for name, rows in cols]
    keys = list(sums[0][1].keys())
    w = max(len(k) for k in keys)
    print(" " * w + "  " + "  ".join(f"{name[:26]:>26}" for name, _ in sums))
    for k in keys:
        vals = []
        for _, s in sums:
            v = s[k]
            vals.append(f"{v:>26.0f}" if k in ("n", "prompt_tokens") else (f"{v:>26.2f}" if not math.isnan(v) else f"{'-':>26}"))
        print(f"{k:<{w}}  " + "  ".join(vals))
    # per source x type reward + gates_passed
    print("\nby source/type: gates_passed / correct_rate")
    groups = sorted({(tasks[r['task_id']]['source'], tasks[r['task_id']]['task_type']) for _, rows in cols for r in rows})
    for g in groups:
        cells = []
        for _, rows in cols:
            sub = [r for r in rows if (tasks[r['task_id']]['source'], tasks[r['task_id']]['task_type']) == g]
            if not sub:
                cells.append(f"{'-':>26}"); continue
            passed = [r for r in sub if r.get('gate_failed') is None]
            cr = sum(1 for r in passed if r['components']['correctness'] >= 0.99) / len(sub)
            cells.append(f"{len(passed)/len(sub):>12.2f} / {cr:<11.2f}")
        print(f"{'/'.join(g):<{w}}  " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
