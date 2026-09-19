"""B6b: pass-rate window + per-repo cap + repo-aware split -> data/tasks/train/*.jsonl, eval/fast.jsonl, counts table.

Keep a training task when its difficulty score (correctness given format, else ungated correctness) lies in
[lo, hi]. Tasks below lo go to raw/reserve_hard.jsonl, above hi to raw/reserve_easy.jsonl (curriculum later).
Unmeasured tasks are kept only with --keep-unmeasured. Eval repos never appear in train (checked, not assumed).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from typing import Any

from codeqa.datagen.filter import PASSRATE, RAW_SOURCES, dedupe, difficulty_score, load_raw
from codeqa.datagen.sources import import_sweqa as sq
from codeqa.shared import paths
from codeqa.shared.contracts import Task
from codeqa.shared.jsonl import read_all, write

REPORTS = paths.TASKS / "reports"


def _log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def eval_repo_prefixes() -> set[str]:
    return {f"{o}__{r}__" for o, r in sq.REPOS.values()}


def build(lo: float = 0.1, hi: float = 0.9, per_repo_cap: int = 120, keep_unmeasured: bool = False, fast_n: int = 60,
          seed: int = 7, refresh_fast: bool = False) -> dict[str, Any]:
    raw = load_raw(RAW_SOURCES)
    tasks, dropped = dedupe(raw)
    from codeqa.datagen.filter import load_passrate
    measured = {k: v for k, v in load_passrate().items() if v.get("n")}
    evals = eval_repo_prefixes()
    kept: list[Task] = []
    hard: list[Task] = []
    easy: list[Task] = []
    unmeasured: list[Task] = []
    for t in tasks:
        if any(t.repo_id.startswith(p) for p in evals):
            raise RuntimeError(f"training task {t.task_id} is on an eval repo {t.repo_id}")
        rec = measured.get(t.task_id)
        if rec is None:
            unmeasured.append(t)
            if keep_unmeasured:
                kept.append(t)
            continue
        d = difficulty_score(rec)
        if d is None:
            unmeasured.append(t)
            if keep_unmeasured:
                kept.append(t)
        elif d < lo:
            hard.append(t)
        elif d > hi:
            easy.append(t)
        else:
            kept.append(t)
    # per-repo cap, balanced across sources within the repo
    rng = random.Random(seed)
    by_repo: dict[str, list[Task]] = defaultdict(list)
    for t in kept:
        by_repo[t.repo_id.removesuffix("__nodoc")].append(t)
    capped: list[Task] = []
    overflow: list[Task] = []
    for repo, ts in by_repo.items():
        if len(ts) <= per_repo_cap:
            capped.extend(ts); continue
        by_src: dict[str, list[Task]] = defaultdict(list)
        for t in ts:
            by_src[t.source].append(t)
        for v in by_src.values():
            rng.shuffle(v)
        take: list[Task] = []
        while len(take) < per_repo_cap and any(by_src.values()):
            for v in by_src.values():
                if v and len(take) < per_repo_cap:
                    take.append(v.pop())
        capped.extend(take)
        overflow.extend(t for v in by_src.values() for t in v)
    rng.shuffle(capped)
    # write train files
    for src in RAW_SOURCES:
        write(paths.TASKS_TRAIN / f"{src}.jsonl", [t for t in capped if t.source == src])
    n_all = write(paths.TASKS_TRAIN / "all.jsonl", capped)
    write(paths.TASKS_RAW / "reserve_hard.jsonl", hard)
    write(paths.TASKS_RAW / "reserve_easy.jsonl", easy + overflow)
    # fast eval: keep the ids already in eval/fast.jsonl (baselines were run on them), refreshed from the current records;
    # only sample anew when there is no file or refresh_fast is set
    fast: list[Task] = []
    fast_path = paths.TASKS_EVAL / "fast.jsonl"
    if fast_path.exists() and not refresh_fast:
        keep_ids = [t.task_id for t in read_all(fast_path, Task)]
        pool = {t.task_id: t for name in ("deepcodebench_test", "sweqa") for t in read_all(paths.TASKS_EVAL / f"{name}.jsonl", Task)
                if (paths.TASKS_EVAL / f"{name}.jsonl").exists()}
        fast = [pool[i] for i in keep_ids if i in pool]
        # ids that vanished (excluded after review) are replaced from the same source, seeded, repos not yet in fast first
        have = {t.task_id for t in fast}
        for src, name in (("deepcodebench", "deepcodebench_test"), ("sweqa", "sweqa")):
            need = fast_n - sum(1 for t in fast if t.source == src)
            if need <= 0:
                continue
            cands = [t for t in pool.values() if t.source == src and t.task_id not in have]
            rng.shuffle(cands)
            used_repos = {t.repo_id for t in fast if t.source == src}
            cands.sort(key=lambda t: t.repo_id in used_repos)
            for t in cands[:need]:
                fast.append(t); have.add(t.task_id)
    for name in (() if fast else ("deepcodebench_test", "sweqa")):
        p = paths.TASKS_EVAL / f"{name}.jsonl"
        if not p.exists():
            continue
        ts = read_all(p, Task)
        by_r: dict[str, list[Task]] = defaultdict(list)
        for t in ts:
            by_r[t.repo_id].append(t)
        for v in by_r.values():
            rng.shuffle(v)
        pick: list[Task] = []
        while len(pick) < fast_n and any(by_r.values()):
            for v in by_r.values():
                if v and len(pick) < fast_n:
                    pick.append(v.pop())
        fast.extend(pick)
    n_fast = write(paths.TASKS_EVAL / "fast.jsonl", fast)
    counts = {"train": _table(capped), "reserve_hard": len(hard), "reserve_easy": len(easy), "over_cap": len(overflow),
              "unmeasured": len(unmeasured), "kept_unmeasured": keep_unmeasured, "deduped": len(dropped),
              "eval": {"deepcodebench_test": _count(paths.TASKS_EVAL / "deepcodebench_test.jsonl"),
                       "sweqa": _count(paths.TASKS_EVAL / "sweqa.jsonl"), "fast": n_fast},
              "window": [lo, hi], "per_repo_cap": per_repo_cap, "train_all": n_all}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "split.json").write_text(json.dumps(counts, indent=2))
    (paths.TASKS / "README.md").write_text(readme(counts))
    _log(f"  split: train/all.jsonl = {n_all}; hard {len(hard)}, easy {len(easy)}, over-cap {len(overflow)}, unmeasured {len(unmeasured)}; fast eval {n_fast}")
    return counts


def _count(p) -> int:
    return sum(1 for line in p.read_text().splitlines() if line.strip()) if p.exists() else 0


def _table(tasks: list[Task]) -> dict[str, Any]:
    by = Counter((t.source, t.task_type) for t in tasks)
    sources = sorted({s for s, _ in by}); types = ["locate", "value", "enumerate", "trace", "explain"]
    return {"rows": {s: {tt: by.get((s, tt), 0) for tt in types} | {"total": sum(by.get((s, tt), 0) for tt in types)} for s in sources},
            "types": {tt: sum(by.get((s, tt), 0) for s in sources) for tt in types}, "total": len(tasks),
            "repos": len({t.repo_id.removesuffix('__nodoc') for t in tasks}),
            "programmatic_share": round(sum(1 for t in tasks if t.task_type in ("locate", "value", "enumerate") or t.source in ("codescout", "structural")) / max(len(tasks), 1), 3)}


def readme(c: dict[str, Any]) -> str:
    rows = c["train"]["rows"]; types = ["locate", "value", "enumerate", "trace", "explain"]
    lines = ["# data/tasks", "", "Generated by `codeqa.datagen` (lane B). Records are C5. Do not edit by hand; re-run the CLI.", "",
             f"## train/all.jsonl — {c['train']['total']} tasks over {c['train']['repos']} repos "
             f"(pass-rate window {c['window']}, per-repo cap {c['per_repo_cap']}, programmatic share {c['train']['programmatic_share']:.0%})", "",
             "| source | " + " | ".join(types) + " | total |", "|---|" + "---|" * (len(types) + 1)]
    for s, r in rows.items():
        lines.append(f"| {s} | " + " | ".join(str(r[t]) for t in types) + f" | {r['total']} |")
    lines += ["| **all** | " + " | ".join(str(c["train"]["types"][t]) for t in types) + f" | **{c['train']['total']}** |", "",
              f"Reserves in `raw/`: {c['reserve_hard']} hard (base model never right), {c['reserve_easy']} easy or over the per-repo cap. "
              f"{c['unmeasured']} tasks unmeasured{' (kept)' if c['kept_unmeasured'] else ' (held out of train)'}; {c['deduped']} near-duplicates dropped.", "",
              "## eval/", "", f"- `deepcodebench_test.jsonl`: {c['eval']['deepcodebench_test']} (held-out, same 8 repos as train)",
              f"- `sweqa.jsonl`: {c['eval']['sweqa']} (15 repos never trained on)",
              f"- `fast.jsonl`: {c['eval']['fast']} (60 + 60 stratified by repo; the every-N-steps eval)", "",
              "Reports with resolution rates, yields, pass-rate summaries: `reports/`."]
    return "\n".join(lines) + "\n"
