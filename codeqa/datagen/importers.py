"""Full imports: DeepCodeBench (train + test) and SWE-QA-Bench (eval), with snapshots ensured and citations resolved.

Produces C5 files:
  data/tasks/raw/deepcodebench.jsonl          912 train records
  data/tasks/eval/deepcodebench_test.jsonl     232 held-out in-repo records
  data/tasks/eval/sweqa.jsonl                  720 unseen-repo records
and a counts report per source under data/tasks/reports/<source>.json.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from codeqa.clients import github
from codeqa.datagen import cache
from codeqa.datagen.repos import ensure_repo
from codeqa.datagen.resolve import RepoIndex
from codeqa.datagen.sources import import_deepcodebench as dcb
from codeqa.datagen.sources import import_codeqabench as cqb
from codeqa.datagen.sources import import_sweqa as sq
from codeqa.shared import paths
from codeqa.shared.contracts import Task
from codeqa.shared.jsonl import write

REPORTS = paths.TASKS / "reports"


def _log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def _save_report(name: str, report: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{name}.json").write_text(json.dumps(report, indent=2))


def _indexes_for(repo_keys: set[tuple[str, str, str]]) -> dict[str, RepoIndex]:
    out: dict[str, RepoIndex] = {}
    for owner, repo, sha in sorted(repo_keys):
        m, syms = ensure_repo(owner, repo, sha)
        out[m.repo_id] = RepoIndex(m, syms)
    return out


def import_deepcodebench() -> dict[str, Any]:
    t0 = time.time()
    rows = {s: cache.load_cached(dcb.DATASET, "default", s) for s in ("train", "test")}
    indexes = _indexes_for({dcb.repo_of(r) for s in rows for r in rows[s]})
    report: dict[str, Any] = {"source": "deepcodebench", "repos": sorted(indexes)}
    for split, out_path, task_split in (("train", paths.TASKS_RAW / "deepcodebench.jsonl", "train"),
                                        ("test", paths.TASKS_EVAL / "deepcodebench_test.jsonl", "eval")):
        tasks: list[Task] = []
        raw_paths = ok_paths = 0
        for r in rows[split]:
            if r["id"] in dcb.EXCLUDE:
                continue
            idx = indexes[dcb.repo_id_of(r)]
            plain = dcb.to_task(r, task_split)                # unresolved, for the path-resolution rate
            t = dcb.to_task(r, task_split, index=idx)
            raw_paths += len(plain.grading.expected_paths); ok_paths += len(t.grading.expected_paths)
            tasks.append(t)
        n = write(out_path, tasks)
        with_cit = sum(1 for t in tasks if t.grading.required_citations)
        with_path = sum(1 for t in tasks if t.grading.expected_paths)
        per_repo = Counter(t.repo_id for t in tasks)
        cit_by_repo = Counter(t.repo_id for t in tasks if t.grading.required_citations)
        report[split] = {
            "records": n, "file": str(out_path.relative_to(paths.ROOT)),
            "types": dict(Counter(t.task_type for t in tasks)),
            "paths_mentioned": raw_paths, "paths_resolved": ok_paths,
            "tasks_with_expected_path": with_path, "tasks_with_citations": with_cit,
            "citation_rate_by_repo": {k: f"{cit_by_repo[k]}/{per_repo[k]}" for k in sorted(per_repo)},
            "rubric_items_mean": round(sum(len(t.grading.rubric) for t in tasks) / max(n, 1), 2),
        }
        report[split]["excluded"] = sum(1 for r in rows[split] if r["id"] in dcb.EXCLUDE)
        _log(f"  deepcodebench/{split}: {n} records ({report[split]['excluded']} excluded); paths {ok_paths}/{raw_paths} resolved; "
             f"{with_cit}/{n} tasks with required_citations; types {report[split]['types']}")
    report["seconds"] = round(time.time() - t0, 1)
    _save_report("deepcodebench", report)
    return report


def sweqa_commits() -> dict[str, str]:
    cached = paths.DATA / "cache" / "sweqa_repo_commit.txt"
    if cached.exists():
        return sq.parse_repo_commits(cached.read_text())
    text = github.fetch_text(*sq.GITHUB_REPO, "repo_commit.txt")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(text)
    return sq.parse_repo_commits(text)


def import_sweqa(rubrics: bool = True) -> dict[str, Any]:
    t0 = time.time()
    commits = sweqa_commits()
    missing = [s for s in sq.REPOS if s not in commits]
    if missing:
        raise RuntimeError(f"repo_commit.txt lacks commits for {missing}")
    tasks: list[Task] = []
    per_repo: dict[str, dict[str, int]] = {}
    for split, (owner, repo) in sq.REPOS.items():
        m, syms = ensure_repo(owner, repo, commits[split])
        idx = RepoIndex(m, syms)
        rows = cache.load_cached(sq.DATASET, "default", split)
        tot: Counter[str] = Counter()
        with_cit = 0
        for i, r in enumerate(rows):
            if f"{split}/{i}" in sq.EXCLUDE:
                continue
            _, c = sq.resolve_citations(idx, sq.extract_citations(r["answer"]), r["answer"])
            tot.update(c)
            t = sq.to_task(r, split, m.sha, i, index=idx)
            with_cit += bool(t.grading.required_citations)
            tasks.append(t)
        per_repo[split] = {"records": len(rows), "tasks_with_citations": with_cit, **tot}
        rate = tot["in_range"] / tot["extracted"] if tot["extracted"] else 0.0
        _log(f"  sweqa/{split}: {len(rows)} rows; citations {tot['in_range']}/{tot['extracted']} resolved ({rate:.0%}); "
             f"{with_cit}/{len(rows)} tasks with >=1")
    if rubrics:
        import asyncio
        from codeqa.datagen.sources import rubrics as rb
        try:
            derived = asyncio.run(rb.derive_all(tasks, "sweqa_rubrics"))
            tasks = rb.apply(tasks, derived)
        except Exception as e:  # noqa: BLE001 - no key / no credits: keep reference-only records, say so loudly
            _log(f"  sweqa: rubric derivation skipped ({type(e).__name__}: {str(e)[:100]}); records keep reference_answer only")
    with_rubric = sum(1 for t in tasks if t.grading.rubric)
    out = paths.TASKS_EVAL / "sweqa.jsonl"
    n = write(out, tasks)
    agg = Counter()
    for v in per_repo.values():
        agg.update({k: v[k] for k in ("extracted", "path_resolved", "in_range", "tasks_with_citations")})
    report = {"source": "sweqa", "records": n, "file": str(out.relative_to(paths.ROOT)), "with_rubric": with_rubric,
              "rubric_items_mean": round(sum(len(t.grading.rubric) for t in tasks) / max(n, 1), 2),
              "types": dict(Counter(t.task_type for t in tasks)), "totals": dict(agg), "per_repo": per_repo,
              "seconds": round(time.time() - t0, 1)}
    _log(f"  sweqa: {n} records ({with_rubric} with a stored rubric); citations {agg['in_range']}/{agg['extracted']} resolved overall; "
         f"{agg['tasks_with_citations']}/{n} tasks with >=1")
    _save_report("sweqa", report)
    return report


def import_codeqabench() -> dict[str, Any]:
    """528 code-derivable Code-QA-Bench rows -> data/tasks/eval/codeqabench.jsonl (repos snapshotted at repos.json SHAs)."""
    t0 = time.time()
    repos, rows = cqb.load()
    indexes: dict[str, RepoIndex] = {}
    for key in sorted({r["repo"] for r in rows}):
        owner, name, sha = cqb.owner_repo(repos, key)
        m, syms = ensure_repo(owner, name, sha)
        indexes[key] = RepoIndex(m, syms)
    tasks: list[Task] = []
    raw_n = ok_n = 0
    for r in rows:
        plain = cqb.to_task(r, repos)
        t = cqb.to_task(r, repos, index=indexes[r["repo"]])
        raw_n += len(plain.grading.expected_paths); ok_n += len(t.grading.expected_paths)
        tasks.append(t)
    out = paths.TASKS_EVAL / "codeqabench.jsonl"
    n = write(out, tasks)
    report = {"source": "codeqabench", "records": n, "file": str(out.relative_to(paths.ROOT)),
              "repos": {k: indexes[k].repo_id for k in indexes}, "types": dict(Counter(t.task_type for t in tasks)),
              "key_files_mentioned": raw_n, "key_files_resolved": ok_n,
              "rubric_items_mean": round(sum(len(t.grading.rubric) for t in tasks) / max(n, 1), 2),
              "condition": "documented (our snapshots keep docs; the paper's primary condition strips them)",
              "seconds": round(time.time() - t0, 1)}
    _save_report("codeqabench", report)
    _log(f"  codeqabench: {n} records over {len(indexes)} repos; key_files {ok_n}/{raw_n} resolved; types {report['types']}; {report['seconds']}s")
    return report
