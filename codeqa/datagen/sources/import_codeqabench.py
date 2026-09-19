"""Code-QA-Bench (Lens-Frontier, MIT) -> eval Tasks. https://github.com/Lens-Frontier/code-qa-bench

528 code-derivable questions over 10 SWE-bench repos (9 overlap SWE-QA-Bench at other commits; seaborn is new), each with
a gold answer, a rubric (~8 atomic items) and key_files (~4). Maps 1:1 onto C5: rubric -> rubric, gold_answer ->
reference_answer, key_files -> expected_paths (resolved against the snapshot). Judged types only: `how` -> trace, the
rest -> explain. Our snapshots keep documentation, so scores correspond to the paper's "documented" condition, not its
primary code-only ("stripped") one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeqa.datagen.resolve import RepoIndex, resolve_paths
from codeqa.shared import paths
from codeqa.shared.contracts import Grading, Task, make_repo_id

RAW_BASE = "https://raw.githubusercontent.com/Lens-Frontier/code-qa-bench/HEAD/"
CACHE = paths.DATA / "cache" / "codeqabench"
FILES = {"repos.json": "repos.json", "tasks.json": "tasks/tasks.json", "tasks_doc_dependent.json": "tasks/tasks_doc_dependent.json"}


def fetch(force: bool = False) -> dict[str, Path]:
    import httpx
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for local, remote in FILES.items():
        p = CACHE / local
        if force or not p.exists():
            r = httpx.get(RAW_BASE + remote, timeout=60, follow_redirects=True)
            r.raise_for_status()
            p.write_bytes(r.content)
        out[local] = p
    return out


def load(doc_dependent: bool = False) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    f = fetch()
    repos = json.loads(f["repos.json"].read_text())
    tasks = json.loads(f["tasks_doc_dependent.json" if doc_dependent else "tasks.json"].read_text())
    return repos, (tasks if isinstance(tasks, list) else tasks["tasks"])


def owner_repo(repos: dict[str, dict[str, Any]], key: str) -> tuple[str, str, str]:
    url = repos[key]["url"].removesuffix(".git").removeprefix("https://github.com/")
    owner, name = url.split("/")[:2]
    return owner, name, repos[key]["ref"]


def task_type_of(category: str) -> str:
    return "trace" if category == "how" else "explain"     # all rows carry a rubric; judged, never path-F1


def to_task(row: dict[str, Any], repos: dict[str, dict[str, Any]], index: RepoIndex | None = None) -> Task:
    owner, name, sha = owner_repo(repos, row["repo"])
    raw_paths = [str(p) for p in row.get("key_files", [])]
    if index is not None:
        paths_, _ = resolve_paths(index, raw_paths)
    else:
        paths_ = sorted(set(raw_paths))
    return Task(
        task_id=f"cqb-{row['id']}",
        repo_id=make_repo_id(owner, name, sha),
        split="eval",
        question=" ".join(str(row["question"]).split()),
        task_type=task_type_of(str(row.get("category", ""))),  # type: ignore[arg-type]
        source="codeqabench",
        source_id=str(row["id"]),
        grading=Grading(expected_paths=paths_, reference_answer=str(row["gold_answer"]).strip(),
                        rubric=[" ".join(str(r).split()) for r in row.get("rubric", []) if str(r).strip()]),
    )
