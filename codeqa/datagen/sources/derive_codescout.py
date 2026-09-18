"""CodeScout rows -> locate/trace Task with programmatic gold.

Question is a placeholder until the Haiku rewrite step (datagen/sources/rewrite.py) runs;
gold (expected_paths, expected_symbols) is final and needs no LLM.
(OpenHands/SWE-rebench-code-search; license unverified on HF)
"""
from __future__ import annotations

from typing import Any

from codeqa.shared.contracts import Grading, Task, make_repo_id

DATASET = "OpenHands/SWE-rebench-code-search"

SKIP_FILE_PATTERNS = ("test", "docs/", ".rst", ".md", "setup.py", "pyproject", "CHANGELOG", "__init__.py")


def repo_id_of(row: dict[str, Any]) -> str:
    owner, repo = row["repo"].split("/")[:2]
    return make_repo_id(owner, repo, row["base_commit"])


def gold(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    symbols: list[str] = []
    for fc in row["file_changes"]:
        f = fc["file"]
        if any(p in f for p in SKIP_FILE_PATTERNS):
            continue
        ch = fc["changes"] or {}
        ents = (ch.get("edited_entities") or []) + (ch.get("edited_modules") or [])
        if not ents:
            continue
        paths.append(f)
        symbols.extend(ents)
    return sorted(set(paths)), sorted(set(symbols))


def placeholder_question(problem_statement: str) -> str:
    first = problem_statement.strip().splitlines()[0][:200]
    return f"Where in the codebase is the behavior described here implemented? {first}"


def to_task(row: dict[str, Any], question: str | None = None) -> Task | None:
    paths, symbols = gold(row)
    if not paths or len(paths) > 3:
        return None
    return Task(
        task_id=f"cs-{row['instance_id']}",
        repo_id=repo_id_of(row),
        split="train",
        question=question or placeholder_question(row["problem_statement"]),
        task_type="locate" if len(paths) == 1 else "trace",
        source="codescout",
        source_id=row["instance_id"],
        grading=Grading(expected_paths=paths, expected_symbols=symbols),
    )
