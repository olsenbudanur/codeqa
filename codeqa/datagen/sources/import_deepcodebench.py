"""DeepCodeBench rows -> Task. facts -> rubric; cited paths -> expected_paths; named symbols -> required_citations.
(Qodo/deep_code_bench, Apache-2.0; 912 train + 232 test over 8 repos, one commit per repo in row metadata)
"""
from __future__ import annotations

import re
from typing import Any

from codeqa.datagen.resolve import RepoIndex, resolve_answer_spans, resolve_paths
from codeqa.shared.contracts import Grading, Task, make_repo_id

DATASET = "Qodo/deep_code_bench"
PATH_RE = re.compile(r"(?<![\w/])((?:[\w.\-]+/)+[\w.\-]+\.(?:py|pyx|pyi|cc|cpp|h|hpp|js|ts|tsx|md|toml|cfg|yaml|yml))\b")


def repo_of(row: dict[str, Any]) -> tuple[str, str, str]:
    url = row["metadata"]["repo"].removesuffix(".git").removeprefix("https://github.com/")
    owner, repo = url.split("/")[:2]
    return owner, repo, row["metadata"]["commit"]


def repo_id_of(row: dict[str, Any]) -> str:
    return make_repo_id(*repo_of(row))


def infer_type(question: str) -> str:
    q = question.lower()
    if q.startswith(("where", "which file", "in which file", "which module")):
        return "locate"
    if q.startswith(("what is the default", "what value", "what scaling", "what number", "which internal attribute", "what is the value")):
        return "value"
    if q.startswith(("which", "what are the", "list")):
        return "enumerate"
    return "explain"


# Row-level corrections found in review (LOG 2026-09-18 22:10). Applied at import so re-runs keep them.
# - rubric/answer typo: the version guard is _DASK_2024_12_1, so the range starts at 2024.12.1, not 2024.1.1
# - three rows labelled `enumerate` by the first-word heuristic are only answerable from the file context DeepCodeBench
#   assumes; they carry a facts rubric, so they are graded as `explain` (judge) instead of path/symbol F1.
FIXES: dict[str, dict[str, Any]] = {
    "4344b2a4-42dd-441f-8c9c-2438db99176b": {"replace": ("≥ 2024.1.1", "≥ 2024.12.1")},
    "d806e6bd-9e8a-4759-8b83-01fdfdfba005": {"task_type": "explain", "question_suffix": " (in the Python package)"},   # now in EXCLUDE
    "d9050518-df81-40ed-a90d-a8a3310f577f": {"task_type": "explain", "question_suffix": " (in the CSI500 index collector)"},
    # Sonnet validation of the fast set (23:55): three questions answered correctly about a different, plausible target in
    # a large repo (DeepCodeBench assumes the source file as context); the qualifier restores that context
    "6602505a-78ab-43d8-ba75-00346d015c8f": {"question_suffix": " (in the SAM model)"},
    "c5c09820-4ff9-43e1-aabc-7c29436eb32d": {"question_suffix": " (for SparseCategoricalCrossentropy)"},
    # fast-set frontier check, round 2: correct answers zeroed by path-F1 on rows that carry a facts rubric -> explain;
    # one more question that never names its pipeline (Sonnet answered about SDXL, rubric is SD3)
    "ba0f687e-5de9-4782-8933-5288e598159b": {"task_type": "explain"},
    "eaf163e5-ed17-45ad-9069-47e202b6130b": {"task_type": "explain"},
    "61f91ebc-d983-406e-b9c2-1ad2fbb55569": {"question_suffix": " (in StableDiffusion3Pipeline)", "paths_from_citations": True},
    "d58dd0e8-89c7-48b4-8da4-1fa6a3ae3aac": {"task_type": "explain", "paths_from_citations": True},
    # repos with a C++ core and a Python package: the rubric is about the Python side, the question did not say so
    # (Sonnet answered from src/common/ranking_utils.h and src/io/dataset.cpp and scored 0/4, 0/5)
    "625f7f86-54ab-4445-89b2-807c16a8cee8": {"question_suffix": " (in the Python package)"},
}


# Rows whose gold does not hold at the pinned commit (verified by grep on the snapshot, 2026-09-19):
# - b900be44: "gpu_coord_descent" appears only in src/gbm/gblinear.cc (C++ fatal), not in python-package/ as the answer claims
# - d806e6bd: the "different number of rows" LightGBMError message exists nowhere in the snapshot; Python raises ValueError + warns
EXCLUDE: set[str] = {"b900be44-8949-4444-8667-4f44cc0a1ba3", "d806e6bd-9e8a-4759-8b83-01fdfdfba005"}


def apply_fixes(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fix = FIXES.get(row["id"], {})
    if "question_suffix" in fix and not row["question"].rstrip().endswith(fix["question_suffix"].strip()):
        row = dict(row, question=row["question"].rstrip() + fix["question_suffix"])
    if "replace" in fix:
        a, b = fix["replace"]
        row = dict(row, answer=row["answer"].replace(a, b), facts=[f.replace(a, b) for f in row.get("facts", [])])
    return row, fix


def to_task(row: dict[str, Any], split: str = "train", index: RepoIndex | None = None) -> Task:
    """Without an index: paths as written in the answer, no citations. With one: paths verified, symbols -> spans."""
    row, fix = apply_fixes(row)
    answer = row["answer"].strip()
    raw_paths = sorted({m.group(1) for m in PATH_RE.finditer(answer)})
    paths, citations = raw_paths, []
    if index is not None:
        paths, _ = resolve_paths(index, raw_paths)
        citations = resolve_answer_spans(index, answer + "\n" + "\n".join(row.get("facts", [])), paths)
        if fix.get("paths_from_citations") and not paths:
            paths = sorted({c.path for c in citations})
    return Task(
        task_id=f"dcb-{row['id'][:8]}",
        repo_id=repo_id_of(row),
        split=split,  # type: ignore[arg-type]
        question=row["question"].strip(),
        task_type=fix.get("task_type", infer_type(row["question"])),  # type: ignore[arg-type]
        source="deepcodebench",
        source_id=row["id"],
        grading=Grading(
            expected_paths=paths,
            reference_answer=answer,
            rubric=[f.strip() for f in row.get("facts", []) if f.strip()],
            required_citations=citations,
        ),
    )
