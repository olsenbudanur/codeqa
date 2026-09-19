"""SWE-QA-Bench rows -> eval Task. Citations regex-extracted from the answer prose, resolved against the snapshot.
(swe-qa/SWE-QA-Benchmark, Apache-2.0; commits in repo_commit.txt on GitHub; 15 repos x 48 = 720 rows)
"""
from __future__ import annotations

import re
from typing import Any

from codeqa.datagen.resolve import RepoIndex, candidate_identifiers
from codeqa.shared.contracts import Grading, Span, Task, make_repo_id

DATASET = "swe-qa/SWE-QA-Benchmark"
GITHUB_REPO = ("peng-weihan", "SWE-QA-Bench")

PATH_RE = re.compile(r"`?((?:[\w.\-]+/)*[\w.\-]+\.py)`?")
LINES_RE = re.compile(r"\blines?\s+(\d+)\s*(?:[-–]|to)\s*(\d+)|\bline\s+(\d+)\b", re.I)

# HF split name -> (owner, repo)
REPOS = {
    "astropy": ("astropy", "astropy"), "conan": ("conan-io", "conan"), "django": ("django", "django"),
    "flask": ("pallets", "flask"), "matplotlib": ("matplotlib", "matplotlib"), "pylint": ("pylint-dev", "pylint"),
    "pytest": ("pytest-dev", "pytest"), "reflex": ("reflex-dev", "reflex"), "requests": ("psf", "requests"),
    "scikit_learn": ("scikit-learn", "scikit-learn"), "sphinx": ("sphinx-doc", "sphinx"), "sqlfluff": ("sqlfluff", "sqlfluff"),
    "streamlink": ("streamlink", "streamlink"), "sympy": ("sympy", "sympy"), "xarray": ("pydata", "xarray"),
}


# Rows dropped after review: question has no determinate answer (its own reference says the answer is builtin `object`)
EXCLUDE: set[str] = {"pylint/34"}


def parse_repo_commits(text: str) -> dict[str, str]:
    """repo_commit.txt lines look like '<url or name> <sha>' in some order; be permissive."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        sha = next((p for p in parts if re.fullmatch(r"[0-9a-f]{7,40}", p)), None)
        name = next((p for p in parts if p != sha), None)
        if sha and name:
            key = name.rstrip("/").split("/")[-1].removesuffix(".git").replace("-", "_").lower()
            out[key] = sha
    return out


def infer_type(question: str) -> str:
    w = question.strip().split()[0].lower() if question.strip() else ""
    return {"where": "locate", "what": "explain", "why": "explain", "how": "explain"}.get(w, "explain")


def extract_citations(answer: str) -> list[Span]:
    """Pair each 'lines a-b' mention with the nearest preceding .py path (paths as written, unresolved)."""
    spans: list[Span] = []
    last_path: str | None = None
    tokens = sorted(
        [(m.start(), "path", m) for m in PATH_RE.finditer(answer)] + [(m.start(), "lines", m) for m in LINES_RE.finditer(answer)],
        key=lambda t: t[0],
    )
    for _, kind, m in tokens:
        if kind == "path":
            last_path = m.group(1)
        elif last_path:
            a, b, single = m.group(1), m.group(2), m.group(3)
            start = int(a or single)
            end = int(b) if b else start
            if end >= start:
                spans.append(Span(path=last_path, start=start, end=end))
    seen, out = set(), []
    for s in spans:
        k = (s.path, s.start, s.end)
        if k not in seen:
            seen.add(k); out.append(s)
    return out


def resolve_citations(index: RepoIndex, spans: list[Span], answer: str = "") -> tuple[list[Span], dict[str, int]]:
    """Resolve each span's path (exact / unique basename / unique suffix) and clip to file length.
    Returns (kept spans, counts: extracted, path_resolved, in_range)."""
    counts = {"extracted": len(spans), "path_resolved": 0, "in_range": 0}
    out: list[Span] = []
    seen: set[tuple[str, int, int]] = set()
    hints = candidate_identifiers(answer, include_bare=True) if answer else None
    for s in spans:
        p = index.resolve_path(s.path, hints)
        if not p:
            continue
        counts["path_resolved"] += 1
        c = index.clip_span(Span(path=p, start=s.start, end=s.end))
        if not c:
            continue
        counts["in_range"] += 1
        k = (c.path, c.start, c.end)
        if k not in seen:
            seen.add(k); out.append(c)
    return out, counts


def to_task(row: dict[str, Any], split_name: str, sha: str, idx: int, index: RepoIndex | None = None) -> Task:
    owner, repo = REPOS[split_name]
    answer = row["answer"].strip()
    citations = extract_citations(answer)
    raw_paths = sorted({m.group(1) for m in PATH_RE.finditer(answer)})
    if index is not None:
        citations, _ = resolve_citations(index, citations, answer)
        hints = candidate_identifiers(answer, include_bare=True)
        paths = sorted({p for p in (index.resolve_path(r, hints) for r in raw_paths) if p})
    else:
        paths = [p for p in raw_paths if "/" in p]
    return Task(
        task_id=f"sweqa-{split_name}-{idx:03d}",
        repo_id=make_repo_id(owner, repo, sha),
        split="eval",
        question=row["question"].strip(),
        task_type=infer_type(row["question"]),  # type: ignore[arg-type]
        source="sweqa",
        source_id=f"{split_name}/{idx}",
        grading=Grading(reference_answer=answer, required_citations=citations, expected_paths=paths),
    )
