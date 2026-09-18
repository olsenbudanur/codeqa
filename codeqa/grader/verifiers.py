"""Verifiable correctness with zero API calls: paths, literals, symbol sets (C7 correctness for locate|value|enumerate|codescout)."""
from __future__ import annotations

import re
from collections.abc import Iterable

from codeqa.grader.repo import RepoFiles
from codeqa.shared.contracts import CitationReport, IndexSymbol, Task


def f1(pred: set, gold: set) -> float:
    if not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def normalize_literal(s: str) -> str:
    s = s.strip().strip("`'\"").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower().rstrip(".;,")


def literal_score(answer: str, expected_literal: str) -> float:
    """1.0 if the normalized literal appears in the answer as a whole token."""
    lit = normalize_literal(expected_literal)
    if not lit:
        return 0.0
    text = re.sub(r"\s+", " ", answer.lower())
    pat = r"(?<![\w.])" + re.escape(lit) + r"(?![\w.])" if re.fullmatch(r"[\w.\-]+", lit) else re.escape(lit)
    return 1.0 if re.search(pat, text) else 0.0


def path_score(cited: set[str], expected_paths: Iterable[str], repo: RepoFiles) -> float:
    """Single gold path: exact match among cited paths. Several: F1 between cited and gold."""
    gold = {repo.normalize(p) for p in expected_paths}
    if not gold:
        return 0.0
    if len(gold) == 1:
        return 1.0 if gold <= cited else 0.0
    return f1(cited, gold)


def _mentioned(answer: str, sym: IndexSymbol) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(sym.name) + r"(?![\w])", answer) is not None


def predicted_symbols(answer: str, report: CitationReport, repo: RepoFiles) -> set[str]:
    """Symbols the answer names AND whose definition line lies in a cited range. Both are required so a
    symbol cannot be claimed by name-dropping or by citing a whole file."""
    out: set[str] = set()
    for c in report.citations:
        for s in repo.symbols_in(c.path, c.start, c.end):
            if _mentioned(answer, s):
                out.add(s.qualified)
    return out


def resolve_gold_symbols(expected_symbols: Iterable[str], repo: RepoFiles) -> set[str]:
    gold: set[str] = set()
    for q in expected_symbols:
        hits = repo.find_symbols(q)
        gold.update(s.qualified for s in hits) if hits else gold.add(q)
    return gold


def symbol_score(answer: str, report: CitationReport, expected_symbols: Iterable[str], repo: RepoFiles) -> float:
    """One gold symbol: any-of (1.0 if predicted). Several: F1 over qualified names."""
    gold = resolve_gold_symbols(expected_symbols, repo)
    if not gold:
        return 0.0
    pred = predicted_symbols(answer, report, repo)
    if len(list(expected_symbols)) == 1:
        return 1.0 if pred & gold else 0.0
    return f1(pred, gold)


def uses_judge(task: Task) -> bool:
    """Judged types are trace|explain with a rubric or reference answer. Other types verify when they carry gold
    (literal, symbols, or paths); a task with no gold but a rubric/reference falls back to the judge rather than
    scoring 0 for "nothing to verify"."""
    g = task.grading
    has_material = bool(g.rubric or g.reference_answer)
    if task.task_type in ("trace", "explain"):
        return has_material
    return has_material and not g.is_verifiable


def verify(task: Task, answer: str, report: CitationReport, repo: RepoFiles) -> tuple[float, str]:
    """Correctness for verifiable tasks, in priority order: literal, symbols, paths."""
    g = task.grading
    if g.expected_literal is not None:
        s = literal_score(answer, g.expected_literal)
        return s, f"literal {g.expected_literal!r}: {'hit' if s else 'miss'}"
    if g.expected_symbols:
        s = symbol_score(answer, report, g.expected_symbols, repo)
        return s, f"symbols {len(g.expected_symbols)} gold: {s:.2f}"
    if g.expected_paths:
        cited = {repo.normalize(c.path) for c in report.citations}
        s = path_score(cited, g.expected_paths, repo)
        return s, f"paths {len(g.expected_paths)} gold: {s:.2f}"
    return 0.0, "no gold in task; nothing to verify"
