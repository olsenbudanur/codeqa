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


TRIVIAL_LITERALS = {"true", "false", "none", "null", "0", "1", "0.", "1.", "2", "3", "4", "5", "-1", "", "[]", "{}", "''", '""'}


def normalize_literal(s: str) -> str:
    s = s.strip().strip("`'\"").strip()
    s = re.sub(r"\s+", " ", s).lower()
    stripped = s.rstrip(".;,")
    return stripped or s          # a punctuation-only literal such as "." keeps its text


def is_trivial_literal(expected_literal: str) -> bool:
    lit = normalize_literal(expected_literal)
    return lit in TRIVIAL_LITERALS or len(lit) <= 1


def literal_score(answer: str, expected_literal: str) -> float:
    """1.0 if the normalized literal appears in the answer as a whole token."""
    lit = normalize_literal(expected_literal)
    if not lit:
        return 0.0
    text = re.sub(r"\s+", " ", answer.lower())
    pat = r"(?<![\w.])" + re.escape(lit) + r"(?![\w.])" if re.fullmatch(r"[\w.\-]+", lit) else re.escape(lit)
    return 1.0 if re.search(pat, text) else 0.0


def cites_required(report: CitationReport, required: Iterable[Span], repo: RepoFiles) -> bool:
    """True if some grounded citation overlaps one of the task's required_citations (the evidence lines)."""
    req = list(required)
    if not req:
        return True
    for c in report.citations:
        if not c.grounded:
            continue
        p = repo.normalize(c.path)
        if any(repo.normalize(r.path) == p and c.start <= r.end and c.end >= r.start for r in req):
            return True
    return False


SINGLE_PATH_PRECISION = 2      # one gold path: citing more than this many distinct files starts to cost
SINGLE_SYMBOL_PRECISION = 3    # one gold symbol: naming more than this many cited symbols starts to cost


def path_score(cited: set[str], expected_paths: Iterable[str], repo: RepoFiles) -> float:
    """Single gold path: exact match among cited paths, discounted when many files are cited. Several: F1."""
    gold = {repo.normalize(p) for p in expected_paths}
    if not gold:
        return 0.0
    if len(gold) == 1:
        return min(1.0, SINGLE_PATH_PRECISION / max(len(cited), 1)) if gold <= cited else 0.0
    return f1(cited, gold)


def strip_citations(text: str) -> str:
    from codeqa.shared.contracts import CITATION_RE
    return CITATION_RE.sub(" ", text)


def mentions_path(answer: str, path: str) -> bool:
    """The answer names the file outside citation brackets (full path or basename)."""
    body = strip_citations(answer)
    base = path.rsplit("/", 1)[-1]
    return path in body or re.search(r"(?<![\w/])" + re.escape(base) + r"(?![\w])", body) is not None


def _mentioned(answer: str, sym: IndexSymbol) -> bool:
    """The answer names the symbol: bare (`validate_session`) or qualified (`Session.is_expired`, `obj.is_expired`)."""
    return re.search(r"(?<!\w)" + re.escape(sym.name) + r"(?!\w)", answer) is not None


def predicted_symbols(answer: str, report: CitationReport, repo: RepoFiles, exclude_names: Iterable[str] = (),
                      gold: Iterable[str] = ()) -> set[str]:
    """Symbols the answer names AND that a cited range supports. Support is (a) the citation overlaps the symbol's
    definition, or (b) for gold symbols, a grounded cited line's text contains the name (a call site). Both naming
    and support are required so a symbol cannot be claimed by name-dropping or by citing a whole file.
    `exclude_names` (symbols the question itself names, e.g. the caller) are never counted as predictions."""
    excl = set(exclude_names)
    out: set[str] = set()
    for c in report.citations:
        for s in repo.symbols_in(c.path, c.start, c.end):
            if s.name not in excl and _mentioned(answer, s):
                out.add(s.qualified)
    gold_syms = [g for q in gold for g in repo.find_symbols(q)]
    if gold_syms:
        cited_lines = [(repo.normalize(c.path), n) for c in report.citations if c.grounded for n in range(c.start, min(c.end, c.start + 200) + 1)]
        texts = {(p, n): repo.line_text(p, n) for p, n in cited_lines}
        for s in gold_syms:
            if s.qualified in out or s.name in excl or not _mentioned(answer, s):
                continue
            if any(re.search(r"(?<![\w])" + re.escape(s.name) + r"(?![\w])", t) for t in texts.values()):
                out.add(s.qualified)
    return out


def resolve_gold_symbols(expected_symbols: Iterable[str], repo: RepoFiles) -> set[str]:
    gold: set[str] = set()
    for q in expected_symbols:
        hits = repo.find_symbols(q)
        gold.update(s.qualified for s in hits) if hits else gold.add(q)
    return gold


def question_names(question: str, repo: RepoFiles) -> set[str]:
    """Symbol names that appear verbatim in the question (they are givens, not answers)."""
    names = {s.name for s in repo.symbols}
    return {w for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", question) if w in names and len(w) > 2}


def symbol_score(answer: str, report: CitationReport, expected_symbols: Iterable[str], repo: RepoFiles,
                 question: str = "") -> float:
    """One gold symbol: any-of, discounted when many cited symbols are named. Several: F1 over qualified names."""
    expected = list(expected_symbols)
    gold = resolve_gold_symbols(expected, repo)
    if not gold:
        return 0.0
    excl = question_names(question, repo) - {q.rsplit(":", 1)[-1].split(".")[-1] for q in expected}
    pred = predicted_symbols(answer, report, repo, exclude_names=excl, gold=expected)
    if len(expected) == 1:
        return min(1.0, SINGLE_SYMBOL_PRECISION / max(len(pred), 1)) if pred & gold else 0.0
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
        if s and is_trivial_literal(g.expected_literal) and not cites_required(report, g.required_citations, repo):
            return 0.0, f"literal {g.expected_literal!r}: hit but no citation on the evidence lines"
        return s, f"literal {g.expected_literal!r}: {'hit' if s else 'miss'}"
    if g.expected_symbols:
        s = symbol_score(answer, report, g.expected_symbols, repo, task.question)
        return s, f"symbols {len(g.expected_symbols)} gold: {s:.2f}"
    if g.expected_paths:
        cited = {repo.normalize(c.path) for c in report.citations}
        s = path_score(cited, g.expected_paths, repo)
        if s and not any(mentions_path(answer, repo.normalize(p)) for p in g.expected_paths):
            return s * 0.5, f"paths {len(g.expected_paths)} gold: {s:.2f}, file not named in the answer (x0.5)"
        return s, f"paths {len(g.expected_paths)} gold: {s:.2f}"
    return 0.0, "no gold in task; nothing to verify"
