"""Citation checks: parse, exist, grounded (read this episode), anchors a gold symbol (C7 check_citations)."""
from __future__ import annotations

from collections.abc import Iterable

from codeqa.grader.repo import RepoFiles, load_repo
from codeqa.shared.contracts import CITATION_RE, Citation, CitationReport, Span


def parse_citations(answer: str) -> list[Citation]:
    """All [path:Lstart-Lend] citations in the answer, deduplicated, in order. Reversed ranges are kept as-is
    (they will fail `exists`)."""
    seen: set[tuple[str, int, int]] = set()
    out: list[Citation] = []
    for m in CITATION_RE.finditer(answer):
        path, start = m.group(1), int(m.group(2))
        end = int(m.group(3)) if m.group(3) else start
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(path=path, start=start, end=end))
    return out


def read_lines_by_path(files_read: Iterable[Span], normalize=lambda p: p) -> dict[str, set[int]]:
    """Union of every line read, per path. Grounding is per line, so re-reading never helps and
    a cited range must be fully covered."""
    cov: dict[str, set[int]] = {}
    for s in files_read:
        cov.setdefault(normalize(s.path), set()).update(range(s.start, s.end + 1))
    return cov


def redundant_read_lines(files_read: Iterable[Span]) -> int:
    """Lines read more than once (a read of a range already covered counts every overlapping line)."""
    seen: dict[str, set[int]] = {}
    dup = 0
    for s in files_read:
        lines = set(range(s.start, s.end + 1))
        have = seen.setdefault(s.path, set())
        dup += len(lines & have)
        have |= lines
    return dup


def check_citations(answer: str, files_read: list[Span], repo_id: str,
                    expected_symbols: Iterable[str] = (), repo: RepoFiles | None = None) -> CitationReport:
    """Per-citation flags. `exists`: file present and range inside it. `grounded`: every cited line was read
    this episode. `anchors_symbol`: the citation contains the definition line of a gold symbol (index lookup)."""
    repo = repo or load_repo(repo_id)
    cov = read_lines_by_path(files_read, repo.normalize)
    gold_defs: list[tuple[str, int]] = []
    for q in expected_symbols:
        for s in repo.find_symbols(q):
            gold_defs.append((s.path, s.start))
    cits = parse_citations(answer)
    for c in cits:
        path = repo.normalize(c.path)
        c.exists = c.start <= c.end and repo.exists(path, c.start, c.end)
        lines = set(range(c.start, c.end + 1))
        c.grounded = bool(lines) and c.start <= c.end and lines <= cov.get(path, set())
        c.anchors_symbol = any(p == path and c.start <= line <= c.end for p, line in gold_defs)
    return CitationReport(citations=cits)


def cited_paths(report: CitationReport, repo: RepoFiles) -> set[str]:
    return {repo.normalize(c.path) for c in report.citations}
