from codeqa.grader.citations import check_citations, parse_citations, read_lines_by_path
from codeqa.shared.contracts import Span

S = "src/miniapp/auth/session.py"


def test_parse_dedupes_and_single_line():
    cits = parse_citations(f"a [{S}:L35-L42] b [{S}:L35-L42] c [{S}:L7] d `x.py:L3` e")
    assert [(c.path, c.start, c.end) for c in cits] == [(S, 35, 42), (S, 7, 7)]


def test_exists_missing_file_beyond_eof_reversed(repo):
    r = check_citations(f"[{S}:L35-L42] [{S}:L40-L900] [src/nope.py:L1-L2] [{S}:L42-L35]", [Span(path=S, start=1, end=48)], repo.repo_id, repo=repo)
    assert [c.exists for c in r.citations] == [True, False, False, False]
    assert not r.all_exist


def test_grounded_needs_every_cited_line_read(repo):
    reads = [Span(path=S, start=30, end=38), Span(path=S, start=39, end=42)]
    r = check_citations(f"[{S}:L35-L42] [{S}:L28-L42]", reads, repo.repo_id, repo=repo)
    assert [c.grounded for c in r.citations] == [True, False]


def test_anchors_symbol_requires_definition_line(repo):
    reads = [Span(path=S, start=1, end=48)]
    r = check_citations(f"[{S}:L35-L42] [{S}:L36-L42]", reads, repo.repo_id, expected_symbols=[f"{S}:validate_session"], repo=repo)
    assert [c.anchors_symbol for c in r.citations] == [True, False]


def test_normalizes_dot_slash_paths(repo):
    r = check_citations(f"[./{S}:L35-L42]", [Span(path=S, start=35, end=42)], repo.repo_id, repo=repo)
    assert r.all_exist and r.all_grounded


def test_read_lines_union():
    cov = read_lines_by_path([Span(path="a", start=1, end=3), Span(path="a", start=3, end=5)])
    assert cov["a"] == {1, 2, 3, 4, 5}


def test_unique_basename_resolves(repo):
    assert repo.normalize("session.py") == S
    assert repo.normalize("auth/session.py") == S
    assert repo.normalize("__init__.py") == "__init__.py"          # ambiguous: left alone (and will not exist)
    r = check_citations("[session.py:L35-L42]", [Span(path=S, start=35, end=42)], repo.repo_id, repo=repo)
    assert r.all_exist and r.all_grounded
