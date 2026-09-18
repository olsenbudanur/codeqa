"""Tool tests on the flask snapshot: happy path, bad path, oversize, caps, budget note. Offline (rg only)."""
from __future__ import annotations

import asyncio

import pytest
from tinker_cookbook.tool_use.types import ToolInput

from codeqa.agent.curation import Caps
from codeqa.agent.tools import RepoTools
from codeqa.shared import paths
from codeqa.shared.contracts import Span

REPO_ID = "pallets__flask__85c5d93"
pytestmark = pytest.mark.skipif(not (paths.repo_dir(REPO_ID) / "manifest.json").exists(), reason="flask snapshot missing")


def call(tool_obj, **kwargs) -> str:
    """Invoke a cookbook FunctionTool the way the env does: through run(ToolInput), so validation is exercised."""
    result = asyncio.run(tool_obj.run(ToolInput(arguments=kwargs)))
    return result.messages[0]["content"]


@pytest.fixture
def t() -> RepoTools:
    return RepoTools(REPO_ID)


def test_specs_are_the_five_tools(t):
    specs = t.specs()
    assert [s["name"] for s in specs] == ["overview", "find_symbol", "grep", "read_file", "list_dir"]
    assert all("properties" in s["parameters"] and s["description"] for s in specs)
    assert set(specs[3]["parameters"]["required"]) == {"path"}


def test_overview_root_dir_and_file(t):
    root = call(t.overview, path=".")
    assert "src/" in root and "tests/" in root
    d = call(t.overview, path="src/flask")
    assert d.startswith("src/flask/") and "app.py" in d and "Flask" in d
    f = call(t.overview, path="src/flask/app.py")
    assert "L81-L1536  class  class Flask(App):" in f
    bad = call(t.overview, path="src/flsk")
    assert bad.startswith("ERROR not_found") and "src/flask" in bad
    assert t.calls == 4 and t.errors == 1


def test_find_symbol_exact_ci_substring_filters(t):
    out = call(t.find_symbol, name="Flask")
    assert out.splitlines()[1].startswith("src/flask/app.py:L81-L1536  class  class Flask(App):")
    ci = call(t.find_symbol, name="flask", kind="class")
    assert "(case-insensitive)" in ci and "src/flask/app.py:L81" in ci
    dotted = call(t.find_symbol, name="Scaffold.route")
    assert "scaffold.py" in dotted and "[in Scaffold]" in dotted
    filtered = call(t.find_symbol, name="App", file_pattern="sansio")
    assert all("sansio" in ln for ln in filtered.splitlines()[1:] if ln and not ln.startswith("("))
    missing = call(t.find_symbol, name="zzz_not_a_symbol")
    assert missing.startswith("No symbol matching") and t.errors == 1


def test_find_symbol_validation_error_is_a_tool_message(t):
    out = call(t.find_symbol)  # missing required `name`
    assert "validation" in out.lower() or "error" in out.lower()


def test_find_symbol_cap():
    t = RepoTools(REPO_ID, caps=Caps(symbol_hits=2))
    out = call(t.find_symbol, name="get")
    assert "more; narrow" in out and len([l for l in out.splitlines() if ":L" in l]) == 2


def test_grep_hit_error_nomatch(t):
    out = call(t.grep, pattern=r"class Flask\(")
    assert "src/flask/app.py:L81: class Flask(App):" in out
    bad = call(t.grep, pattern="[")
    assert bad.startswith("ERROR bad_pattern")
    none = call(t.grep, pattern="zzzqqqxxx_nothing")
    assert none.startswith("No matches")
    ci = call(t.grep, pattern=r"CLASS FLASK\(")
    assert "(case-insensitive)" in ci
    assert t.errors == 2


def test_grep_caps_and_ranking():
    t = RepoTools(REPO_ID, caps=Caps(grep_hits=5, grep_files=2))
    out = call(t.grep, pattern="import", file_pattern="src/flask")
    hits = [l for l in out.splitlines() if ":L" in l]
    assert len(hits) == 5 and "more hits not shown" in out
    assert all(l.startswith("src/flask/") for l in hits)


def test_read_file_range_cap_and_errors(t):
    out = call(t.read_file, path="src/flask/app.py", start=81, end=85)
    assert out.splitlines()[0] == "src/flask/app.py:L81-L85"
    assert out.splitlines()[1] == "L81 | class Flask(App):"
    assert out.endswith("(total 1536 lines)")
    assert t.files_read == [Span(path="src/flask/app.py", start=81, end=85)]
    big = call(t.read_file, path="src/flask/app.py", start=1, end=1000)
    assert "continue from L151" in big and t.files_read[-1] == Span(path="src/flask/app.py", start=1, end=150)
    tail = call(t.read_file, path="src/flask/app.py", start=1530)
    assert "L1536 |" in tail and t.files_read[-1].end == 1536
    typo = call(t.read_file, path="src/flask/appp.py", start=1, end=5)
    assert typo.startswith("ERROR not_found") and "src/flask/app.py" in typo
    isdir = call(t.read_file, path="src/flask", start=1, end=5)
    assert isdir.startswith("ERROR is_directory")
    past = call(t.read_file, path="src/flask/app.py", start=5000, end=5010)
    assert past.startswith("ERROR bad_range")
    assert len(t.files_read) == 3 and t.errors == 3


def test_list_dir_cap_and_errors():
    t = RepoTools(REPO_ID, caps=Caps(list_entries=3))
    out = call(t.list_dir, path="src/flask")
    assert out.startswith("src/flask/") and "(+" in out and len(out.splitlines()) == 5
    assert call(t.list_dir, path="src/flask/app.py").startswith("ERROR is_file")
    assert call(t.list_dir, path="srcc").startswith("ERROR not_found")


def test_budget_note():
    t = RepoTools(REPO_ID, max_tool_calls=2)
    first = call(t.list_dir, path=".")
    assert first.endswith("[1 tool call remaining. Answer on your next turn.]")
    second = call(t.list_dir, path=".")
    assert second.endswith("[No tool calls remaining. Answer now.]")


def test_grep_hits_count_as_seen_lines(t):
    out = call(t.grep, pattern=r"class Flask\(")
    assert "src/flask/app.py:L81: class Flask(App):" in out
    assert Span(path="src/flask/app.py", start=81, end=81) in t.files_read
    # only shown hits are recorded: the cap bounds the number of spans
    t2 = RepoTools(REPO_ID, caps=Caps(grep_hits=3, grep_files=2))
    call(t2.grep, pattern="import")
    assert len(t2.files_read) == 3
