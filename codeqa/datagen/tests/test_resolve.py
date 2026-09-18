from codeqa.datagen.resolve import RepoIndex, candidate_identifiers, resolve_answer_spans, resolve_paths
from codeqa.shared.contracts import FileEntry, IndexSymbol, Manifest, Span

RID = "acme__proj__abc1234"


def _index() -> RepoIndex:
    m = Manifest(repo_id=RID, url="https://github.com/acme/proj", sha="abc1234" * 5 + "abcde", files=[
        FileEntry(path="src/proj/app.py", lang="python", lines=120, bytes=1),
        FileEntry(path="src/proj/auth/session.py", lang="python", lines=80, bytes=1),
        FileEntry(path="tests/test_app.py", lang="python", lines=40, bytes=1),
        FileEntry(path="tests/auth/test_session.py", lang="python", lines=40, bytes=1),
        FileEntry(path="docs/conf.py", lang="python", lines=10, bytes=1),
        FileEntry(path="src/proj/util/conf.py", lang="python", lines=10, bytes=1),
    ])
    syms = [
        IndexSymbol(name="App", kind="class", path="src/proj/app.py", start=10, end=90),
        IndexSymbol(name="run", kind="method", path="src/proj/app.py", start=40, end=55, parent="App"),
        IndexSymbol(name="validate_session", kind="function", path="src/proj/auth/session.py", start=20, end=35),
        IndexSymbol(name="run", kind="function", path="tests/test_app.py", start=1, end=5),
        IndexSymbol(name="setup", kind="function", path="docs/conf.py", start=1, end=5),
        IndexSymbol(name="setup", kind="function", path="src/proj/util/conf.py", start=1, end=5),
    ]
    return RepoIndex(m, syms)


def test_resolve_path_exact_basename_suffix():
    idx = _index()
    assert idx.resolve_path("src/proj/app.py") == "src/proj/app.py"
    assert idx.resolve_path("session.py") == "src/proj/auth/session.py"          # unique basename
    assert idx.resolve_path("proj/app.py") == "src/proj/app.py"                  # unique suffix
    assert idx.resolve_path("conf.py") is None                                   # ambiguous basename
    assert idx.resolve_path("util/conf.py") == "src/proj/util/conf.py"           # suffix disambiguates
    assert idx.resolve_path("nope.py") is None
    assert idx.resolve_path("`src/proj/app.py`") == "src/proj/app.py"


def test_clip_span():
    idx = _index()
    assert idx.clip_span(Span(path="src/proj/app.py", start=100, end=200)) == Span(path="src/proj/app.py", start=100, end=120)
    assert idx.clip_span(Span(path="src/proj/app.py", start=121, end=130)) is None
    assert idx.clip_span(Span(path="missing.py", start=1, end=2)) is None


def test_find_symbol_qualified_and_ambiguity():
    idx = _index()
    assert idx.resolve_symbol_span("App.run") == Span(path="src/proj/app.py", start=40, end=55)
    assert idx.resolve_symbol_span("proj.app.App.run") == Span(path="src/proj/app.py", start=40, end=55)
    assert idx.resolve_symbol_span("run") is None                                 # two defs, ambiguous
    assert idx.resolve_symbol_span("run", ["src/proj/app.py"]).start == 40         # path disambiguates
    assert idx.resolve_symbol_span("setup") is None
    assert idx.resolve_symbol_span("validate_session()") .start == 20


def test_candidate_identifiers_prefers_backticks():
    text = "In `App.run` (src/proj/app.py) we call `validate_session()` unless `self` is None. See tests/test_app.py."
    assert candidate_identifiers(text) == ["App.run", "validate_session"]
    bare = candidate_identifiers("The validate_session helper and the SessionStore class.", include_bare=True)
    assert "validate_session" in bare and "SessionStore" in bare


def test_resolve_answer_spans_orders_expected_paths_first():
    idx = _index()
    spans = resolve_answer_spans(idx, "`validate_session` is invoked by `App.run` and `App`.", ["src/proj/app.py"])
    assert spans[0].path == "src/proj/app.py" and spans[0].start == 40      # method before its class (shorter)
    assert {(s.path, s.start) for s in spans} == {("src/proj/app.py", 40), ("src/proj/app.py", 10), ("src/proj/auth/session.py", 20)}


def test_resolve_paths_reports_unresolved():
    ok, bad = resolve_paths(_index(), ["app.py", "session.py", "ghost.py", "src/proj/app.py"])
    assert ok == ["src/proj/app.py", "src/proj/auth/session.py"] and bad == ["ghost.py"]


def test_resolve_path_strips_checkout_prefix_and_uses_symbol_hints():
    idx = _index()
    assert idx.resolve_path("workspace/src/proj/app.py") == "src/proj/app.py"
    assert idx.resolve_path("checkout/proj/auth/session.py") == "src/proj/auth/session.py"
    assert idx.resolve_path("conf.py", ["setup"]) is None                        # both candidates define setup
    idx2 = _index(); idx2.by_name["setup"] = [h for h in idx2.by_name["setup"] if h.path == "docs/conf.py"]
    assert idx2.resolve_path("conf.py", ["setup"]) == "docs/conf.py"


def test_find_symbol_falls_back_to_owning_class():
    idx = _index()
    assert idx.resolve_symbol_span("App._supports_thing") == Span(path="src/proj/app.py", start=10, end=49)   # class clipped to 40 lines
