import textwrap

import pytest

from codeqa.agent.indexing.index import python_symbols
from codeqa.datagen.sources import structural
from codeqa.shared import paths
from codeqa.shared.contracts import FileEntry, Manifest

FILES = {
    "pkg/__init__.py": "",
    "pkg/core.py": '''
        from pkg.util import helper_fn, CONST_A
        from . import util

        MAX_RETRIES = 5
        GREETING = "hello"
        BAD = f"{MAX_RETRIES}"

        class Base:
            """Base class for all handlers that process incoming records in a pipeline stage."""
            def process_record(self, record, strict=False):
                """Handle one record and return whether it was accepted by this stage of the pipeline."""
                return helper_fn(record) and self.check_record(record) and util.other_fn(record)

            def check_record(self, record):
                return True

        class AlphaHandler(Base):
            pass

        class BetaHandler(Base):
            pass
    ''',
    "pkg/util.py": '''
        CONST_A = 300

        def helper_fn(x, mode="rb"):
            """Return a normalized copy of the record so that downstream stages see consistent keys."""
            return x

        def other_fn(x):
            return x
    ''',
    "pkg/extra.py": '''
        from pkg import util
        from pkg.core import Base

        class GammaHandler(Base):
            pass
    ''',
    "tests/test_core.py": "from pkg.core import Base\nMAX_RETRIES = 7\n",
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    rid = "acme__pipe__abc1234"
    root = tmp_path / "repos" / rid
    monkeypatch.setattr(paths, "REPOS", tmp_path / "repos")
    monkeypatch.setattr(paths, "INDEX", tmp_path / "index")
    files, syms = [], []
    for rel, body in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        text = textwrap.dedent(body).lstrip("\n")
        p.write_text(text)
        files.append(FileEntry(path=rel, lang="python", lines=text.count("\n") + 1, bytes=len(text)))
        syms.extend(python_symbols(rel, text.encode()))
    return Manifest(repo_id=rid, url="u", sha="a" * 40, files=files), syms


def test_facts(repo):
    m, syms = repo
    f = structural.build_facts(m, syms)
    assert f.constants["MAX_RETRIES"] == [("pkg/core.py", 4, "5")]       # tests/ excluded
    assert f.constants["GREETING"][0][2] == "hello" and "BAD" not in f.constants
    assert f.module_of_path["pkg/core.py"] == "pkg.core"
    assert {p for p, _ in f.importers["pkg/util.py"]} == {"pkg/core.py", "pkg/extra.py"}
    handle = next(x for x in f.funcs if x.qual == "Base.process_record")
    assert handle.defaults == [("strict", "False")]
    assert set(handle.callees) == {"pkg/util.py:helper_fn", "pkg/core.py:Base.check_record", "pkg/util.py:other_fn"}
    assert any(s.name == "Base" and bases == [] for s, _, bases in f.classes)
    assert sorted(s.name for s, _, b in f.classes if b == ["Base"]) == ["AlphaHandler", "BetaHandler", "GammaHandler"]


def test_generate_shapes(repo):
    m, syms = repo
    tasks, stats = structural.generate(m, syms, m.repo_id + "__nodoc", per_type=4)
    by = {}
    for t in tasks:
        by.setdefault(t.task_type, []).append(t)
    assert all(t.repo_id.endswith("__nodoc") for t in tasks)
    vals = {t.grading.expected_literal for t in by["value"]}
    assert vals >= {"5", "hello", "rb", "False"} - {None} or len(vals) >= 3
    assert "None" not in vals
    sub = next(t for t in by["enumerate"] if t.source_id.startswith("subclasses:"))
    assert sorted(sub.grading.expected_symbols) == ["pkg/core.py:AlphaHandler", "pkg/core.py:BetaHandler", "pkg/extra.py:GammaHandler"]
    imps = [t for t in by["enumerate"] if t.source_id.startswith("importers:")]
    assert imps and all("`pkg" in t.question for t in imps)
    util_imp = next(t for t in imps if t.source_id == "importers:pkg/util.py")
    assert util_imp.grading.expected_paths == ["pkg/core.py", "pkg/extra.py"]
    tr = next(t for t in by["trace"] if t.source_id.endswith("Base.process_record"))
    assert len(tr.grading.expected_symbols) == 3 and tr.grading.required_citations[0].path == "pkg/core.py"
    loc = by["locate"]
    assert loc and all(t.question == "" and t.grading.reference_answer for t in loc)
    assert {t.grading.expected_symbols[0] for t in loc} >= {"pkg/core.py:Base", "pkg/util.py:helper_fn"}   # process_record has no docstring >= 60 chars? it does; Base too


def test_leak_tokens():
    toks = structural.name_tokens("SecureCookieSession.save_session", "sessions")
    assert {"secure", "cookie", "session", "save", "sessions", "save_session"} <= toks
    assert structural.contains_leak("Where is the cookie signed?", toks) == "cookie"
    assert structural.contains_leak("Where is the browser state persisted between requests?", toks) is None
