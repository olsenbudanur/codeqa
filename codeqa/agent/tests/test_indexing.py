"""Offline tests for the indexing layer on the flask snapshot (no network, no LLM)."""
from __future__ import annotations

import pytest

from codeqa.agent.indexing.index import load_symbols, python_symbols
from codeqa.agent.indexing.repomap import build_map, count_tokens, render
from codeqa.agent.indexing.snapshot import load_manifest
from codeqa.agent.indexing.strip_docstrings import strip_python_docstrings
from codeqa.shared import paths

REPO_ID = "pallets__flask__85c5d93"
needs_flask = pytest.mark.skipif(not (paths.repo_dir(REPO_ID) / "manifest.json").exists(), reason="flask snapshot missing")

SRC = b'''"""Module doc."""
import os


class A:
    """Class doc
    spanning lines."""

    def f(self):
        """Method doc."""
        return 1

    def g(self):
        # no docstring
        s = "not a docstring"
        return s


def h():
    \'\'\'single quotes\'\'\'
    return "keep me"
'''


def test_strip_docstrings_preserves_lines_and_code():
    out, n = strip_python_docstrings(SRC)
    assert n == 4
    assert out.count(b"\n") == SRC.count(b"\n")
    text = out.decode()
    for gone in ("Module doc", "Class doc", "Method doc", "single quotes"):
        assert gone not in text
    for kept in ('"not a docstring"', '"keep me"', "# no docstring"):
        assert kept in text
    compile(text, "x.py", "exec")  # still valid python
    syms = python_symbols("x.py", out)
    assert {(s.name, s.start) for s in syms} == {("A", 5), ("f", 9), ("g", 13), ("h", 19)}


@needs_flask
def test_map_fits_budget_and_mentions_core_files():
    m = load_manifest(REPO_ID)
    syms = load_symbols(REPO_ID)
    text = build_map(m, syms, {}, max_tokens=3000, write=False)
    assert count_tokens(text) <= 3000
    assert "src/" in text and "app.py" in text and "Flask" in text
    tiny = build_map(m, syms, {}, max_tokens=300, write=False)
    assert count_tokens(tiny) <= 300


@needs_flask
def test_render_levels_monotone():
    m = load_manifest(REPO_ID)
    syms = load_symbols(REPO_ID)
    sizes = [count_tokens(render(m, syms, {}, level=l, max_depth=99)) for l in range(4)]
    assert sizes == sorted(sizes, reverse=True), sizes
