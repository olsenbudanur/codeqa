"""Resolve the things upstream answers mention (paths, bare filenames, symbol names) against a snapshot (C1) + index (C2).

Every task record must have paths that exist and line ranges that are real at the pinned commit; this is the one
place that logic lives for all sources.
"""
from __future__ import annotations

import re
from collections import defaultdict

from codeqa.shared.contracts import IndexSymbol, Manifest, Span

IDENT_RE = re.compile(r"`([A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*)(?:\(\))?`")          # `foo`, `Cls.meth`, `pkg.mod.fn()`
BARE_IDENT_RE = re.compile(r"(?<![\w./])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+|[a-z]+(?:_[a-z0-9]+)+)(?:\(\))?(?![\w/])")
PY_PATH_RE = re.compile(r"(?<![\w/])((?:[\w.\-]+/)*[\w.\-]+\.(?:py|pyx|pyi))(?![\w/])")
NOISE_IDENTS = {"self", "None", "True", "False", "cls", "args", "kwargs", "e.g", "i.e", "etc"}
CLASS_SPAN_LINES = 40      # a citation for "which class" needs the header and the start of the body, not 2,000 lines


def symbol_span(sym: IndexSymbol) -> Span:
    """Evidence span for a symbol; class bodies are clipped to CLASS_SPAN_LINES."""
    end = min(sym.end, sym.start + CLASS_SPAN_LINES - 1) if sym.kind == "class" else sym.end
    return Span(path=sym.path, start=sym.start, end=end)


class RepoIndex:
    """Fast lookups over one repo's manifest + symbols."""

    def __init__(self, manifest: Manifest, symbols: list[IndexSymbol]):
        self.repo_id = manifest.repo_id
        self.lines: dict[str, int] = {f.path: f.lines for f in manifest.files}
        self.paths: set[str] = set(self.lines)
        self.by_basename: dict[str, list[str]] = defaultdict(list)
        for p in self.paths:
            self.by_basename[p.rsplit("/", 1)[-1]].append(p)
        self.symbols = symbols
        self.by_name: dict[str, list[IndexSymbol]] = defaultdict(list)
        self.by_qualified: dict[str, list[IndexSymbol]] = defaultdict(list)
        for s in symbols:
            self.by_name[s.name].append(s)
            if s.parent:
                self.by_qualified[f"{s.parent}.{s.name}"].append(s)

    # -- paths ---------------------------------------------------------------
    def resolve_path(self, p: str, hint_symbols: list[str] | None = None) -> str | None:
        """Exact path, else unique basename, else unique suffix match ('flask/app.py' -> 'src/flask/app.py'),
        else a stripped-prefix match; an ambiguous basename is settled by which file defines a hinted symbol."""
        p = p.strip().strip("`'\"").lstrip("./")
        if not p:
            return None
        if p in self.paths:
            return p
        base = p.rsplit("/", 1)[-1]
        cands = self.by_basename.get(base, [])
        if len(cands) == 1:
            return cands[0]
        if "/" in p:
            suffix = [c for c in cands if c.endswith("/" + p)]
            if len(suffix) == 1:
                return suffix[0]
            # annotators sometimes prefix their checkout dir ('workspace/conan/x.py'): drop leading parts one by one
            parts = p.split("/")
            for i in range(1, len(parts) - 1):
                sub = "/".join(parts[i:])
                if sub in self.paths:
                    return sub
                suffix = [c for c in cands if c.endswith("/" + sub)]
                if len(suffix) == 1:
                    return suffix[0]
        if len(cands) > 1 and hint_symbols:
            # ambiguous basename: pick the one file that defines a symbol the answer names
            defining = {c for c in cands for name in hint_symbols for h in self.by_name.get(name.split(".")[-1], []) if h.path == c}
            if len(defining) == 1:
                return defining.pop()
        return None

    def clip_span(self, span: Span) -> Span | None:
        """Drop spans whose file is missing or whose start is past EOF; clip ends to file length."""
        n = self.lines.get(span.path)
        if n is None or span.start > n:
            return None
        return Span(path=span.path, start=span.start, end=min(span.end, n))

    # -- symbols -------------------------------------------------------------
    def find_symbol(self, name: str, paths: list[str] | None = None) -> list[IndexSymbol]:
        """'a.b.Cls.meth' -> try 'Cls.meth' (parent.name), then 'meth'. Restrict to paths when given and non-empty there."""
        name = name.strip().strip("`").removesuffix("()")
        parts = name.split(".")
        hits: list[IndexSymbol] = []
        if len(parts) >= 2:
            hits = list(self.by_qualified.get(".".join(parts[-2:]), []))
        if not hits:
            hits = list(self.by_name.get(parts[-1], []))
        if not hits and len(parts) >= 2:
            hits = list(self.by_name.get(parts[0], []))      # 'GPT2Model._some_attr' -> the class that owns the attribute
        if paths:
            in_paths = [h for h in hits if h.path in paths]
            if in_paths:
                hits = in_paths
        return hits

    def resolve_symbol_span(self, name: str, paths: list[str] | None = None) -> Span | None:
        """Unique symbol -> its span. Ambiguous (several definitions, none disambiguated by path) -> None."""
        hits = self.find_symbol(name, paths)
        if not hits:
            return None
        if len(hits) > 1:
            # methods with the same parent.name in different files (or duplicates) are ambiguous
            keys = {(h.path, h.start, h.end) for h in hits}
            if len(keys) > 1:
                return None
        return symbol_span(hits[0])


def candidate_identifiers(text: str, include_bare: bool = False) -> list[str]:
    """Identifiers an answer mentions, backticked first (most reliable), then optionally bare snake/Camel/dotted names."""
    out: list[str] = []
    seen: set[str] = set()
    for m in IDENT_RE.finditer(text):
        ident = m.group(1)
        if ident not in seen and ident not in NOISE_IDENTS and not ident.endswith(".py"):
            seen.add(ident); out.append(ident)
    if include_bare:
        for m in BARE_IDENT_RE.finditer(text):
            ident = m.group(1)
            if ident not in seen and ident not in NOISE_IDENTS and not ident.endswith(".py"):
                seen.add(ident); out.append(ident)
    return out


def resolve_paths(idx: RepoIndex, raw_paths: list[str]) -> tuple[list[str], list[str]]:
    """-> (resolved, unresolved), resolved unique and sorted."""
    ok: set[str] = set()
    bad: list[str] = []
    for p in raw_paths:
        r = idx.resolve_path(p)
        (ok.add(r) if r else bad.append(p))
    return sorted(ok), bad


def resolve_answer_spans(idx: RepoIndex, text: str, expected_paths: list[str], max_spans: int = 5) -> list[Span]:
    """Spans of the symbols an answer names, preferring symbols defined in the expected paths.

    Backticked identifiers first; bare identifiers only when backticks yield nothing. Unique matches only.
    """
    spans: list[Span] = []
    seen: set[tuple[str, int, int]] = set()

    def add(cands: list[str]) -> None:
        for ident in cands:
            sp = idx.resolve_symbol_span(ident, expected_paths)
            if sp and (sp.path, sp.start, sp.end) not in seen:
                seen.add((sp.path, sp.start, sp.end)); spans.append(sp)
            if len(spans) >= max_spans:
                return

    add(candidate_identifiers(text))
    if not spans:
        add(candidate_identifiers(text, include_bare=True))
    # prefer spans in expected paths, then shorter spans (a method beats its whole class)
    spans.sort(key=lambda s: (s.path not in expected_paths, s.end - s.start))
    return spans[:max_spans]
