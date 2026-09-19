"""The five read-only tools over a snapshot + index, as cookbook `@tool` methods on a stateful class. (C3)

One `RepoTools` per episode. It counts calls, records every line range shown to the model (for the
grounding check) and appends the budget warning. Output formats are the contract; caps are config.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated

from tinker_cookbook.tool_use.tools import simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult

from codeqa.agent.curation import (DEFAULT_CAPS, Caps, budget_note, cap_lines, is_test_path, match_file_pattern,
                                   not_found, rank_paths, rg_glob, truncate)
from codeqa.agent.indexing.index import load_symbols
from codeqa.agent.indexing.snapshot import load_manifest
from codeqa.agent.indexing.summaries import load_summaries
from codeqa.shared import paths
from codeqa.shared.contracts import IndexSymbol, Span

RG_TIMEOUT = 10.0
RG_BIN = shutil.which("rg") or next((p for p in ("/opt/homebrew/bin/rg", "/usr/local/bin/rg") if Path(p).is_file()), None)


class RepoTools:
    def __init__(self, repo_id: str, max_tool_calls: int | None = None, caps: Caps = DEFAULT_CAPS):
        self.repo_id = repo_id
        self.root: Path = paths.repo_dir(repo_id)
        self.manifest = load_manifest(repo_id)
        self.symbols: list[IndexSymbol] = load_symbols(repo_id)
        self.summaries: dict[str, str] = load_summaries(repo_id)
        self.caps = caps
        self.max_tool_calls = max_tool_calls
        self.calls = 0
        self.errors = 0
        self.files_read: list[Span] = []
        self._lines: dict[str, int] = {f.path: f.lines for f in self.manifest.files}
        self._paths: list[str] = [f.path for f in self.manifest.files]
        self._dirs: set[str] = {"."}
        for p in self._paths:
            parts = p.split("/")
            for i in range(1, len(parts)):
                self._dirs.add("/".join(parts[:i]))
        self._by_name: dict[str, list[IndexSymbol]] = defaultdict(list)
        self._by_lower: dict[str, list[IndexSymbol]] = defaultdict(list)
        self._by_path: dict[str, list[IndexSymbol]] = defaultdict(list)
        for s in self.symbols:
            self._by_name[s.name].append(s)
            self._by_lower[s.name.lower()].append(s)
            self._by_path[s.path].append(s)

    # ------------------------------------------------------------------ plumbing
    DEFAULT_TOOLS = ("overview", "find_symbol", "grep", "read_file", "list_dir")

    def tools(self, names: tuple[str, ...] | list[str] | None = None) -> list:
        """The cookbook tool objects for a variant; default = the five. `bash` exists but is opt-in (agent/variants.py)."""
        by_name = {"overview": self.overview, "find_symbol": self.find_symbol, "grep": self.grep,
                   "read_file": self.read_file, "list_dir": self.list_dir, "bash": self.bash}
        return [by_name[n] for n in (names or self.DEFAULT_TOOLS)]

    def specs(self, names: tuple[str, ...] | list[str] | None = None) -> list[dict]:
        return [t.to_spec() for t in self.tools(names)]

    def _norm(self, path: str) -> str:
        p = (path or ".").strip().replace("\\", "/")
        p = re.sub(r"^\./", "", p).rstrip("/")
        return p or "."

    def _finish(self, text: str, error: bool = False) -> ToolResult:
        self.calls += 1
        if error:
            self.errors += 1
        return simple_tool_result(budget_note(text, self.calls, self.max_tool_calls))

    # ------------------------------------------------------------------ tools
    @tool
    async def overview(self, path: Annotated[str, "Directory or file path, repo-relative. '.' for the repo root."] = ".") -> ToolResult:
        """Summary of a directory (what it is for, children with one-line summaries) or a file (its symbols with line ranges). Start here."""
        p = self._norm(path)
        cap = self.caps.overview_lines
        if p in self._lines:  # a file
            lines = [f"{p}  ({self._lines[p]} lines)"]
            if self.summaries.get(p):
                lines.append(self.summaries[p])
            syms = self._by_path.get(p, [])
            for s in syms:
                indent = "  " if s.parent else ""
                lines.append(f"{indent}L{s.start}-L{s.end}  {s.kind}  {truncate(s.signature, self.caps.signature_chars)}")
            if not syms:
                lines.append("(no indexed symbols; use read_file)")
            return self._finish("\n".join(cap_lines(lines, cap, "(+{n} more symbols; use find_symbol or read_file)")))
        if p not in self._dirs:
            return self._finish(not_found("path", p, self._paths + sorted(self._dirs)), error=True)
        lines = [f"{p}/" if p != "." else f"{self.repo_id} (repo root)"]
        if self.summaries.get(p):
            lines.append(self.summaries[p])
        subdirs = sorted(d for d in self._dirs if d != "." and (d.rsplit("/", 1)[0] if "/" in d else ".") == p)
        files = [f for f in self._paths if (f.rsplit("/", 1)[0] if "/" in f else ".") == p]
        for d in rank_paths(subdirs):
            summ = self.summaries.get(d, "")
            first = summ.split(". ", 1)[0] if summ else ""
            lines.append(f"  {d.rsplit('/', 1)[-1]}/  {truncate(first, 90)}".rstrip())
        for f in rank_paths(files):
            top = [s.name for s in self._by_path.get(f, []) if s.parent is None][:6]
            more = len([s for s in self._by_path.get(f, []) if s.parent is None]) - len(top)
            names = (", ".join(top) + (f", +{more}" if more > 0 else "")) if top else ""
            lines.append(f"  {f.rsplit('/', 1)[-1]}  ({self._lines[f]} lines)" + (f"  {names}" if names else ""))
        return self._finish("\n".join(cap_lines(lines, cap, "(+{n} more entries; use list_dir)")))

    @tool
    async def find_symbol(self, name: Annotated[str, "Class, function or method name. 'Class.method' also works."],
                          kind: Annotated[str | None, "Optional: function | class | method"] = None,
                          file_pattern: Annotated[str | None, "Optional path glob or substring, e.g. 'src/flask' or '*/app.py'"] = None) -> ToolResult:
        """Find where a class, function or method is defined: path, line range, kind, signature. Exact name first, then case-insensitive."""
        raw = (name or "").strip().strip("`'\"()")
        parent = None
        if "." in raw:
            parent, raw = raw.rsplit(".", 1)
        hits = list(self._by_name.get(raw, []))
        note = ""
        if not hits:
            hits = list(self._by_lower.get(raw.lower(), []))
            note = " (case-insensitive)" if hits else ""
        if not hits and len(raw) >= 4:
            hits = [s for s in self.symbols if raw.lower() in s.name.lower()]
            note = " (substring match)" if hits else ""
        if parent:
            exact_parent = [s for s in hits if s.parent == parent]
            hits = exact_parent or hits
        if kind:
            k = kind.strip().lower()
            if k in ("function", "class", "method"):
                hits = [s for s in hits if s.kind == k]
        if file_pattern:
            hits = [s for s in hits if match_file_pattern(s.path, file_pattern)]
        if not hits:
            return self._finish(f"No symbol matching {name!r}" + (f" with kind={kind}" if kind else "") + (f" in {file_pattern!r}" if file_pattern else "")
                                + ". Try grep with a distinctive substring, or overview on a likely directory.", error=True)
        order = {"class": 0, "function": 1, "method": 2}
        hits.sort(key=lambda s: (s.name != raw, is_test_path(s.path), order[s.kind], s.path.count("/"), s.path, s.start))
        cap = self.caps.symbol_hits
        lines = [f"{s.path}:L{s.start}-L{s.end}  {s.kind}  {truncate(s.signature, self.caps.signature_chars)}"
                 + (f"  [in {s.parent}]" if s.parent else "") for s in hits[:cap]]
        if len(hits) > cap:
            lines.append(f"(+{len(hits) - cap} more; narrow with kind or file_pattern)")
        return self._finish(f"{len(hits)} match(es){note}:\n" + "\n".join(lines))

    @tool
    async def grep(self, pattern: Annotated[str, "Regular expression (ripgrep syntax). Use a distinctive literal."],
                   file_pattern: Annotated[str | None, "Optional path glob or substring to restrict files, e.g. 'src/' or '*.py'"] = None) -> ToolResult:
        """Search file contents with a regex; hits come back as path:Lnn: text, grouped by file, source files first. Fallback when you do not know the name."""
        pat = str(pattern or "").strip()
        if not pat:
            return self._finish("ERROR bad_pattern: empty pattern.", error=True)
        hits, err = await self._rg(pat, file_pattern, ignore_case=False)
        note = ""
        if err:
            return self._finish(f"ERROR bad_pattern: {truncate(err, 200)}", error=True)
        if not hits:
            hits, _ = await self._rg(pat, file_pattern, ignore_case=True)
            note = " (case-insensitive)" if hits else ""
        if not hits:
            return self._finish(f"No matches for /{pat}/" + (f" in {file_pattern!r}" if file_pattern else "")
                                + ". Try a shorter literal, drop file_pattern, or use find_symbol.", error=True)
        by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for path, line, text in hits:
            by_file[path].append((line, text))
        files = rank_paths(list(by_file))
        # files with the most hits within the same rank tier first
        files.sort(key=lambda p: (is_test_path(p), -len(by_file[p])))
        total = len(hits)
        shown_files = files[: self.caps.grep_files]
        seen_texts: set[str] = set()
        lines: list[str] = []
        shown = 0
        collapsed = 0
        for f in shown_files:
            if shown >= self.caps.grep_hits:
                break
            file_lines = []
            for ln, text in sorted(by_file[f]):
                key = text.strip()
                if key in seen_texts:
                    collapsed += 1
                    continue
                seen_texts.add(key)
                file_lines.append(f"{f}:L{ln}: {truncate(text.strip(), self.caps.line_chars)}")
                # a grep hit shows the model that line's content, so citing it is grounded (decisions.md #7)
                self.files_read.append(Span(path=f, start=ln, end=ln))
                shown += 1
                if shown >= self.caps.grep_hits:
                    break
            if file_lines:
                lines.extend(file_lines)
        footer = []
        if total > shown:
            footer.append(f"({total - shown} more hits not shown across {len(files)} files; narrow with file_pattern or a longer literal)")
        elif len(files) > len(shown_files):
            footer.append(f"(+{len(files) - len(shown_files)} more files)")
        if collapsed:
            footer.append(f"({collapsed} duplicate lines collapsed)")
        return self._finish(f"{total} hit(s) in {len(files)} file(s){note}:\n" + "\n".join(lines) + ("\n" + " ".join(footer) if footer else ""))

    async def _rg(self, pattern: str, file_pattern: str | None, ignore_case: bool) -> tuple[list[tuple[str, int, str]], str | None]:
        """ripgrep when installed (fast), otherwise a pure-Python scan over the manifest files (same output)."""
        if RG_BIN is None:
            return await asyncio.to_thread(self._py_grep, pattern, file_pattern, ignore_case)
        args = [RG_BIN, "--json", "--max-count", "40", "--max-columns", "400", "--no-messages"]
        if ignore_case:
            args.append("-i")
        g = rg_glob(file_pattern)
        if g:
            args += ["-g", g]
        args += ["-e", pattern, "."]
        try:
            proc = await asyncio.create_subprocess_exec(*args, cwd=self.root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(proc.communicate(), timeout=RG_TIMEOUT)
        except asyncio.TimeoutError:
            return [], "search timed out; use a more specific pattern"
        except FileNotFoundError:
            return [], "ripgrep (rg) is not installed"
        if proc.returncode == 2:
            return [], err.decode(errors="replace").strip() or "invalid regex"
        hits: list[tuple[str, int, str]] = []
        for raw in out.decode(errors="replace").splitlines():
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "match":
                continue
            d = ev["data"]
            path = d["path"].get("text", "")
            path = re.sub(r"^\./", "", path)
            text = d["lines"].get("text", "")
            if path in self._lines:
                hits.append((path, int(d["line_number"]), text.rstrip("\n")))
        return hits, None

    def _py_grep(self, pattern: str, file_pattern: str | None, ignore_case: bool) -> tuple[list[tuple[str, int, str]], str | None]:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            return [], f"invalid regex: {e}"
        hits: list[tuple[str, int, str]] = []
        for path in self._paths:
            if file_pattern and not match_file_pattern(path, file_pattern):
                continue
            try:
                lines = (self.root / path).read_text(errors="replace").splitlines()
            except OSError:
                continue
            n = 0
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append((path, i, line[:400]))
                    n += 1
                    if n >= 40:
                        break
            if len(hits) >= 2000:
                break
        return hits, None

    @tool
    async def read_file(self, path: Annotated[str, "Repo-relative file path"],
                        start: Annotated[int, "First line, 1-based"] = 1,
                        end: Annotated[int | None, "Last line, inclusive. At most 150 lines per call."] = None) -> ToolResult:
        """Read a line range of a file; lines come back as 'L41 | code' so you can cite them. Read ranges, not whole files."""
        p = self._norm(path)
        if p not in self._lines:
            if p in self._dirs:
                return self._finish(f"ERROR is_directory: {p!r}. Use list_dir or overview.", error=True)
            return self._finish(not_found("file", p, self._paths), error=True)
        total = self._lines[p]
        cap = self.caps.read_lines
        try:
            s = max(1, int(start or 1))
            e = int(end) if end is not None else s + cap - 1
        except (TypeError, ValueError):
            return self._finish("ERROR bad_range: start and end must be integers.", error=True)
        if e < s:
            s, e = e, s
        if s > total:
            return self._finish(f"ERROR bad_range: {p} has {total} lines; start={s} is past the end.", error=True)
        truncated = False
        if e - s + 1 > cap:
            e, truncated = s + cap - 1, True
        e = min(e, total)
        src = (self.root / p).read_text(errors="replace").splitlines()
        body = "\n".join(f"L{i} | {truncate(src[i - 1], self.caps.line_chars)}" for i in range(s, e + 1))
        self.files_read.append(Span(path=p, start=s, end=e))
        footer = f"(total {total} lines)"
        if truncated:
            footer = f"(range cut to {cap} lines; continue from L{e + 1}; total {total} lines)"
        return self._finish(f"{p}:L{s}-L{e}\n{body}\n{footer}")

    @tool
    async def bash(self, command: Annotated[str, "One read-only shell command, run in the repository root. Pipes are fine. "
                                             "Show line numbers for anything you will cite: grep -rn PATTERN DIR, or nl -ba FILE | sed -n 'A,Bp'."]) -> ToolResult:
        """Run one read-only shell command in the repo root (ls, find, grep -n, nl, sed -n, head, tail, wc). No writes, no cd, no .. or absolute paths."""
        from codeqa.agent import shell
        out, err = await shell.run(str(command or ""), cwd=self.root, repo_id=self.repo_id)
        if err:
            return self._finish(f"ERROR {err}", error=True)
        if not out.strip():
            out = "(no output)"
        self.files_read.extend(shell.seen_spans(str(command), out, self._lines))
        return self._finish(out)

    @tool
    async def list_dir(self, path: Annotated[str, "Directory path, repo-relative. '.' for the root."] = ".") -> ToolResult:
        """List a directory: subdirectories and files with line counts."""
        p = self._norm(path)
        if p in self._lines:
            return self._finish(f"ERROR is_file: {p!r} is a file. Use overview or read_file.", error=True)
        if p not in self._dirs:
            return self._finish(not_found("directory", p, sorted(self._dirs)), error=True)
        subdirs = sorted(d for d in self._dirs if d != "." and (d.rsplit("/", 1)[0] if "/" in d else ".") == p)
        files = sorted(f for f in self._paths if (f.rsplit("/", 1)[0] if "/" in f else ".") == p)
        entries = [f"dir/  {d.rsplit('/', 1)[-1]}" for d in subdirs] + [f"file  {f.rsplit('/', 1)[-1]}  ({self._lines[f]} lines)" for f in files]
        if not entries:
            return self._finish(f"{p}/ is empty.")
        return self._finish(f"{p}/  ({len(subdirs)} dirs, {len(files)} files)\n" + "\n".join(cap_lines(entries, self.caps.list_entries)))
