"""Curation of tool output: rank, dedupe, collapse, cap, forgiving errors, budget note. Pure functions, no I/O."""
from __future__ import annotations

import difflib
import fnmatch
from dataclasses import dataclass

from codeqa.agent.prompts import BUDGET_WARNING


@dataclass(frozen=True)
class Caps:
    read_lines: int = 80
    grep_hits: int = 20
    grep_files: int = 8
    symbol_hits: int = 20
    overview_lines: int = 40
    list_entries: int = 40
    line_chars: int = 160
    signature_chars: int = 110


DEFAULT_CAPS = Caps()

TEST_MARKERS = ("test", "tests/", "testing/", "conftest", "examples/", "example", "docs/", "benchmark")


def is_test_path(path: str) -> bool:
    p = path.lower()
    return any(m in p for m in TEST_MARKERS)


def rank_paths(paths: list[str]) -> list[str]:
    """Source before tests/docs/examples; shorter (shallower) paths first; then alphabetical."""
    return sorted(paths, key=lambda p: (is_test_path(p), p.count("/"), p))


def truncate(text: str, n: int) -> str:
    text = text.rstrip("\n")
    return text if len(text) <= n else text[: n - 1] + "…"


def cap_lines(lines: list[str], n: int, more: str = "(+{n} more)") -> list[str]:
    if len(lines) <= n:
        return lines
    return lines[:n] + [more.format(n=len(lines) - n)]


def nearest_paths(path: str, all_paths: list[str], n: int = 3) -> list[str]:
    """Forgiving path resolution: exact basename matches first, then fuzzy on the full path."""
    path = path.strip().lstrip("./")
    base = path.rsplit("/", 1)[-1]
    by_base = [p for p in all_paths if p.rsplit("/", 1)[-1] == base]
    out = rank_paths(by_base)[:n]
    if len(out) < n:
        fuzzy = difflib.get_close_matches(path, all_paths, n=n * 2, cutoff=0.6)
        out += [p for p in fuzzy if p not in out]
    if len(out) < n and "/" in path:  # maybe a directory prefix typo; try the last two components
        tail = "/".join(path.split("/")[-2:])
        out += [p for p in all_paths if p.endswith(tail) and p not in out]
    return out[:n]


def not_found(kind: str, path: str, all_paths: list[str]) -> str:
    near = nearest_paths(path, all_paths)
    hint = f" Did you mean: {', '.join(near)}" if near else ""
    return f"ERROR not_found: {kind} {path!r}.{hint}"


def match_file_pattern(path: str, pattern: str | None) -> bool:
    """Glob when it looks like one; otherwise a forgiving substring match on the path."""
    if not pattern:
        return True
    pattern = pattern.strip()
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"*{pattern}") or fnmatch.fnmatch(path, f"*{pattern}*")
    return pattern.lower() in path.lower()


def rg_glob(pattern: str | None) -> str | None:
    if not pattern:
        return None
    pattern = pattern.strip()
    if any(ch in pattern for ch in "*?["):
        return pattern if pattern.startswith("*") or "/" in pattern else f"*{pattern}"
    return f"*{pattern}*"


def budget_note(text: str, calls_made: int, max_calls: int | None) -> str:
    """Append the warning on the last-but-one call and a hard note when the budget is spent."""
    if max_calls is None:
        return text
    remaining = max_calls - calls_made
    if remaining == 1:
        return text.rstrip("\n") + BUDGET_WARNING
    if remaining <= 0:
        return text.rstrip("\n") + "\n[No tool calls remaining. Answer now.]"
    return text
