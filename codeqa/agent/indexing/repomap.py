"""Repo map: the token-capped text overview that goes into every prompt. (C2 map.txt)

Rendered from the manifest (tree), the symbol index (top-level names) and summaries (one line per directory).
Detail is reduced level by level until the map fits `max_tokens`, so the prompt never overflows.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from codeqa.shared import paths
from codeqa.shared.contracts import IndexSymbol, Manifest

MAX_TOKENS = 3000
CODE_LANGS = {"python", "javascript", "typescript", "tsx", "go", "rust", "java", "c", "cpp"}
TOKEN_MODEL = "Qwen/Qwen3.5-4B"
DEMOTED_DIRS = {"tests", "test", "testing", "docs", "doc", "examples", "example", "benchmarks", "scripts", "ci", "tools"}


def map_path(repo_id: str) -> Path:
    return paths.index_dir(repo_id) / "map.txt"


def load_map(repo_id: str) -> str:
    p = map_path(repo_id)
    return p.read_text() if p.exists() else ""


@lru_cache(maxsize=1)
def _tok():
    try:
        from codeqa.clients.tinker import tokenizer
        return tokenizer(TOKEN_MODEL)
    except Exception:  # noqa: BLE001  (no HF cache / offline): approximate
        return None


def count_tokens(text: str) -> int:
    t = _tok()
    return len(t.encode(text, add_special_tokens=False)) if t is not None else len(text) // 4 + 1


def _dir_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else "."


def _summary_line(s: str, width: int) -> str:
    """First sentence of the paragraph, cut at a word boundary."""
    s = " ".join(s.split())
    first = s.split(". ", 1)[0]
    s = first if len(first) >= 20 else s
    return s if len(s) <= width else s[: width - 1].rsplit(" ", 1)[0] + "…"


def render(manifest: Manifest, symbols: list[IndexSymbol], summaries: dict[str, str], *, level: int, max_depth: int) -> str:
    """level 0 = fullest; 3 = directories + summaries only."""
    by_file: dict[str, list[str]] = defaultdict(list)
    for s in symbols:
        if s.parent is None:
            by_file[s.path].append(s.name)
    n_syms = {0: 8, 1: 5, 2: 3, 3: 0}[level]
    sum_width = {0: 100, 1: 80, 2: 64, 3: 64}[level]
    min_lines_for_syms = {0: 0, 1: 100, 2: 150, 3: 10**9}[level]

    tree: dict[str, dict] = {}
    files_in: dict[str, list] = defaultdict(list)
    for f in manifest.files:
        d = _dir_of(f.path)
        files_in[d].append(f)
        parts = [] if d == "." else d.split("/")
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    out: list[str] = []
    hidden_count = 0

    def emit_dir(dpath: str, node: dict, depth: int) -> None:
        nonlocal hidden_count
        indent = "  " * depth
        if depth > max_depth:
            n_files = sum(len(files_in[d]) for d in files_in if d == dpath or d.startswith(dpath + "/"))
            out.append(f"{indent}{dpath.rsplit('/', 1)[-1]}/  (+{n_files} files)")
            return
        if dpath != ".":
            summ = summaries.get(dpath, "")
            out.append(f"{indent}{dpath.rsplit('/', 1)[-1]}/" + (f"  {_summary_line(summ, sum_width)}" if summ else ""))
        def code_files(d: str) -> int:
            return sum(1 for dd, fs in files_in.items() if dd == d or dd.startswith(d + "/") for f in fs if f.lang in CODE_LANGS)

        def demoted(d: str) -> bool:
            return d.rsplit("/", 1)[-1].lower() in DEMOTED_DIRS

        # source first (most code files), then tests/docs/examples; hidden directories always collapsed
        for name in sorted(node, key=lambda n: (demoted(n), -code_files(n if dpath == "." else f"{dpath}/{n}"), n)):
            child = f"{name}" if dpath == "." else f"{dpath}/{name}"
            if name.startswith("."):
                hidden_count += 1
                continue
            emit_dir(child, node[name], depth + (0 if dpath == "." else 1))
        files = files_in.get(dpath, [])
        code = [f for f in files if f.lang in CODE_LANGS]
        other = [f for f in files if f.lang not in CODE_LANGS]
        findent = indent + ("  " if dpath != "." else "")
        for f in code:
            name = f.path.rsplit("/", 1)[-1]
            syms = by_file.get(f.path, []) if f.lines >= min_lines_for_syms else []
            shown = ", ".join(syms[:n_syms]) + (f", +{len(syms) - n_syms}" if len(syms) > n_syms else "")
            if level >= 3:
                continue
            out.append(f"{findent}{name}  {shown}".rstrip() + f"  ({f.lines})")
        if level >= 3 and code:
            out.append(f"{findent}({len(code)} code files)")
        if other:
            if level == 0 and len(other) <= 8:
                for f in other:
                    out.append(f"{findent}{f.path.rsplit('/', 1)[-1]}")
            else:
                out.append(f"{findent}(+{len(other)} other files)")

    emit_dir(".", tree, 0)
    if hidden_count:
        out.append(f"(+{hidden_count} hidden directories)")
    return "\n".join(out) + "\n"


def build_map(manifest: Manifest, symbols: list[IndexSymbol], summaries: dict[str, str], *,
              max_tokens: int = MAX_TOKENS, write: bool = True) -> str:
    """Try full detail first; reduce until it fits. Always returns something under the cap."""
    text = ""
    for level in (0, 1, 2, 3):
        for max_depth in (99, 4, 3, 2, 1):
            text = render(manifest, symbols, summaries, level=level, max_depth=max_depth)
            if count_tokens(text) <= max_tokens:
                break
        else:
            continue
        break
    if count_tokens(text) > max_tokens:  # pathological repo: hard truncate by lines
        lines = text.splitlines()
        while lines and count_tokens("\n".join(lines)) > max_tokens - 10:
            lines = lines[: max(1, int(len(lines) * 0.8))]
        text = "\n".join(lines) + "\n(map truncated)\n"
    if write:
        p = map_path(manifest.repo_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return text
