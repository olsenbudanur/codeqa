"""Read-only view of a repo snapshot and its symbol index, for the grader (C1, C2).

The grader may import only `codeqa.shared` and `codeqa.clients`, so it reads
`symbols.json` itself instead of going through `codeqa.agent.indexing`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from codeqa.shared import paths
from codeqa.shared.contracts import IndexSymbol

FIXTURE_REPO_ID = "mini__repo__0000001"   # tests/fixtures/mini_repo + tests/fixtures/index


@dataclass
class RepoFiles:
    """Where a repo's files and symbols live. `root=None` means the snapshot is missing."""

    repo_id: str
    root: Path | None
    symbols: list[IndexSymbol] = field(default_factory=list)
    _line_counts: dict[str, int] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, repo_id: str) -> "RepoFiles":
        if repo_id == FIXTURE_REPO_ID:
            root = paths.FIXTURES / "mini_repo"
            sym_path = paths.FIXTURES / "index" / "symbols.json"
        else:
            root = paths.repo_dir(repo_id)
            sym_path = paths.index_dir(repo_id) / "symbols.json"
        symbols: list[IndexSymbol] = []
        if sym_path.is_file():
            data = json.loads(sym_path.read_text())
            rows = data["symbols"] if isinstance(data, dict) else data     # index.py writes {"repo_id", "symbols"}; C2 shows the list
            symbols = [IndexSymbol.model_validate(s) for s in rows]
        return cls(repo_id=repo_id, root=root if root.is_dir() else None, symbols=symbols)

    def normalize(self, path: str) -> str:
        p = path.strip().replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        return p.lstrip("/")

    def line_count(self, path: str) -> int | None:
        """Number of lines in the file, or None if the file does not exist in the snapshot."""
        path = self.normalize(path)
        if path in self._line_counts:
            return self._line_counts[path]
        if self.root is None:
            return None
        fp = self.root / path
        if not fp.is_file() or ".." in Path(path).parts:
            return None
        try:
            n = fp.read_bytes().count(b"\n")
            if not fp.read_bytes().endswith(b"\n"):
                n += 1
        except OSError:
            return None
        self._line_counts[path] = n
        return n

    def exists(self, path: str, start: int, end: int) -> bool:
        n = self.line_count(path)
        return n is not None and 1 <= start <= end <= n

    def find_symbols(self, qualified: str) -> list[IndexSymbol]:
        """Resolve a CodeScout-style 'path:Class.method' (or 'path:name', or bare 'name') to index entries."""
        if ":" in qualified:
            path, name = qualified.rsplit(":", 1)
            path = self.normalize(path)
            hits = [s for s in self.symbols if s.path == path and (s.qualified == f"{path}:{name}" or s.name == name.split(".")[-1])]
            exact = [s for s in hits if s.qualified == f"{path}:{name}"]
            return exact or hits
        return [s for s in self.symbols if s.name == qualified]

    def symbols_in(self, path: str, start: int, end: int) -> list[IndexSymbol]:
        """Symbols whose definition (def line through end of body) overlaps the given range."""
        path = self.normalize(path)
        return [s for s in self.symbols if s.path == path and start <= s.end and end >= s.start]

    def line_text(self, path: str, n: int) -> str:
        """Text of line n (1-based), "" if unavailable."""
        path = self.normalize(path)
        if self.root is None or ".." in Path(path).parts:
            return ""
        fp = self.root / path
        try:
            lines = fp.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return lines[n - 1] if 1 <= n <= len(lines) else ""


@lru_cache(maxsize=64)
def load_repo(repo_id: str) -> RepoFiles:
    return RepoFiles.load(repo_id)
