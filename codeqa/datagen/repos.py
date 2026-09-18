"""Make sure a repo snapshot (C1) and its symbol index (C2) exist locally, for the commit a task source pins.

Wraps lane A's indexing CLI pieces; datagen never writes under data/repos or data/index by any other route.
"""
from __future__ import annotations

import sys
import time

from codeqa.agent.indexing.index import build_index, load_symbols
from codeqa.agent.indexing.snapshot import snapshot
from codeqa.shared import paths
from codeqa.shared.contracts import IndexSymbol, Manifest


def ensure_repo(owner: str, repo: str, sha: str, force: bool = False) -> tuple[Manifest, list[IndexSymbol]]:
    """Snapshot + index if missing. Returns (manifest, symbols). Cheap when both exist."""
    t0 = time.time()
    m = snapshot(owner, repo, sha, force=force)
    sym_path = paths.index_dir(m.repo_id) / "symbols.json"
    if sym_path.exists() and not force:
        syms = load_symbols(m.repo_id)
        fresh = False
    else:
        syms = build_index(m)
        fresh = True
    if fresh or time.time() - t0 > 1:
        py = sum(1 for f in m.files if f.lang == "python")
        print(f"  repo {m.repo_id}: {len(m.files)} files ({py} py), {len(syms)} symbols, {time.time()-t0:.1f}s{' (built)' if fresh else ''}",
              flush=True, file=sys.stderr)
    return m, syms


def load_repo(repo_id: str) -> tuple[Manifest, list[IndexSymbol]]:
    """Load an existing snapshot + index by repo_id; raises if either is missing."""
    mpath = paths.repo_dir(repo_id) / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"no snapshot for {repo_id}; run ensure_repo first")
    return Manifest.model_validate_json(mpath.read_text()), load_symbols(repo_id)
