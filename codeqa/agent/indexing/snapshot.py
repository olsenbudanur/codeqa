"""repo@sha -> data/repos/<repo_id>/ (cleaned, LF) + manifest.json  (C1)"""
from __future__ import annotations

import shutil
from pathlib import Path

from codeqa.clients import github
from codeqa.shared import paths
from codeqa.shared.contracts import FileEntry, Manifest, make_repo_id
from codeqa.shared.jsonl import dump_json

VENDOR_DIRS = {"node_modules", "vendor", "vendors", "third_party", "thirdparty", "external", "deps", "dist", "build",
               ".git", ".github", ".tox", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "site-packages"}
DROP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip", ".gz", ".tar", ".whl", ".so", ".dylib", ".dll",
            ".pyc", ".pyo", ".lock", ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".bin", ".npy", ".npz", ".pkl", ".parquet"}
DROP_NAMES = {"package-lock.json", "yarn.lock", "poetry.lock", "uv.lock", "Pipfile.lock", "Cargo.lock"}
MAX_BYTES = 1_000_000
LANG_BY_EXT = {".py": "python", ".pyx": "python", ".pyi": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
               ".go": "go", ".rs": "rust", ".java": "java", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
               ".md": "markdown", ".rst": "rst", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".cfg": "ini", ".ini": "ini", ".txt": "text"}


def _is_binary(b: bytes) -> bool:
    return b"\x00" in b[:8000]


def clean(root: Path) -> tuple[list[FileEntry], dict[str, int]]:
    files: list[FileEntry] = []
    dropped = {"vendored": 0, "binary": 0, "oversize": 0, "ext": 0}
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            if p.name in VENDOR_DIRS:
                shutil.rmtree(p, ignore_errors=True); dropped["vendored"] += 1
            continue
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in VENDOR_DIRS for part in rel.parts[:-1]):
            continue  # parent will be removed
        if p.suffix.lower() in DROP_EXT or p.name in DROP_NAMES:
            p.unlink(); dropped["ext"] += 1; continue
        size = p.stat().st_size
        if size > MAX_BYTES:
            p.unlink(); dropped["oversize"] += 1; continue
        raw = p.read_bytes()
        if _is_binary(raw):
            p.unlink(); dropped["binary"] += 1; continue
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        p.write_text(text, encoding="utf-8")
        files.append(FileEntry(path=str(rel), lang=LANG_BY_EXT.get(p.suffix.lower()), lines=text.count("\n") + (0 if text.endswith("\n") or not text else 1), bytes=len(text.encode())))
    return files, dropped


_MANIFEST_CACHE: dict[tuple[str, float], Manifest] = {}


def load_manifest(repo_id: str) -> Manifest:
    p = paths.repo_dir(repo_id) / "manifest.json"
    key = (repo_id, p.stat().st_mtime)
    if key not in _MANIFEST_CACHE:
        _MANIFEST_CACHE.clear() if len(_MANIFEST_CACHE) > 64 else None
        _MANIFEST_CACHE[key] = Manifest.model_validate_json(p.read_text())
    return _MANIFEST_CACHE[key]


def snapshot(owner: str, repo: str, sha: str, force: bool = False) -> Manifest:
    sha = github.resolve_sha(owner, repo, sha) if len(sha) < 40 else sha
    repo_id = make_repo_id(owner, repo, sha)
    dest = paths.repo_dir(repo_id)
    mpath = dest / "manifest.json"
    if mpath.exists() and not force:
        return Manifest.model_validate_json(mpath.read_text())
    if dest.exists():
        shutil.rmtree(dest)
    github.fetch_tarball(owner, repo, sha, dest)
    files, dropped = clean(dest)
    manifest = Manifest(repo_id=repo_id, url=f"https://github.com/{owner}/{repo}", sha=sha, files=files, dropped=dropped)
    dump_json(mpath, manifest)
    return manifest
