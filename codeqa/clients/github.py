"""GitHub tarball at a pinned commit. Token from GITHUB_TOKEN or `gh auth token`."""
from __future__ import annotations

import io
import os
import subprocess
import tarfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()


def token() -> str | None:
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip() or None
    except Exception:
        return None


def _headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "codeqa"}
    t = token()
    if t:
        h["Authorization"] = f"Bearer {t}"
    return h


def resolve_sha(owner: str, repo: str, ref: str) -> str:
    """Full sha for a branch, tag, or short sha."""
    r = httpx.get(f"https://api.github.com/repos/{owner}/{repo}/commits/{ref}", headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()["sha"]


def fetch_tarball(owner: str, repo: str, sha: str, dest: Path) -> Path:
    """Download and extract repo@sha into dest (top-level dir stripped). Returns dest."""
    url = f"https://api.github.com/repos/{owner}/{repo}/tarball/{sha}"
    with httpx.Client(follow_redirects=True, timeout=300) as c:
        r = c.get(url, headers=_headers())
        r.raise_for_status()
        data = r.content
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = tf.getmembers()
        prefix = members[0].name.split("/")[0] if members else ""
        for m in members:
            if not m.name.startswith(prefix + "/"):
                continue
            m.name = m.name[len(prefix) + 1:]
            if not m.name:
                continue
            if m.isfile() or m.isdir():
                tf.extract(m, dest, filter="data")
    return dest


def fetch_text(owner: str, repo: str, path: str, ref: str = "HEAD") -> str:
    r = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
        headers={**_headers(), "Accept": "application/vnd.github.raw+json"},
        params={"ref": ref}, timeout=60,
    )
    r.raise_for_status()
    return r.text
