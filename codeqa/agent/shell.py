"""Restricted local shell for the `bash` agent variant, plus the seen-lines extractor that keeps grounding honest.

Containment (belt and braces; Modal's sandbox is the real container for training at scale):
  1. pre-check rejects `..`, absolute paths, `~`, redirections, `find -delete/-exec`, `sed -i`;
  2. `bash -r` (restricted: no cd, no PATH changes, no redirections) with PATH = an allowlist dir of read-only tools;
  3. on macOS, `sandbox-exec` denies writes everywhere, network, and reads outside the snapshot + system dirs.
The model sees the snapshot as its working directory; every command counts as one tool call.
"""
from __future__ import annotations

import asyncio
import os
import platform
import re
import shlex
import shutil
from functools import lru_cache
from pathlib import Path

from codeqa.shared import paths
from codeqa.shared.contracts import Span

ALLOWED_BINARIES = ("ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "find", "nl", "wc", "sort", "uniq", "cut", "tr",
                    "echo", "basename", "dirname", "sed", "xargs", "tree", "rg", "diff", "stat", "file", "true", "false")
OUTPUT_CAP = 8000
TIMEOUT = 15.0
SANDBOX_EXEC = shutil.which("sandbox-exec")   # resolved now: the child PATH is the allowlist dir only
BASH = "/bin/bash"

_BLOCK = [
    (re.compile(r"(^|[\s'\"=])\.\.(/|$|[\s'\"])"), "parent paths (..) are not allowed"),
    (re.compile(r"(^|[\s'\"=])~"), "home paths (~) are not allowed"),
    (re.compile(r"(^|[\s'\"=])/(?!dev/null)"), "absolute paths are not allowed; paths are relative to the repo root"),
    (re.compile(r"[<>]"), "redirection is not allowed (read-only shell)"),
    (re.compile(r"`|\$\(|\$\{"), "command substitution is not allowed"),
    (re.compile(r"\bfind\b[^|]*\s-(delete|exec|execdir|ok|okdir|fprint\w*)\b"), "find -delete/-exec are not allowed"),
    (re.compile(r"\bsed\b[^|]*\s(-[a-zA-Z]*i[a-zA-Z]*|--in-place)\b"), "sed -i is not allowed"),
    (re.compile(r"\b(rm|mv|cp|chmod|chown|tee|dd|truncate|python\d?|perl|ruby|node|curl|wget|ssh|nc|bash|sh|zsh|env|export|eval|exec)\b"),
     "only read-only commands are available: " + ", ".join(ALLOWED_BINARIES)),
]


STDERR_NULL = re.compile(r"\s*2>\s*/dev/null")   # harmless read-only idiom; the model uses it constantly (13 % of commands were blocked for it)


def precheck(command: str) -> str | None:
    command = STDERR_NULL.sub("", command)
    cmd = command.strip()
    if not cmd:
        return "empty command"
    if len(cmd) > 600:
        return "command too long"
    for rx, why in _BLOCK:
        if rx.search(cmd):
            return why
    return None


@lru_cache(maxsize=1)
def allowlist_dir() -> Path:
    """A directory of symlinks to the allowed binaries; PATH points only here."""
    d = paths.DATA / ".bin"
    d.mkdir(parents=True, exist_ok=True)
    for name in ALLOWED_BINARIES:
        target = shutil.which(name)
        if name == "rg" and (target is None or "claude" in str(target)):
            continue  # the Claude Code shim is not a real rg
        link = d / name
        if target and not link.exists():
            try:
                link.symlink_to(target)
            except OSError:
                pass
    return d


def _sandbox_profile(cwd: Path) -> str:
    home = str(Path.home())
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (literal "/dev/null") (subpath "/dev") (subpath "/private/tmp") (subpath "/tmp"))
(deny file-read* (subpath "{home}"))
(allow file-read* (subpath "{cwd}") (subpath "{allowlist_dir()}"))
"""


@lru_cache(maxsize=1)
def sandbox_exec_works() -> bool:
    if platform.system() != "Darwin" or SANDBOX_EXEC is None:
        return False
    try:
        import subprocess
        probe = paths.DATA / ".sandbox_probe"
        probe.mkdir(exist_ok=True)
        ok = subprocess.run([SANDBOX_EXEC, "-p", _sandbox_profile(probe), BASH, "-r", "-c", "echo ok"],
                            capture_output=True, text=True, timeout=10, env={"PATH": str(allowlist_dir())})
        # the layer must also deny a read outside the sandboxed dir (the project's .env is the file that matters)
        denied = subprocess.run([SANDBOX_EXEC, "-p", _sandbox_profile(probe), BASH, "-r", "-c", f"cat {paths.ROOT / '.env.example'}"],
                                capture_output=True, text=True, timeout=10, env={"PATH": str(allowlist_dir())})
        return ok.returncode == 0 and "ok" in ok.stdout and denied.returncode != 0 and "TINKER" not in denied.stdout
    except Exception:  # noqa: BLE001
        return False


EXECUTOR = os.environ.get("CODEQA_BASH_EXECUTOR", "local")   # local | modal


async def run(command: str, cwd: Path, timeout: float = TIMEOUT, cap: int = OUTPUT_CAP, repo_id: str | None = None) -> tuple[str, str | None]:
    """Returns (output, error). Output is stdout+stderr, capped. Error is a short reason when the command was refused."""
    why = precheck(command)
    if why:
        return "", f"blocked: {why}"
    if EXECUTOR == "modal":
        from codeqa.agent import modal_shell
        return await modal_shell.run(command, repo_id or cwd.name, timeout, cap)
    argv = [BASH, "-r", "-c", command]
    if sandbox_exec_works():
        argv = [SANDBOX_EXEC, "-p", _sandbox_profile(cwd)] + argv
    env = {"PATH": str(allowlist_dir()), "LC_ALL": "C", "HOME": "/nonexistent", "TERM": "dumb"}
    try:
        proc = await asyncio.create_subprocess_exec(*argv, cwd=str(cwd), env=env,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return "", f"timeout after {timeout:.0f}s; narrow the command"
    text = out.decode(errors="replace")
    if len(text) > cap:
        text = text[:cap] + f"\n(output truncated to {cap} chars; narrow the command, e.g. add | head -50 or a line range)"
    if proc.returncode not in (0, 1) and not text.strip():  # grep returns 1 on no match; that is not an error
        text = f"(exit {proc.returncode}, no output)"
    return text.rstrip("\n"), None


# ---------------------------------------------------------------------------
# Seen-lines extraction: which (path, line) pairs did the output show with their numbers?
# ---------------------------------------------------------------------------

_PATH_LINE = re.compile(r"^\.?/?([^\s:]+?):(\d+)[:-]")      # grep -rn / rg -n: path:NN: or path:NN-
_NUM_LINE = re.compile(r"^\s*(\d+)[\t :]")                  # nl -ba / cat -n / grep -n on one file: NN<tab> or NN:


def _file_args(command: str, known: dict[str, int]) -> list[str]:
    """Repo paths mentioned as tokens anywhere in the command (any pipe stage)."""
    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()
    out = []
    for t in toks:
        p = re.sub(r"^\./", "", t)
        if p in known and p not in out:
            out.append(p)
    return out


def seen_spans(command: str, output: str, known: dict[str, int]) -> list[Span]:
    """Lines shown WITH their numbers. Unnumbered output records nothing, so the model learns to use grep -n / nl -ba."""
    hits: set[tuple[str, int]] = set()
    files = _file_args(command, known)
    single = files[0] if len(files) == 1 else None
    for line in output.splitlines():
        m = _PATH_LINE.match(line)
        if m:
            p = m.group(1)
            if p in known and 1 <= int(m.group(2)) <= known[p]:
                hits.add((p, int(m.group(2))))
            continue
        m = _NUM_LINE.match(line)
        if m and single is not None and 1 <= int(m.group(1)) <= known[single]:
            hits.add((single, int(m.group(1))))
    # merge consecutive lines per path into ranges
    spans: list[Span] = []
    for p in sorted({h[0] for h in hits}):
        nums = sorted(n for q, n in hits if q == p)
        start = prev = nums[0]
        for n in nums[1:]:
            if n != prev + 1:
                spans.append(Span(path=p, start=start, end=prev))
                start = n
            prev = n
        spans.append(Span(path=p, start=start, end=prev))
    return spans
