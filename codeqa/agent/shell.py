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
OUTPUT_CAP = 12000   # 2026-09-20: 8000 forced 100-line chunks; explain tasks burned every call reading one file
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
    # (the read-only allowlist is enforced on COMMAND WORDS in precheck(); a word-anywhere regex blocked `grep x python-package/`)
]


STDERR_NULL = re.compile(r"\s*2>\s*/dev/null")   # harmless read-only idiom; the model uses it constantly (13 % of commands were blocked for it)


DISALLOWED_WORDS = {"rm", "mv", "cp", "chmod", "chown", "tee", "dd", "truncate", "python", "python2", "python3", "perl", "ruby", "node",
                    "curl", "wget", "ssh", "nc", "bash", "sh", "zsh", "env", "export", "eval", "exec", "cd"}


def _command_words(command: str) -> list[str]:
    """First word of every pipeline / list segment, quote-aware: `grep -E "a|b" . | head` -> [grep, head]."""
    import shlex
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:                                   # unbalanced quotes: fall back to a naive split
        tokens = re.split(r"(\|\||&&|;|\|)", command)
        tokens = [t for part in tokens for t in ([part] if part in ("||", "&&", ";", "|") else part.split())]
    words, start = [], True
    for t in tokens:
        if t in ("|", "||", "&&", ";", "&", "|&"):
            start = True
            continue
        if start:
            words.append(t.rsplit("/", 1)[-1])
            start = False
    return words


def precheck(command: str) -> str | None:
    command = STDERR_NULL.sub("", command)
    bad = [w for w in _command_words(command) if w in DISALLOWED_WORDS or (w not in ALLOWED_BINARIES and not w.startswith("-"))]
    if bad:
        return f"only read-only commands are available: {', '.join(ALLOWED_BINARIES)} (got: {bad[0]})"
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


# ---------------------------------------------------------------------------
# Grep self-healing (CODEQA_BASH_HEAL=1, v3 harness): a bad regex is retried once as a literal search with the escapes
# removed; zero hits are retried once case-insensitively. Both retries are labelled and stay inside the same tool call.
# Multi-variation searching (synonyms, splitting words) is deliberately NOT done: that is the model's job to learn.
# ---------------------------------------------------------------------------

_GREP_WORDS = ("grep", "egrep", "fgrep")
_REGEX_ERR = re.compile(r"Unmatched|Invalid (regular expression|preceding|range|content|character class|back reference|collation)"
                        r"|Trailing backslash|repetition-operator|brackets|parenthes|Regular expression too big|Premature end|Unterminated",
                        re.IGNORECASE)
_TAKES_VALUE = {"-e", "--regexp", "-f", "--file", "-A", "-B", "-C", "-m", "--include", "--exclude", "--exclude-dir", "--max-count"}
HEAL_LITERAL_NOTE = "(the pattern was not a valid regex; showing literal matches for {pat!r})\n"
HEAL_ICASE_NOTE = "(no exact matches; showing case-insensitive matches)\n"


# The product serves several checkpoints from one process, so the v3 knobs can also be set per asyncio task
# (contextvars) instead of process-wide; None = fall back to the environment (training and CLIs set the env).
import contextvars as _cv
HEAL_OVERRIDE: _cv.ContextVar[bool | None] = _cv.ContextVar("codeqa_bash_heal", default=None)
PIPELINES_OVERRIDE: _cv.ContextVar[bool | None] = _cv.ContextVar("codeqa_seen_pipelines", default=None)


def heal_enabled() -> bool:
    o = HEAL_OVERRIDE.get()
    return o if o is not None else os.environ.get("CODEQA_BASH_HEAL", "0") == "1"


def _split_first_segment(command: str) -> tuple[str, str]:
    """('grep -rn foo .', ' | head -5') split at the first unquoted pipe; ('cmd', '') when there is none."""
    depth_q: str | None = None
    for i, ch in enumerate(command):
        if depth_q:
            if ch == depth_q:
                depth_q = None
        elif ch in ("'", '"'):
            depth_q = ch
        elif ch == "|":
            if command[i + 1:i + 2] == "|":
                return command, ""
            return command[:i], command[i:]
    return command, ""


def grep_variant(command: str, *, literal: bool = False, ignore_case: bool = False) -> str | None:
    """The same command with the grep pattern unescaped and -F / -i added; None when it is not a plain grep call."""
    first, rest = _split_first_segment(command)
    try:
        args = shlex.split(first, posix=True)
    except ValueError:
        return None
    if not args or args[0] not in _GREP_WORDS:
        return None
    pat_idx: int | None = None
    i = 1
    while i < len(args):
        a = args[i]
        if a == "--":
            pat_idx = i + 1 if i + 1 < len(args) else None
            break
        if a.startswith("-") and len(a) > 1:
            if a in ("-e", "--regexp"):
                pat_idx = i + 1 if i + 1 < len(args) else None
                break
            if a in ("-f", "--file"):
                return None
            if a in _TAKES_VALUE and "=" not in a:
                i += 2
                continue
            i += 1
            continue
        pat_idx = i
        break
    if pat_idx is None or pat_idx >= len(args):
        return None
    flags = [a for a in args[1:pat_idx] if a.startswith("-")]
    if literal and any(f in ("-F", "--fixed-strings") or (f.startswith("-") and not f.startswith("--") and "F" in f[1:]) for f in flags):
        return None
    if ignore_case and any(f in ("-i", "--ignore-case") or (f.startswith("-") and not f.startswith("--") and "i" in f[1:]) for f in flags):
        return None
    new = list(args)
    if literal:
        new[pat_idx] = re.sub(r"\\([^A-Za-z0-9])", r"\1", new[pat_idx])
        if new[0] == "egrep":
            new[0] = "grep"
        new = [a for a in new if a not in ("-E", "--extended-regexp", "-P", "--perl-regexp")]
        new.insert(1, "-F")
    if ignore_case:
        new.insert(1, "-i")
    return shlex.join(new) + rest


async def _exec(command: str, cwd: Path, timeout: float, cap: int, repo_id: str | None) -> tuple[str, int | None, str | None]:
    """(output, returncode, error): the raw execution on either executor."""
    if EXECUTOR == "modal":
        from codeqa.agent import modal_shell
        return await modal_shell.run_rc(command, repo_id or cwd.name, timeout, cap)
    # Restricted bash forbids every redirection, so the one idiom the precheck allows (`2>/dev/null`) is honoured
    # here instead: strip it and drop stderr. Same output as on the Modal executor, where the redirect runs as written.
    drop_stderr = bool(STDERR_NULL.search(command))
    argv = [BASH, "-r", "-c", STDERR_NULL.sub("", command) if drop_stderr else command]
    if sandbox_exec_works():
        argv = [SANDBOX_EXEC, "-p", _sandbox_profile(cwd)] + argv
    env = {"PATH": str(allowlist_dir()), "LC_ALL": "C", "HOME": "/nonexistent", "TERM": "dumb"}
    try:
        proc = await asyncio.create_subprocess_exec(*argv, cwd=str(cwd), env=env, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL if drop_stderr else asyncio.subprocess.STDOUT)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        return "", None, f"timeout after {timeout:.0f}s; narrow the command"
    text = out.decode(errors="replace")
    if len(text) > cap:
        text = text[:cap] + f"\n(output truncated to {cap} chars; narrow the command, e.g. add | head -50 or a line range)"
    return text, proc.returncode, None


async def run(command: str, cwd: Path, timeout: float = TIMEOUT, cap: int = OUTPUT_CAP, repo_id: str | None = None) -> tuple[str, str | None]:
    """Returns (output, error). Output is stdout+stderr, capped. Error is a short reason when the command was refused."""
    why = precheck(command)
    if why:
        return "", f"blocked: {why}"
    text, rc, err = await _exec(command, cwd, timeout, cap, repo_id)
    if err:
        return "", err
    if heal_enabled():
        if rc == 2 and _REGEX_ERR.search(text):
            alt = grep_variant(command, literal=True)
            if alt and alt != command:
                t2, rc2, err2 = await _exec(alt, cwd, timeout, cap, repo_id)
                if not err2 and rc2 == 0 and t2.strip():
                    pat = shlex.split(_split_first_segment(alt)[0])
                    lit = next((a for a in pat[2:] if not a.startswith("-")), "")
                    return (HEAL_LITERAL_NOTE.format(pat=lit) + t2).rstrip("\n"), None
        elif rc == 1 and not text.strip():
            alt = grep_variant(command, ignore_case=True)
            if alt and alt != command:
                t2, rc2, err2 = await _exec(alt, cwd, timeout, cap, repo_id)
                if not err2 and rc2 == 0 and t2.strip():
                    return (HEAL_ICASE_NOTE + t2).rstrip("\n"), None
    if rc not in (0, 1, None) and not text.strip():  # grep returns 1 on no match; that is not an error
        text = f"(exit {rc}, no output)"
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


_SED_RANGE = re.compile(r"\bsed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?")
_HEAD_N = re.compile(r"\bhead\s+(?:-n\s*|-)(\d+)")
_TAIL_N = re.compile(r"\btail\s+(?:-n\s*|-)(\d+)")


_STAGE_HEAD = re.compile(r"^\s*head\s+(?:-n\s*|-)(\d+)\s*$")
_STAGE_TAIL = re.compile(r"^\s*tail\s+(?:-n\s*|-)(\d+)\s*$")
_STAGE_SED = re.compile(r"^\s*sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s*$")


def pipelines_enabled() -> bool:
    """CODEQA_SEEN_PIPELINES=1 (v3 harness): `cat FILE | head -150 | tail -30` style reads of ONE file register the lines
    they print, like the plain forms below. Off by default so the phase-6 arms keep one grounding rule."""
    o = PIPELINES_OVERRIDE.get()
    return o if o is not None else os.environ.get("CODEQA_SEEN_PIPELINES", "0") == "1"


def _pipeline_span(command: str, output: str, single: str, n_lines: int) -> Span | None:
    """Evaluate a chain of head/tail/sed -n stages over one file's lines 1..n_lines: (start, end) after every stage."""
    stages = [seg.strip() for seg in command.split("|")]
    first, rest = stages[0], stages[1:]
    if not rest:
        return None
    words = first.split()
    if not words:
        return None
    if words[0] == "cat":
        s, e = 1, n_lines
    elif words[0] == "head":
        m = _HEAD_N.search(first)
        if not m:
            return None
        s, e = 1, min(int(m.group(1)), n_lines)
    elif words[0] == "tail":
        m = _TAIL_N.search(first)
        if not m:
            return None
        s, e = max(n_lines - int(m.group(1)) + 1, 1), n_lines
    elif words[0] == "sed":
        m = _SED_RANGE.search(first)
        if not m:
            return None
        s, e = int(m.group(1)), min(int(m.group(2)), n_lines)
    else:
        return None
    for stage in rest:
        if m := _STAGE_HEAD.match(stage):
            e = min(e, s + int(m.group(1)) - 1)
        elif m := _STAGE_TAIL.match(stage):
            s = max(s, e - int(m.group(1)) + 1)
        elif m := _STAGE_SED.match(stage):
            a, b = int(m.group(1)), int(m.group(2))
            s, e = s + a - 1, min(e, s + b - 1)
        else:
            return None                                   # grep, sort, wc ...: the printed lines are not a contiguous range
    printed = len([ln for ln in output.splitlines() if not ln.startswith("(output cut") and not ln.startswith("[")])
    if printed == 0 or s > e or s > n_lines:
        return None
    return Span(path=single, start=s, end=min(e, s + printed - 1))


def _unnumbered_span(command: str, output: str, single: str, n_lines: int) -> Span | None:
    """`sed -n 'A,Bp' FILE`, `head -N FILE`, `tail -N FILE`, `cat FILE` show real lines without numbers; the model can
    count from the range it asked for, so they count as seen (2026-09-20: a correct Pillow answer scored 0 for this).
    Only the lines actually printed count (the output cap can cut a range short)."""
    printed = len([ln for ln in output.splitlines() if not ln.startswith("(output cut") and not ln.startswith("[")])
    if printed == 0:
        return None
    first = command.split("|")[0] if "|" in command else command
    if "nl " in command or " -n" in first.split("sed")[0]:
        return None                                       # numbered forms are handled by the per-line parser
    if "|" in command and pipelines_enabled():
        return _pipeline_span(command, output, single, n_lines)
    m = _SED_RANGE.search(command)
    if m and "sed" in first:
        a, b = int(m.group(1)), int(m.group(2))
        return Span(path=single, start=a, end=min(b, n_lines, a + printed - 1)) if a <= n_lines else None
    m = _HEAD_N.search(command)
    if m and first.strip().startswith("head"):
        return Span(path=single, start=1, end=min(int(m.group(1)), n_lines, printed))
    m = _TAIL_N.search(command)
    if m and first.strip().startswith("tail"):
        k = min(int(m.group(1)), n_lines, printed)
        return Span(path=single, start=n_lines - k + 1, end=n_lines)
    if first.strip().startswith("cat ") and "|" not in command:
        return Span(path=single, start=1, end=min(n_lines, printed))
    return None


def seen_spans(command: str, output: str, known: dict[str, int]) -> list[Span]:
    """Lines shown with their numbers (grep -n, nl -ba, cat -n), plus unnumbered reads of one file whose range is
    implied by the command (sed -n 'A,Bp', head, tail, cat) — see _unnumbered_span."""
    hits: set[tuple[str, int]] = set()
    files = _file_args(command, known)
    single = files[0] if len(files) == 1 else None
    if single is not None:
        sp = _unnumbered_span(command, output, single, known[single])
        if sp is not None:
            for k in range(sp.start, sp.end + 1):
                hits.add((single, k))
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
