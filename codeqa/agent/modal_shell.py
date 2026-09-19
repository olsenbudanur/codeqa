"""Modal-backed executor for the `bash` agent variant: a pool of sandboxes over the `codeqa-data` volume.

One sandbox serves many concurrent commands (each `exec` is a separate process in the container), so a small pool
carries dozens of parallel episodes. Snapshots live at /data/repos/<repo_id> on the volume (`modal volume put`).
Selected with CODEQA_BASH_EXECUTOR=modal; the pre-check and seen-lines logic in agent/shell.py stay identical,
so grounding and reward are the same as the local executor. Pool size: CODEQA_MODAL_SANDBOXES (default 4).
"""
from __future__ import annotations

import asyncio
import itertools
import os
from typing import Any

APP_NAME = "codeqa-sandbox"
VOLUME_NAME = "codeqa-data"
DATA_MOUNT = "/data"
SANDBOX_TIMEOUT = 3600          # hard lifetime of one sandbox, seconds
IDLE_TIMEOUT = 600              # torn down by Modal after this long without an exec
POOL_SIZE = int(os.environ.get("CODEQA_MODAL_SANDBOXES", "4"))

_pool: list[Any] = []
_lock: asyncio.Lock | None = None
_rr = itertools.count()


def image():
    import modal
    return modal.Image.debian_slim(python_version="3.12").apt_install("ripgrep", "tree", "file")


async def _create_one():
    import modal
    app = await asyncio.to_thread(modal.App.lookup, APP_NAME, create_if_missing=True)   # sync lookups in this SDK version
    vol = modal.Volume.from_name(VOLUME_NAME)
    return await modal.Sandbox.create.aio(
        app=app, image=image(), volumes={DATA_MOUNT: vol}, timeout=SANDBOX_TIMEOUT, idle_timeout=IDLE_TIMEOUT,
        cpu=2.0, memory=2048, block_network=True,
    )


async def _get_sandbox():
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        while len(_pool) < POOL_SIZE:
            _pool.append(await _create_one())
    return _pool[next(_rr) % len(_pool)]


async def run(command: str, repo_id: str, timeout: float, cap: int) -> tuple[str, str | None]:
    """Same contract as shell.run: (output, error). Retries once on a dead sandbox."""
    text, _rc, err = await run_rc(command, repo_id, timeout, cap)
    return text, err


async def run_rc(command: str, repo_id: str, timeout: float, cap: int) -> tuple[str, int | None, str | None]:
    """(output, returncode, error): like run() but keeps the exit code so shell.run can heal grep failures."""
    workdir = f"{DATA_MOUNT}/repos/{repo_id}"
    last_err = None
    for attempt in range(2):
        sb = await _get_sandbox()
        try:
            proc = await sb.exec.aio("bash", "-c", command, workdir=workdir, timeout=int(timeout),
                                     env={"LC_ALL": "C", "TERM": "dumb"})
            out = await asyncio.wait_for(proc.stdout.read.aio(), timeout=timeout + 5)
            err = await asyncio.wait_for(proc.stderr.read.aio(), timeout=5)
            rc = await proc.wait.aio()
            text = out + (("\n" + err) if err.strip() else "")
            if len(text) > cap:
                text = text[:cap] + f"\n(output truncated to {cap} chars; narrow the command, e.g. add | head -50 or a line range)"
            return text.rstrip("\n"), rc, None
        except asyncio.TimeoutError:
            return "", None, f"timeout after {timeout:.0f}s; narrow the command"
        except Exception as e:  # noqa: BLE001  (sandbox died / terminated): drop it and retry once
            last_err = f"{type(e).__name__}: {e}"
            try:
                _pool.remove(sb)
            except ValueError:
                pass
    return "", None, f"sandbox error: {last_err}"


async def close() -> None:
    for sb in list(_pool):
        try:
            await sb.terminate.aio()
        except Exception:  # noqa: BLE001
            pass
    _pool.clear()
