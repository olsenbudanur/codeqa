"""Detached Modal runner for the agent runtime jobs (decision #8): train / evals / filter / teach.

The CLIs are unchanged; they run inside a container whose data root is the `codeqa-data` volume (/data) and whose
keys come from the `codeqa` secret. Outputs (logs, evals, traces, models manifest, profiles.yaml) are committed
to the volume every 30 s and at exit; pull them with scripts/modal_sync.sh.

  modal run --detach apps/trainer/modal_runner.py --module codeqa.evals.run \
      --args "--profile qwen4b-base --tasks data/tasks/eval/fast.jsonl --set fast_modal --concurrency 8"
  modal run --detach apps/trainer/modal_runner.py --module codeqa.trainer.run \
      --args "--tasks data/tasks/train/run1.jsonl --profile qwen4b-base --run-name run1 --steps 50 --group-size 8 \
              --groups-per-batch 16 --lr 1e-4 --eval-tasks data/tasks/eval/fast.jsonl --eval-every 10 --save-every 10"
  modal run --detach apps/trainer/modal_runner.py --module codeqa.trainer.run --env CODEQA_AGENT_VARIANT=bash,CODEQA_BASH_EXECUTOR=modal --args "..."

`data/...` paths in --args are rewritten to /data/... . Logs stream to the terminal until you detach; later:
`modal app logs codeqa-jobs`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import modal

APP_NAME = "codeqa-jobs"
VOLUME = "codeqa-data"
SECRET = "codeqa"
DATA = "/data"
COMMIT_EVERY = 30

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ripgrep", "git", "tree")
    .uv_sync(uv_project_dir=".", groups=None)                    # deps from pyproject + uv.lock (package=false)
    .env({"CODEQA_DATA_DIR": DATA, "CODEQA_PROFILES": f"{DATA}/profiles.yaml", "PYTHONUNBUFFERED": "1", "CODEQA_WARM_CLIENTS": "0"})
    .add_local_python_source("codeqa", "apps", "scripts")      # mounted at run time: code changes need no rebuild
)


def _rewrite(argv: list[str]) -> list[str]:
    out = []
    for a in argv:
        if a.startswith("data/"):
            a = f"{DATA}/" + a[len("data/"):]
        elif "=data/" in a:
            a = a.replace("=data/", f"={DATA}/")
        out.append(a)
    return out


@app.function(image=image, volumes={DATA: volume}, secrets=[modal.Secret.from_name(SECRET)],
              timeout=24 * 3600, cpu=4.0, memory=16384)   # 16 GB: 128 concurrent envs each hold a repo index (cached per repo since 2026-09-19)
def job(module: str, argv: list[str], env: dict[str, str] | None = None) -> int:
    """Run `python -m <module> <argv>` in the container; commit the volume periodically and at exit."""
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(COMMIT_EVERY):
            try:
                volume.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[runner] commit failed: {e}", flush=True)

    t = threading.Thread(target=committer, daemon=True)
    t.start()
    cmd = [sys.executable, "-u", "-m", module, *_rewrite(argv)]
    print(f"[runner] $ {' '.join(cmd)}  (env: {sorted((env or {}).keys())})", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/root", env={**os.environ, **(env or {})})
    stop.set()
    volume.commit()
    print(f"[runner] exit {proc.returncode} after {time.time() - t0:.0f}s; volume committed", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main(module: str, args: str = "", env: str = "", wait: bool = False) -> None:
    """--module codeqa.evals.run --args "<cli args>" [--env KEY=VAL,KEY2=VAL2] [--wait]

    Default: spawn the job and return at once with its call id (the job keeps running on Modal).
    --wait: block and stream until it ends (smokes). Logs any time: `modal app logs codeqa-jobs`.
    """
    import shlex
    argv = shlex.split(args)
    env_map = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    if wait:
        rc = job.remote(module, argv, env_map or None)
        print(f"[runner] job finished with exit code {rc}")
        if rc:
            raise SystemExit(rc)
        return
    call = job.spawn(module, argv, env_map or None)
    print(f"[runner] spawned {module} as {call.object_id}; it keeps running after this returns.\n"
          f"  logs:    modal app logs codeqa-jobs\n  results: scripts/modal_sync.sh\n  wait:    add --wait to block", flush=True)
