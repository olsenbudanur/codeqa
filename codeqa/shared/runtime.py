"""Where a job runs (decision #8). `maybe_redirect_to_modal()` at the top of a CLI's main() sends the same command to
the detached Modal runner when CODEQA_RUNTIME=modal (set in .env on the laptop) and the process is not already inside
Modal. CODEQA_RUNTIME=local in front of a command opts out (smokes, tests). Inside a Modal container this is a no-op.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys

from dotenv import load_dotenv

RUNNER = "apps/trainer/modal_runner.py"
PASS_THROUGH = ("CODEQA_AGENT_VARIANT", "CODEQA_BASH_EXECUTOR", "CODEQA_MODAL_SANDBOXES", "CODEQA_WARM_CLIENTS")


def inside_modal() -> bool:
    try:
        import modal
        return not modal.is_local()
    except Exception:  # noqa: BLE001
        return bool(os.environ.get("MODAL_TASK_ID"))


def maybe_redirect_to_modal(module: str, argv: list[str] | None = None) -> None:
    load_dotenv(".env")
    if os.environ.get("CODEQA_RUNTIME", "local").lower() != "modal" or inside_modal():
        return
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):   # tests never launch jobs
        return
    argv = list(sys.argv[1:] if argv is None else argv)
    env = ",".join(f"{k}={os.environ[k]}" for k in PASS_THROUGH if os.environ.get(k))
    cmd = ["modal", "run", "--detach", RUNNER, "--module", module, "--args", shlex.join(argv)]
    if os.environ.get("CODEQA_RUNTIME_WAIT"):
        cmd.append("--wait")
    if env:
        cmd += ["--env", env]
    print(f"[runtime] CODEQA_RUNTIME=modal -> launching detached on Modal:\n  {' '.join(shlex.quote(c) for c in cmd)}\n"
          f"  logs: modal app logs codeqa-jobs   |   results: scripts/modal_sync.sh   |   local instead: CODEQA_RUNTIME=local", flush=True)
    rc = subprocess.run(cmd).returncode
    sys.exit(rc)
