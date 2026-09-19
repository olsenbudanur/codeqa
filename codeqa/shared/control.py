"""Out-of-band control for running jobs. A file on the Modal volume is invisible to a running container unless it
calls volume.reload() (which crashes the final eval when files are open), so the stop signal lives in a Modal Dict
that both the laptop and the container read live. The local file (data/STOP) still works for laptop runs.

  laptop:   uv run python -m codeqa.shared.control stop | clear | show
  job:      stop_requested() -> the queue starts no further arm; an arm exits at its next step boundary
"""
from __future__ import annotations

import os
import sys

DICT_NAME = "codeqa-control"
KEY = "stop"


def _dict():
    import modal
    return modal.Dict.from_name(DICT_NAME, create_if_missing=True)


def stop_requested() -> bool:
    from codeqa.shared import paths
    if (paths.DATA / "STOP").exists():
        return True
    try:
        return bool(_dict().get(KEY, False))
    except Exception:  # noqa: BLE001  (no Modal credentials / offline): the file is the only channel
        return False


def request_stop(on: bool = True) -> None:
    d = _dict()
    if on:
        d[KEY] = True
    else:
        d.pop(KEY, None)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "stop":
        request_stop(True); print("stop requested: the queue starts no further arm; the running arm exits at its next step boundary")
    elif cmd == "clear":
        request_stop(False); print("stop cleared")
    else:
        print("stop requested" if stop_requested() else "no stop requested")
