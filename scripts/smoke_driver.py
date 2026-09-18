"""A5 check: run_episode through a chat client (Claude or Tinker base), streaming C9 events, writing a C6 trace.

Run: uv run python -u -m scripts.smoke_driver [profile] [task_index]      (default profile qwen4b-base)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from codeqa.agent.driver import run_episode, save_trace
from codeqa.agent.env import RepoEnv
from codeqa.clients.base import make_client
from codeqa.shared import paths
from codeqa.shared.contracts import Task, Trace
from codeqa.shared.jsonl import read_all
from codeqa.shared.profiles import get_profile

TASKS = paths.TASKS_EVAL / "smoke_sweqa_flask.jsonl"


def printer(ev):
    p = ev.payload
    if ev.type == "thinking":
        print(f"  [think] {p['text'][:140].replace(chr(10), ' ')}…")
    elif ev.type == "tool_call":
        print(f"  [call ] {p['name']}({json.dumps(p['args'])})  why: {p['why'][:80]}")
    elif ev.type == "tool_result":
        print(f"  [resul] {p['name']}: {p['summary'][:100]} ({p['chars']} chars){' ERROR' if p.get('error') else ''}")
    elif ev.type == "answer":
        print(f"  [answr] {p['markdown'][:300].replace(chr(10), ' ')}…")
    elif ev.type == "citations":
        print(f"  [cites] {p['items']}")
    elif ev.type in ("stats", "error", "done"):
        print(f"  [{ev.type:5}] {p}")


async def main(profile_name: str, task_index: int) -> None:
    profile = get_profile(profile_name)
    task = read_all(TASKS, Task)[task_index]
    print(f"profile {profile.name} ({profile.kind}:{profile.model}) | task {task.task_id} [{task.task_type}]\nQ: {task.question[:200]}")
    t0 = time.time()
    client = await asyncio.wait_for(asyncio.to_thread(make_client, profile), timeout=60)
    env = RepoEnv(task, profile)
    trace = await asyncio.wait_for(run_episode(env, client, printer), timeout=240)
    out = save_trace(trace, "dev")
    back = Trace.model_validate_json(out.read_text())
    assert back.stats.turns >= 1 and back.stats.tool_calls >= 1, back.stats
    print(f"trace -> {out}  ({time.time()-t0:.0f}s)  stop={back.stats.stop_reason} turns={back.stats.turns} calls={back.stats.tool_calls} "
          f"prompt_tokens={back.stats.prompt_tokens} completion_tokens={back.stats.completion_tokens}")
    print("SMOKE_DRIVER PASSED")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "qwen4b-base", int(sys.argv[2]) if len(sys.argv) > 2 else 1))
