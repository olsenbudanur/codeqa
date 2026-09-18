"""One full episode through the cookbook tool env on a real task, via RepoEnv (A4 check).

Proves: RepoEnv.initial_messages + tool prefix, all five tools behind build_agent_tool_env, Tinker sampling,
tool-call parsing, tool result injection, budget, reward_fn(history, env), trace_from_history -> C6.

Run: uv run python -u -m scripts.smoke_episode [task_index]   (needs TINKER_API_KEY; ~30s; a few cents)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time

from codeqa.agent.env import RepoEnv
from codeqa.clients import tinker as tk
from codeqa.shared import paths
from codeqa.shared.contracts import CITATION_RE, EndpointProfile, Task
from codeqa.shared.jsonl import read_all

PROFILE = EndpointProfile(name="qwen4b-base", kind="tinker", model="Qwen/Qwen3.5-4B", max_generation_tokens=1024, max_context=32768)
TASKS = paths.TASKS_EVAL / "smoke_sweqa_flask.jsonl"


async def stub_reward(history, env: RepoEnv):
    """Format + grounding only (no judge): 1.0 if every citation parses and lies inside a range that was read."""
    trace = env.trace_from_history(history)
    cits = [(m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))) for m in CITATION_RE.finditer(trace.answer)]
    grounded = [c for c in cits if any(c[0] == s.path and c[1] >= s.start and c[2] <= s.end for s in env.files_read())]
    ok = bool(cits) and len(grounded) == len(cits)
    return (1.0 if ok else 0.0), {"citations": float(len(cits)), "grounded": float(len(grounded)), "format_ok": float(ok),
                                   "tool_calls": float(env.tool_calls_made), "tool_errors": float(env.tool_errors)}


async def run(task_index: int = 0) -> None:
    import tinker
    tasks = read_all(TASKS, Task)
    task = tasks[task_index]
    print(f"task {task.task_id} [{task.task_type}] on {task.repo_id}\nQ: {task.question[:300]}\n")
    env = RepoEnv(task, PROFILE)
    cb_env = env.make_cookbook_env(stub_reward)
    sc, tokenizer = tk.sampling_client(PROFILE.model), tk.tokenizer(PROFILE.model)

    async def maybe(x):
        return await x if inspect.isawaitable(x) else x

    t0 = time.time()
    first = await maybe(cb_env.initial_observation())
    if not isinstance(first, tuple):
        print("initial observation overflow:", first); sys.exit(1)
    obs, stop = first
    print(f"prompt tokens at start: {obs.length} (budget: {env.budget})")
    turn, total_reward, done, res = 0, 0.0, False, None
    while not done:
        turn += 1
        resp = await asyncio.wait_for(sc.sample_async(prompt=obs, num_samples=1,
                                                      sampling_params=tinker.SamplingParams(max_tokens=1024, temperature=1.0, stop=stop)), timeout=90)
        toks = resp.sequences[0].tokens
        text = tokenizer.decode(toks)
        print(f"--- turn {turn}: {len(toks)} tokens | {text[:160].replace(chr(10), ' ')}...")
        res = await maybe(cb_env.step(toks))
        total_reward += res.reward
        done = res.episode_done
        if not done:
            obs, stop = res.next_observation, res.next_stop_condition
            print(f"    tools -> prompt now {obs.length} tokens; calls={env.tool_calls_made} errors={env.tool_errors}")
    trace = env.trace_from_history(cb_env.history if hasattr(cb_env, "history") else cb_env.message_env.history, seconds=time.time() - t0)
    out = paths.TRACES / "dev" / f"{task.task_id}__{PROFILE.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(trace.model_dump_json(indent=1))
    print(f"\n=== done: {trace.stats.turns} turns, {trace.stats.tool_calls} calls, {trace.stats.stop_reason}, {trace.stats.seconds:.0f}s")
    print("reward:", total_reward, "| metrics:", json.dumps(res.metrics, default=float))
    print("files_read:", [(s.path, s.start, s.end) for s in trace.stats.files_read])
    print("answer:", trace.answer[:600])
    print("trace ->", out)
    print("SMOKE_EPISODE PASSED" if trace.stats.turns >= 1 and trace.stats.tool_calls >= 1 else "SMOKE_EPISODE FAILED")


if __name__ == "__main__":
    asyncio.run(run(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
