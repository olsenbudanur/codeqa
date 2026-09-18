"""One real Tinker episode over the flask snapshot, graded by the real grader (judge = Haiku).

Proves the grader end to end before lane A's RepoEnv exists: cookbook env + two real tools + reward_fn that builds a
C6 Trace from the history and calls codeqa.grader.grade on a real SWE-QA task.

Run: uv run python -u -m scripts.smoke_grade [task_index]   (~20s; needs TINKER_API_KEY and ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
from typing import Annotated

from codeqa.agent.indexing.index import load_symbols
from codeqa.agent.prompts import system_prompt, user_prompt
from codeqa.clients import tinker as tk
from codeqa.grader.grade import grade, metrics
from codeqa.shared import paths
from codeqa.shared.contracts import Message, Span, Task, ToolCall, Trace, TraceStats
from codeqa.shared.jsonl import read_all
from tinker_cookbook.tool_use.agent_tool_message_env import build_agent_tool_env
from tinker_cookbook.tool_use.tools import simple_tool_result, tool
from tinker_cookbook.tool_use.types import ToolResult

MODEL = "Qwen/Qwen3.5-4B"
REPO_ID = "pallets__flask__85c5d93"
REPO_MAP = "src/flask/\n  app.py        Flask, ...\n  sansio/app.py App, ...\n  sessions.py   SessionInterface, SecureCookieSessionInterface\n  json/provider.py  JSONProvider, DefaultJSONProvider\n  helpers.py, ctx.py, globals.py, wrappers.py, testing.py\n"


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def history_to_messages(history) -> list[Message]:
    out: list[Message] = []
    for m in history:
        d = m if isinstance(m, dict) else m.__dict__
        role = d.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        calls = [ToolCall(name=tc.get("name", "?"), args=tc.get("args") or tc.get("arguments") or {}) for tc in (d.get("tool_calls") or []) if isinstance(tc, dict)]
        out.append(Message(role=role, content=_text(d.get("content")), thinking=d.get("thinking"), tool_calls=calls, name=d.get("name")))
    return out


def build(task: Task, budget_calls: int):
    root = paths.repo_dir(REPO_ID)
    symbols = load_symbols(REPO_ID)
    b = task.effective_budget()

    class RepoTools:
        def __init__(self):
            self.calls, self.errors, self.files_read = 0, 0, []

        @tool
        async def find_symbol(self, name: Annotated[str, "Exact symbol name (class or function)"]) -> ToolResult:
            self.calls += 1
            hits = [s for s in symbols if s.name == name][:20]
            if not hits:
                self.errors += 1
                return simple_tool_result(f"No symbol named {name!r}.")
            return simple_tool_result("\n".join(f"{s.path}:L{s.start}-L{s.end}  {s.kind}  {s.signature}" for s in hits))

        @tool
        async def read_file(self, path: Annotated[str, "Repo-relative file path"], start: Annotated[int, "First line, 1-based"],
                            end: Annotated[int, "Last line, inclusive"]) -> ToolResult:
            self.calls += 1
            p = root / path
            if not p.is_file():
                self.errors += 1
                return simple_tool_result(f"ERROR not_found: {path}")
            src = p.read_text().splitlines()
            start = max(1, start)
            end = min(end, start + 149, len(src))
            if end < start:
                self.errors += 1
                return simple_tool_result(f"ERROR bad_range: {path} has {len(src)} lines")
            self.files_read.append(Span(path=path, start=start, end=end))
            return simple_tool_result("\n".join(f"{i:5d} | {src[i - 1]}" for i in range(start, end + 1)) + f"\n(total {len(src)} lines)")

    tools_obj = RepoTools()
    result: dict = {}

    async def reward_fn(history):
        msgs = history_to_messages(history)
        trace = Trace(task_id=task.task_id, profile="smoke-qwen4b-step0", messages=msgs,
                      stats=TraceStats(turns=sum(1 for m in msgs if m.role == "assistant"), tool_calls=tools_obj.calls, tool_errors=tools_obj.errors,
                                       prompt_tokens=result.get("prompt_tokens", 0), files_read=list(tools_obj.files_read)))
        t0 = time.time()
        r = await grade(task, trace)
        m = metrics(r, trace, task)
        result.update(grade=r.model_dump(), metrics=m, trace=trace.model_dump(mode="json"), grade_seconds=time.time() - t0)
        return (0.0 if r.reward != r.reward else r.reward), m

    renderer = tk.renderer(MODEL)
    tools = [tools_obj.find_symbol, tools_obj.read_file]
    prefix = renderer.create_conversation_prefix_with_tools(tools=[t.to_spec() for t in tools],
                                                            system_prompt=system_prompt(max_tool_calls=budget_calls, max_answer_tokens=b.max_answer_tokens))
    initial = [*prefix, {"role": "user", "content": user_prompt(REPO_ID, REPO_MAP, task.question)}]
    env = build_agent_tool_env(renderer=renderer, tools=tools, initial_messages=initial, reward_fn=reward_fn, max_turns=b.max_turns,
                               max_tool_calls=budget_calls, max_generation_tokens=1024, max_trajectory_tokens=24000, model_name=MODEL)
    return env, tools_obj, result


async def run(index: int):
    import tinker
    tasks = read_all(paths.TASKS_EVAL / "smoke_sweqa_flask.jsonl", Task)
    task = tasks[index]
    print(f"task {task.task_id} ({task.task_type}/{task.source}): {task.question[:140]}", flush=True)
    env, tools_obj, result = build(task, budget_calls=6)
    sc, tokenizer = tk.sampling_client(MODEL), tk.tokenizer(MODEL)

    async def maybe(x):
        return await x if inspect.isawaitable(x) else x

    obs, stop = await maybe(env.initial_observation())
    t0, turn, done, prompt_tokens = time.time(), 0, False, 0
    while not done:
        turn += 1
        prompt_tokens += obs.length
        result["prompt_tokens"] = prompt_tokens
        resp = await asyncio.wait_for(sc.sample_async(prompt=obs, num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=1024, temperature=1.0, stop=stop)), 90)
        toks = resp.sequences[0].tokens
        print(f"--- turn {turn}: prompt {obs.length} tok, sampled {len(toks)} tok: {tokenizer.decode(toks)[-160:]!r}", flush=True)
        res = await maybe(env.step(toks))
        done = res.episode_done
        if not done:
            obs, stop = res.next_observation, res.next_stop_condition
    print(f"=== episode done: {turn} turns, {tools_obj.calls} tool calls, {prompt_tokens} prompt tokens, {time.time() - t0:.0f}s", flush=True)
    g = result.get("grade")
    if g:
        print(f"grade: reward={g['reward']} gate={g['gate_failed']} notes={g['notes']}\ncomponents={g['components']}\njudge {result['grade_seconds']:.1f}s", flush=True)
        out = paths.TRACES / "smoke_grade" / f"{task.task_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"trace": result["trace"], "grade": g, "metrics": result["metrics"]}, indent=1))
        print("wrote", out, flush=True)
    else:
        print("reward_fn was not called (episode ended without grading); env metrics:", res.metrics, flush=True)


if __name__ == "__main__":
    asyncio.run(run(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
