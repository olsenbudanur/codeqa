"""run_episode: the same agent loop as training, driven by any ModelClient (Tinker, Anthropic, OpenAI). (C3, C6, C9)

Used by the product API (streams C9 events), the teacher generator, and the evals. Training does not use this;
it goes through `RepoEnv.make_cookbook_env`, and both paths share the prompt, the tools and the budget.
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Awaitable, Callable

from tinker_cookbook.tool_use.types import ToolInput

from codeqa.agent.env import RepoEnv
from codeqa.clients.base import ModelClient
from codeqa.shared import paths
from codeqa.shared.contracts import CITATION_RE, Message, SSEEvent, Span, Trace, TraceStats

OnEvent = Callable[[SSEEvent], Any | Awaitable[Any]] | None
CHAT_TIMEOUT = 180.0


def _first_sentence(text: str | None, n: int = 140) -> str:
    if not text:
        return ""
    s = " ".join(text.split())
    s = s.split(". ", 1)[0]
    return s if len(s) <= n else s[: n - 1] + "…"


def citation_items(answer: str, files_read: list[Span], known_paths: set[str]) -> list[dict[str, Any]]:
    """Light check for the product stream: exists (path in snapshot) and verified (range inside a read range).
    The grader's `check_citations` (C7) is the authority; this only feeds the UI badges."""
    items = []
    for m in CITATION_RE.finditer(answer):
        path, start, end = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        exists = path in known_paths
        grounded = any(path == s.path and start >= s.start and end <= s.end for s in files_read)
        items.append({"path": path, "start": start, "end": end, "exists": exists, "verified": exists and grounded})
    return items


async def _emit(on_event: OnEvent, type_: str, **payload: Any) -> None:
    if on_event is None:
        return
    r = on_event(SSEEvent(type=type_, payload=payload))
    if inspect.isawaitable(r):
        await r


def trim_old_tool_outputs(messages: list[Message], keep_turns: int) -> list[Message]:
    """Product-side context control: tool outputs older than the last `keep_turns` assistant turns are replaced
    by a one-line stub in what the model sees. The trace keeps the full messages; grounding still holds because
    `files_read` records what was shown. Not used in training (the cookbook env owns the history there)."""
    turn_of: list[int] = []
    t = 0
    for m in messages:
        if m.role == "assistant":
            t += 1
        turn_of.append(t)
    cutoff = max(t - keep_turns, 0)
    out: list[Message] = []
    for m, tm in zip(messages, turn_of):
        if m.role == "tool" and tm <= cutoff and len(m.content) > 200:
            head = m.content.splitlines()[0][:120] if m.content else ""
            out.append(m.model_copy(update={"content": f"{head}\n[earlier {m.name or 'tool'} output trimmed: {len(m.content)} chars; re-run the call if you need it]"}))
        else:
            out.append(m)
    return out


async def run_episode(env: RepoEnv, client: ModelClient, on_event: OnEvent = None, *,
                      temperature: float = 1.0, max_tokens: int | None = None,
                      trim_tool_outputs_after: int | None = None) -> Trace:
    """`trim_tool_outputs_after=N` keeps only the last N turns' tool outputs in the prompt (product latency/cost knob;
    default None = full context, identical to training)."""
    t0 = time.time()
    messages: list[Message] = env.initial_messages()
    specs = env.specs()
    tools = {t.name: t for t in env.tools()}
    budget = env.budget
    stats = TraceStats()
    answer = ""
    stop: str = "max_turns"
    known_paths = {f.path for f in env.tools_obj.manifest.files}
    try:
        for turn in range(1, budget.max_turns + 1):
            stats.turns = turn
            import asyncio
            visible = trim_old_tool_outputs(messages, trim_tool_outputs_after) if trim_tool_outputs_after else messages
            msg = await asyncio.wait_for(client.chat(visible, tools=specs, max_tokens=max_tokens or client.profile.max_generation_tokens,
                                                     temperature=temperature), timeout=CHAT_TIMEOUT)
            stats.prompt_tokens += int(msg.usage.get("prompt_tokens", 0))
            stats.completion_tokens += int(msg.usage.get("completion_tokens", 0))
            messages.append(msg)
            if msg.thinking:
                await _emit(on_event, "thinking", text=msg.thinking)
            if msg.parse_error:
                stop = "overflow" if "truncated" in msg.parse_error else "parse_error"
                await _emit(on_event, "error", message=msg.parse_error[:300])
                break
            if not msg.tool_calls:
                answer = msg.content
                stop = "answer"
                break
            remaining = budget.max_tool_calls - env.tool_calls_made
            calls = msg.tool_calls[: max(remaining, 0)]
            dropped = len(msg.tool_calls) - len(calls)
            for i, tc in enumerate(calls):
                tc.call_id = tc.call_id or f"call_{turn}_{i}"
                await _emit(on_event, "tool_call", name=tc.name, args=tc.args, why=_first_sentence(msg.thinking))
                tool = tools.get(tc.name)
                if tool is None:
                    env.tools_obj.calls += 1
                    env.tools_obj.errors += 1
                    text = f"ERROR unknown_tool: {tc.name!r}. Available: {', '.join(tools)}"
                else:
                    result = await tool.run(ToolInput(arguments=tc.args, call_id=tc.call_id))
                    text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in result.messages[0]["content"]) \
                        if isinstance(result.messages[0]["content"], list) else str(result.messages[0]["content"])
                messages.append(Message(role="tool", name=tc.name, content=text, call_id=tc.call_id))
                await _emit(on_event, "tool_result", name=tc.name, summary=text.splitlines()[0][:160] if text else "", chars=len(text),
                            error=text.startswith("ERROR"))
            if dropped:
                # a plain user message: it answers no tool_use, so it must not be a tool-role message (Anthropic rejects
                # orphan tool_result ids; Qwen renders user and tool blocks the same way)
                messages.append(Message(role="user", content=f"[{dropped} tool call(s) dropped: no tool calls remaining. Answer now.]"))
            if remaining <= 0:
                stop = "budget"
                break
        else:
            stop = "max_turns"
    except Exception as e:  # noqa: BLE001
        stop = "error"
        await _emit(on_event, "error", message=f"{type(e).__name__}: {e}"[:300])
    stats.tool_calls = env.tool_calls_made
    stats.tool_errors = env.tool_errors
    stats.files_read = env.files_read()
    stats.stop_reason = stop  # type: ignore[assignment]
    stats.seconds = round(time.time() - t0, 2)
    trace = Trace(task_id=env.task.task_id, profile=client.profile.name, messages=messages, stats=stats, answer=answer)
    if answer:
        await _emit(on_event, "answer", markdown=answer)
        await _emit(on_event, "citations", items=citation_items(answer, stats.files_read, known_paths))
    await _emit(on_event, "stats", tool_calls=stats.tool_calls, tool_errors=stats.tool_errors, prompt_tokens=stats.prompt_tokens,
                completion_tokens=stats.completion_tokens, seconds=stats.seconds, stop_reason=stop, turns=stats.turns)
    await _emit(on_event, "done")
    return trace


def save_trace(trace: Trace, run: str = "dev"):
    out = paths.TRACES / run / f"{trace.task_id}__{trace.profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(trace.model_dump_json(indent=1))
    return out
