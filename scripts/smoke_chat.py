"""A1 check: TinkerChatClient round-trips a tool-calling conversation.

Stage 1 (offline, deterministic): render a conversation that already contains an assistant tool call and
a tool result, confirm the prompt text carries them in Qwen3.5 syntax; hand-encode a model reply with a
tool call, parse it, and confirm the parsed ToolCall equals what was written.
Stage 2 (live, ~20s): a scripted three-turn conversation on the flask snapshot with two real tools.

Run: uv run python -u -m scripts.smoke_chat   (needs TINKER_API_KEY)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from codeqa.agent.indexing.index import load_symbols
from codeqa.agent.prompts import system_prompt, user_prompt
from codeqa.clients.base import make_client
from codeqa.clients.tinker import from_cookbook, to_cookbook, with_tool_prefix
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Message, ToolCall

MODEL = "Qwen/Qwen3.5-4B"
REPO_ID = "pallets__flask__85c5d93"
QUESTION = "Where is the Flask application class defined, and what class does it inherit from? Cite the lines."

TOOLS: list[dict[str, Any]] = [
    {"name": "find_symbol", "description": "Find a class or function by exact name; returns path and line range.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Exact symbol name"}}, "required": ["name"]}},
    {"name": "read_file", "description": "Read a line range of a file (at most 150 lines), numbered.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Repo-relative path"},
         "start": {"type": "integer", "description": "First line, 1-based"},
         "end": {"type": "integer", "description": "Last line, inclusive"}}, "required": ["path", "start", "end"]}},
]


def run_tool(tc: ToolCall, symbols, root) -> str:
    if tc.name == "find_symbol":
        hits = [s for s in symbols if s.name == tc.args.get("name")][:20]
        return "\n".join(f"{s.path}:L{s.start}-L{s.end}  {s.kind}  {s.signature}" for s in hits) or f"No symbol named {tc.args.get('name')!r}."
    if tc.name == "read_file":
        p = root / str(tc.args.get("path", ""))
        if not p.is_file():
            return f"ERROR not_found: {tc.args.get('path')}"
        src = p.read_text().splitlines()
        start = max(1, int(tc.args.get("start", 1))); end = min(int(tc.args.get("end", start)), start + 149, len(src))
        return "\n".join(f"{i:5d} | {src[i-1]}" for i in range(start, end + 1)) + f"\n(total {len(src)} lines)"
    return f"ERROR unknown_tool: {tc.name}"


def check_args(tc: ToolCall) -> None:
    spec = next(t for t in TOOLS if t["name"] == tc.name)
    props = spec["parameters"]["properties"]
    missing = [k for k in spec["parameters"].get("required", []) if k not in tc.args]
    extra = [k for k in tc.args if k not in props]
    assert not missing and not extra, f"args mismatch for {tc.name}: missing={missing} extra={extra} got={tc.args}"


def stage_offline(client) -> None:
    rend, tok = client.renderer, client.tokenizer
    sys_msg = Message(role="system", content=system_prompt(6, 200))
    user = Message(role="user", content=user_prompt(REPO_ID, "src/flask/app.py  Flask", QUESTION))
    call = ToolCall(name="read_file", args={"path": "src/flask/app.py", "start": 81, "end": 90})
    convo = [sys_msg, user,
             Message(role="assistant", content="", thinking="Need the class header.", tool_calls=[call]),
             Message(role="tool", name="read_file", content="   81 | class Flask(App):")]
    prefixed = with_tool_prefix(rend, convo, TOOLS)
    assert prefixed[0].content.startswith("# Tools") and "<tools>" in prefixed[0].content
    assert with_tool_prefix(rend, prefixed, TOOLS)[0].content == prefixed[0].content, "prefix not idempotent"
    prompt = client.build_prompt(convo, TOOLS)
    text = tok.decode([t for ch in prompt.chunks for t in ch.tokens])
    # NOTE: the Qwen3.5 renderer strips string content, so a tool result's first line loses leading
    # whitespace ("   81 | x" -> "81 | x"). Tool outputs must not rely on leading padding (A3).
    for needle in ["<function=read_file>", "<parameter=path>\nsrc/flask/app.py", "<parameter=start>\n81",
                   "<tool_response>\n81 | class Flask(App):", "<think>\nNeed the class header.\n</think>",
                   "<|im_start|>assistant\n<think>\n"]:
        assert needle in text, f"prompt missing {needle!r}"
    print(f"[offline] prompt renders tool call, thinking history and tool result in Qwen3.5 syntax ({prompt.length} tokens)")

    reply = ("Looking it up.\n</think>\n\n<tool_call>\n<function=find_symbol>\n<parameter=name>\nFlask\n</parameter>\n</function>\n</tool_call><|im_end|>")
    tokens = tok.encode(reply, add_special_tokens=False)
    parsed, term = rend.parse_response(tokens)
    msg = from_cookbook(parsed, term)
    assert term.is_clean and msg.parse_error is None, (term, msg.parse_error)
    assert msg.thinking == "Looking it up." and msg.tool_calls == [ToolCall(name="find_symbol", args={"name": "Flask"})], msg
    # and back: our Message -> cookbook -> prompt text contains the same call
    again = tok.decode([t for ch in rend.build_generation_prompt(to_cookbook([user, msg])).chunks for t in ch.tokens])
    assert "<function=find_symbol>\n<parameter=name>\nFlask\n</parameter>" in again
    print("[offline] hand-written reply parses to the same ToolCall and re-renders identically")

    bad = tok.encode("<tool_call>\n<function=find_symbol>\n<parameter=name>\nFlask\n</function>\n</tool_call><|im_end|>", add_special_tokens=False)
    m2 = from_cookbook(*rend.parse_response(bad))
    assert m2.parse_error and not m2.tool_calls, m2
    print(f"[offline] malformed call surfaces as parse_error: {m2.parse_error[:60]}...")


async def stage_live(client) -> None:
    root, symbols = paths.repo_dir(REPO_ID), load_symbols(REPO_ID)
    convo = [Message(role="system", content=system_prompt(6, 200)),
             Message(role="user", content=user_prompt(REPO_ID, "src/flask/\n  app.py  Flask\n  sansio/app.py  App\n", QUESTION))]
    t0 = time.time()
    for turn in range(1, 5):
        msg = await asyncio.wait_for(client.chat(convo, tools=TOOLS, max_tokens=1024), timeout=60)
        convo.append(msg)
        print(f"\n--- turn {turn} | usage {msg.usage} | thinking {len(msg.thinking or '')} chars | parse_error={msg.parse_error}")
        if msg.thinking:
            print("  think:", msg.thinking[:200].replace("\n", " "), "...")
        if msg.content:
            print("  text:", msg.content[:400])
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            check_args(tc)
            out = run_tool(tc, symbols, root)
            print(f"  call: {tc.name}({json.dumps(tc.args)}) -> {out.splitlines()[0][:100]} ({len(out)} chars)")
            convo.append(Message(role="tool", name=tc.name, content=out))
    n_calls = sum(len(m.tool_calls) for m in convo if m.role == "assistant")
    print(f"\n[live] {turn} turns, {n_calls} tool calls, {time.time()-t0:.0f}s; final answer above")
    assert n_calls >= 1, "model made no tool call"
    assert not convo[-1].tool_calls, "conversation did not reach a final answer within 4 turns"


async def main() -> None:
    profile = EndpointProfile(name="qwen4b-base", kind="tinker", model=MODEL, max_generation_tokens=1024)
    t0 = time.time()
    client = await asyncio.wait_for(asyncio.to_thread(make_client, profile), timeout=60)
    print(f"client ready in {time.time()-t0:.1f}s (renderer {type(client.renderer).__name__})")
    stage_offline(client)
    await stage_live(client)
    print("\nSMOKE_CHAT PASSED")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print("SMOKE_CHAT FAILED:", e); sys.exit(1)
