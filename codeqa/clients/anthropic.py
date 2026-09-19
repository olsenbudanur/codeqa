"""Anthropic client for the teacher, blind verifier, judge, and summaries."""
from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from codeqa.clients.base import key_from_env
from codeqa.shared.contracts import EndpointProfile, Message, ToolCall


def _to_anthropic(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system = ""
    out: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()   # tool_result blocks must answer a real tool_use; orphans (e.g. the driver's budget notice) become user text
    for m in messages:
        if m.role == "system":
            system += m.content + "\n"
        elif m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for i, tc in enumerate(m.tool_calls):
                cid = tc.call_id or f"toolu_{i}"
                seen_call_ids.add(cid)
                blocks.append({"type": "tool_use", "id": cid, "name": tc.name, "input": tc.args})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif m.role == "tool":
            if m.call_id not in seen_call_ids:
                block = {"type": "text", "text": m.content}
            else:
                block = {"type": "tool_result", "tool_use_id": m.call_id, "content": m.content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    _close_open_tool_uses(out)
    return system.strip(), out


DROPPED_NOTICE = "[tool call dropped: no tool calls remaining. Answer now.]"


def _close_open_tool_uses(out: list[dict[str, Any]]) -> None:
    """Anthropic requires a tool_result for every tool_use in the preceding assistant turn. The driver drops calls
    beyond the budget without answering them, so synthesize a result for each unanswered id."""
    for i, m in enumerate(out):
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        ids = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
        if not ids:
            continue
        nxt = out[i + 1] if i + 1 < len(out) else None
        answered = {b.get("tool_use_id") for b in (nxt["content"] if nxt and nxt["role"] == "user" and isinstance(nxt["content"], list) else [])
                    if isinstance(b, dict) and b.get("type") == "tool_result"}
        missing = [cid for cid in ids if cid not in answered]
        if not missing:
            continue
        blocks = [{"type": "tool_result", "tool_use_id": cid, "content": DROPPED_NOTICE} for cid in missing]
        if nxt and nxt["role"] == "user" and isinstance(nxt["content"], list):
            nxt["content"] = blocks + nxt["content"]          # tool_results must come first in the user turn
        elif nxt and nxt["role"] == "user":
            nxt["content"] = blocks + [{"type": "text", "text": nxt["content"]}]
        else:
            out.insert(i + 1, {"role": "user", "content": blocks})


class AnthropicClient:
    def __init__(self, profile: EndpointProfile):
        self.profile = profile
        import os
        headers = {}
        if os.environ.get("ANTHROPIC_WORKSPACE_ID"):   # keys not scoped to a workspace need this header
            headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
        self._client = AsyncAnthropic(api_key=key_from_env(profile.api_key_env, "ANTHROPIC_API_KEY"), default_headers=headers or None)

    async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0) -> Message:
        system, msgs = _to_anthropic(messages)
        kwargs: dict[str, Any] = dict(
            model=self.profile.model,
            max_tokens=max_tokens or self.profile.max_generation_tokens,
            messages=msgs,
        )
        # anthropic>=1.7 no longer accepts `temperature`; determinism for judging comes from prompt + discrete scores
        if not self.profile.thinking:
            # Sonnet 5 thinks adaptively by default and can spend the whole max_tokens on it (empty reply); judges set thinking=False
            kwargs["thinking"] = {"type": "disabled"}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t.get("description", ""), "input_schema": t.get("parameters", {"type": "object", "properties": {}})}
                for t in tools
            ]
        # Prompt caching: two breakpoints. The system block covers tools + system (~4k tokens: schemas, rules, repo map);
        # the last message block covers the conversation so far. Each turn's prompt is the previous prompt plus a
        # little, so every turn after the first reads the prefix from cache (billed at ~10 %).
        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if msgs:
            last = msgs[-1]
            if isinstance(last["content"], str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            if isinstance(last["content"], list) and last["content"]:
                last["content"][-1] = {**last["content"][-1], "cache_control": {"type": "ephemeral"}}
        resp = await self._client.messages.create(**kwargs)
        text, thinking, calls = "", None, []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "thinking":
                thinking = (thinking or "") + block.thinking
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, args=dict(block.input), call_id=block.id))
        usage = {}
        if getattr(resp, "usage", None) is not None:
            u = resp.usage
            cached = int(getattr(u, "cache_read_input_tokens", 0) or 0)
            written = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
            # prompt_tokens = everything the model read (uncached + cache write + cache read), comparable to the Tinker client
            usage = {"prompt_tokens": int(u.input_tokens) + cached + written, "completion_tokens": int(u.output_tokens),
                     "cache_read_tokens": cached, "cache_write_tokens": written, "uncached_prompt_tokens": int(u.input_tokens)}
        return Message(role="assistant", content=text, thinking=thinking, tool_calls=calls, usage=usage)
