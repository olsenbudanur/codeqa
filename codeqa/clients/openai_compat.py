"""OpenAI-compatible chat client: vLLM on Modal, or any /v1 endpoint."""
from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from codeqa.clients.base import key_from_env
from codeqa.shared.contracts import EndpointProfile, Message, ToolCall


def _to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.call_id or m.name or "call_0", "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {"id": tc.call_id or f"call_{i}", "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                    for i, tc in enumerate(m.tool_calls)
                ],
            })
        else:
            out.append({"role": m.role, "content": m.content})
    return out


class OpenAICompatClient:
    def __init__(self, profile: EndpointProfile):
        self.profile = profile
        api_key = "EMPTY"
        if profile.api_key_env:
            api_key = key_from_env(profile.api_key_env, profile.api_key_env)
        self._client = AsyncOpenAI(base_url=profile.base_url, api_key=api_key)

    async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0) -> Message:
        kwargs: dict[str, Any] = dict(
            model=self.profile.model,
            messages=_to_openai(messages),
            max_tokens=max_tokens or self.profile.max_generation_tokens,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        calls = [
            ToolCall(name=tc.function.name, args=json.loads(tc.function.arguments or "{}"), call_id=tc.id)
            for tc in (choice.tool_calls or [])
        ]
        thinking = getattr(choice, "reasoning_content", None) or getattr(choice, "reasoning", None)
        return Message(role="assistant", content=choice.content or "", thinking=thinking, tool_calls=calls)
