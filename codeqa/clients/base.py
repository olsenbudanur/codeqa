"""ModelClient protocol: messages + tool specs in, one assistant Message out."""
from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv

from codeqa.shared.contracts import EndpointProfile, Message

load_dotenv()


@runtime_checkable
class ModelClient(Protocol):
    profile: EndpointProfile

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float = 1.0,
    ) -> Message: ...


def key_from_env(name: str | None, default_env: str) -> str:
    env = name or default_env
    val = os.environ.get(env)
    if not val:
        raise RuntimeError(f"{env} is not set (add it to .env)")
    return val


def make_client(profile: EndpointProfile) -> ModelClient:
    if profile.kind == "openai":
        from codeqa.clients.openai_compat import OpenAICompatClient
        return OpenAICompatClient(profile)
    if profile.kind == "anthropic":
        from codeqa.clients.anthropic import AnthropicClient
        return AnthropicClient(profile)
    if profile.kind == "tinker":
        from codeqa.clients.tinker import TinkerChatClient
        return TinkerChatClient(profile)
    raise ValueError(f"no chat client for kind={profile.kind}")
