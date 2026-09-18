"""Tinker: service client, sampling clients, tokenizer + renderer, and the TinkerChatClient.

The chat client renders with the same cookbook renderer the trainer uses, so what the product
and the teacher see is token-for-token what the RL loop trains on.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from codeqa.shared.contracts import EndpointProfile, Message, ToolCall

load_dotenv()

CHECKPOINT_PREFIX = "tinker://"


def _require_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set (add it to .env)")


@lru_cache(maxsize=1)
def service_client():
    _require_key()
    import tinker
    # fail fast: the SDK retries 402/5xx for a long time by default
    return tinker.ServiceClient(max_retries=1) if "max_retries" in tinker.ServiceClient.__init__.__code__.co_varnames else tinker.ServiceClient()


@lru_cache(maxsize=8)
def sampling_client(model: str):
    """One sampling client per model. `model` is a base name (Qwen/Qwen3.5-4B) or a tinker:// checkpoint path."""
    sc = service_client()
    if model.startswith(CHECKPOINT_PREFIX):
        return sc.create_sampling_client(model_path=model)
    return sc.create_sampling_client(base_model=model)


@lru_cache(maxsize=4)
def tokenizer(model_name: str):
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    return get_tokenizer(model_name)


@lru_cache(maxsize=4)
def renderer(model_name: str, renderer_name: str | None = None):
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer
    name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    r = get_renderer(name, tokenizer(model_name), model_name=model_name)
    # Keep thinking in history. The RL env re-renders the whole history every step, so with the
    # default (strip=True) turn N+1's observation would drop turn N's thinking, breaking the
    # sequence-extension property. For our shape (one user query, then a tool loop) the HF Qwen3.5
    # template keeps thinking too. The product client uses this same renderer, so it matches training.
    if hasattr(r, "strip_thinking_from_history"):
        r.strip_thinking_from_history = False
    return r


def base_model_of(profile: EndpointProfile) -> str:
    """The HF base model name that owns the tokenizer/renderer for this profile."""
    if profile.base_model:
        return profile.base_model
    if profile.model.startswith(CHECKPOINT_PREFIX):
        raise ValueError(f"profile {profile.name!r} points at a checkpoint; set base_model (e.g. Qwen/Qwen3.5-4B)")
    return profile.model


# ---------------------------------------------------------------------------
# Message conversion: our contracts.Message <-> cookbook renderers.Message
# ---------------------------------------------------------------------------

def to_cookbook(messages: list[Message]) -> list[dict[str, Any]]:
    from tinker_cookbook.renderers.base import ToolCall as CbToolCall
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant":
            content: Any = m.content
            if m.thinking:
                parts: list[dict[str, Any]] = [{"type": "thinking", "thinking": m.thinking}]
                if m.content:
                    parts.append({"type": "text", "text": m.content})
                content = parts
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    CbToolCall(id=tc.call_id, function=CbToolCall.FunctionBody(name=tc.name, arguments=json.dumps(tc.args)))
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            msg = {"role": "tool", "content": m.content}
            if m.name:
                msg["name"] = m.name
            if m.call_id:
                msg["tool_call_id"] = m.call_id
            out.append(msg)
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def from_cookbook(msg: dict[str, Any], termination: Any = None) -> Message:
    content = msg.get("content", "")
    thinking: str | None = None
    if isinstance(content, list):
        text = "".join(p["text"] for p in content if p.get("type") == "text")
        th = "\n".join(p["thinking"] for p in content if p.get("type") == "thinking")
        thinking = th.strip() or None
    else:
        text = content or ""
    calls = [
        ToolCall(name=tc.function.name, args=json.loads(tc.function.arguments or "{}"), call_id=tc.id)
        for tc in msg.get("tool_calls", []) or []
    ]
    errors = [f"{u.error}: {u.raw_text[:300]}" for u in msg.get("unparsed_tool_calls", []) or []]
    if termination is not None and not termination.is_clean:
        errors.append("truncated: generation ended without the stop token (max_tokens hit?)")
    return Message(role="assistant", content=text.strip(), thinking=thinking, tool_calls=calls,
                   parse_error="; ".join(errors) or None)


def with_tool_prefix(rend, messages: list[Message], tools: list[dict[str, Any]] | None) -> list[Message]:
    """Replace the leading system message with the renderer's tool-declaration system prefix.

    Idempotent: if the system message already carries a <tools> block it is returned unchanged.
    The cookbook env does NOT inject tool specs; without this prefix the model invents its own
    tool syntax. The Anthropic/OpenAI clients pass the same specs as `tools=` and ignore this.
    """
    if not tools:
        return messages
    system = ""
    rest = messages
    if messages and messages[0].role == "system":
        system, rest = messages[0].content, messages[1:]
        if "<tools>" in system:
            return messages
    prefix = rend.create_conversation_prefix_with_tools(tools=list(tools), system_prompt=system)
    return [Message(role=m["role"], content=m["content"]) for m in prefix] + list(rest)


# ---------------------------------------------------------------------------
# Chat client
# ---------------------------------------------------------------------------

class TinkerChatClient:
    """ModelClient over Tinker sampling. Renders with the cookbook renderer, parses with it too."""

    def __init__(self, profile: EndpointProfile):
        self.profile = profile
        base = base_model_of(profile)
        self.renderer = renderer(base, profile.renderer)
        self.tokenizer = tokenizer(base)
        self._sc = sampling_client(profile.model)
        self._stop = self.renderer.get_stop_sequences()

    def build_prompt(self, messages: list[Message], tools: list[dict[str, Any]] | None = None):
        return self.renderer.build_generation_prompt(to_cookbook(with_tool_prefix(self.renderer, messages, tools)))

    async def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
                   max_tokens: int | None = None, temperature: float = 1.0) -> Message:
        import tinker
        prompt = self.build_prompt(messages, tools)
        params = tinker.SamplingParams(max_tokens=max_tokens or self.profile.max_generation_tokens,
                                       temperature=temperature, stop=self._stop)
        resp = await self._sc.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        tokens = list(resp.sequences[0].tokens)
        parsed, termination = self.renderer.parse_response(tokens)
        out = from_cookbook(parsed, termination)
        out.usage = {"prompt_tokens": prompt.length, "completion_tokens": len(tokens)}
        return out


async def sample_text(base_model: str, messages: list[dict[str, Any]], max_tokens: int = 64, temperature: float = 1.0) -> str:
    """One short sample through the cookbook renderer. Used by the client smoke test only."""
    import tinker
    r = renderer(base_model)
    prompt = r.build_generation_prompt(messages)
    sc = sampling_client(base_model)
    resp = await sc.sample_async(prompt=prompt, num_samples=1,
                                 sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature))
    return tokenizer(base_model).decode(resp.sequences[0].tokens)
