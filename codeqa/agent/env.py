"""RepoEnv: one repo + one task + one profile = the environment every lane runs against. (C3)

- `initial_messages()` is client-agnostic (system rules + map + question). The Tinker client and
  `make_cookbook_env` prepend the renderer's tool prefix themselves; Anthropic/OpenAI pass `specs()` as tools.
- `make_cookbook_env(reward_fn)` wraps the cookbook `build_agent_tool_env` with the task's budget. The reward
  function receives `(history, env)` so the grader can read `env.files_read()`.
- `trace_from_history()` turns a cookbook history into a C6 `Trace` so grader and product share one type.
"""
from __future__ import annotations

import hashlib
from typing import Any, Awaitable, Callable

from codeqa.agent.curation import DEFAULT_CAPS, Caps
from codeqa.agent.prompts import system_prompt, user_prompt
from codeqa.agent.tools import RepoTools
from codeqa.agent.variants import AgentVariant, repo_map_text, resolve
from codeqa.shared.contracts import (Budget, EndpointProfile, Message, Span, StopReason, Task, TaskType, Trace,
                                     TraceStats)

RewardFn = Callable[[list[dict[str, Any]], "RepoEnv"], Awaitable[tuple[float, dict[str, float]]]]


class RepoEnv:
    def __init__(self, task: Task, profile: EndpointProfile, caps: Caps = DEFAULT_CAPS,
                 variant: str | AgentVariant | None = None):
        self.task = task
        self.profile = profile
        self.repo_id = task.repo_id
        self.budget: Budget = task.effective_budget()
        # explicit arg > the profile's own variant (a checkpoint is served with what it trained on) > CODEQA_AGENT_VARIANT > default
        self.variant: AgentVariant = resolve(variant or profile.variant)
        self.tools_obj = RepoTools(task.repo_id, max_tool_calls=self.budget.max_tool_calls, caps=caps)
        self.repo_map = repo_map_text(task.repo_id, self.variant)   # '' | ~1k structural tree | summarised map.txt

    @classmethod
    def from_question(cls, repo_id: str, question: str, profile: EndpointProfile, task_type: TaskType = "explain",
                      budget: Budget | None = None, variant: str | AgentVariant | None = None) -> "RepoEnv":
        """Ad-hoc env for the product: a question with no gold."""
        tid = "adhoc-" + hashlib.sha1(f"{repo_id}\n{question}".encode()).hexdigest()[:10]
        return cls(Task(task_id=tid, repo_id=repo_id, split="eval", question=question, task_type=task_type,
                        source="teacher", budget=budget), profile, variant=variant)

    # ------------------------------------------------------------------ what every client needs
    def tools(self) -> list:
        return self.tools_obj.tools(self.variant.tools)

    def specs(self) -> list[dict[str, Any]]:
        return self.tools_obj.specs(self.variant.tools)

    def initial_messages(self) -> list[Message]:
        return [
            Message(role="system", content=system_prompt(self.budget.max_tool_calls, self.budget.max_answer_tokens, self.variant.rules, max_turns=self.budget.max_turns)),
            Message(role="user", content=user_prompt(self.repo_id, self.repo_map, self.task.question)),
        ]

    def files_read(self) -> list[Span]:
        return list(self.tools_obj.files_read)

    @property
    def tool_calls_made(self) -> int:
        return self.tools_obj.calls

    @property
    def tool_errors(self) -> int:
        return self.tools_obj.errors

    def stop_reason_from(self, last_assistant: Message | None, turns: int) -> StopReason:
        if last_assistant is None:
            return "error"
        if last_assistant.parse_error:
            return "overflow" if "truncated" in last_assistant.parse_error else "parse_error"
        if not last_assistant.tool_calls:
            return "answer"
        if self.tools_obj.calls >= self.budget.max_tool_calls:
            return "budget"
        return "max_turns" if turns >= self.budget.max_turns else "answer"

    # ------------------------------------------------------------------ training path
    def make_cookbook_env(self, reward_fn: RewardFn, *, max_generation_tokens: int | None = None,
                          max_trajectory_tokens: int | None = None):
        from tinker_cookbook.tool_use.agent_tool_message_env import build_agent_tool_env

        from codeqa.clients import tinker as tk

        base = tk.base_model_of(self.profile)
        renderer = tk.renderer(base, self.profile.renderer)
        initial = tk.to_cookbook(tk.with_tool_prefix(renderer, self.initial_messages(), self.specs()))

        async def _reward(history):
            return await reward_fn(history, self)

        return build_agent_tool_env(
            renderer=renderer, tools=self.tools(), initial_messages=initial, reward_fn=_reward,
            max_turns=self.budget.max_turns, max_tool_calls=self.budget.max_tool_calls,
            max_generation_tokens=max_generation_tokens or self.profile.max_generation_tokens,
            max_trajectory_tokens=max_trajectory_tokens or self.profile.max_context,
            model_name=base,
        )

    def trace_from_history(self, history: list[dict[str, Any]], seconds: float = 0.0) -> Trace:
        """Cookbook history (incl. the tool-prefixed system message) -> C6 Trace. Token counts are not in the history."""
        from codeqa.clients.tinker import from_cookbook

        msgs: list[Message] = []
        for m in history:
            role = m["role"]
            if role == "assistant":
                msgs.append(from_cookbook(m))
            elif role == "tool":
                content = m.get("content", "")
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
                msgs.append(Message(role="tool", content=content, name=m.get("name"), call_id=m.get("tool_call_id")))
            else:
                content = m.get("content", "")
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
                msgs.append(Message(role=role, content=content))
        assistants = [m for m in msgs if m.role == "assistant"]
        last = assistants[-1] if assistants else None
        answer = last.content if last is not None and not last.tool_calls else ""
        return Trace(task_id=self.task.task_id, profile=self.profile.name, messages=msgs, answer=answer,
                     stats=TraceStats(turns=len(assistants), tool_calls=self.tools_obj.calls, tool_errors=self.tools_obj.errors,
                                      files_read=self.files_read(), stop_reason=self.stop_reason_from(last, len(assistants)),
                                      seconds=seconds))
