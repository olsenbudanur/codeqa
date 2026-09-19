"""Agent harness per profile: which tools, prompt and budget an ask runs with. The v3 round harness (bash_v3,
scripts/arm.py) has no call cap: a context cap, a message cap and several commands per message, plus grep
self-healing and pipeline reads. Training turns it on with process env vars; the product serves many checkpoints
from one process, so it applies the same knobs per request. Shared by /ask and /judge."""
from __future__ import annotations

import os
from typing import Any

from codeqa.shared.contracts import DEFAULT_BUDGETS, UNLIMITED_CALLS, Budget, EndpointProfile, TaskType

V3_VARIANTS = {"bash_v3"}
V3_CONTEXT_TOKENS = int(os.environ.get("CODEQA_CAPS_CONTEXT", "32000"))
V3_MESSAGES = int(os.environ.get("CODEQA_CAPS_MESSAGES", "24"))
V3_COMMANDS = int(os.environ.get("CODEQA_CAPS_COMMANDS", "4"))


def is_v3(profile: EndpointProfile, variant: str | None = None) -> bool:
    """`variant` overrides the profile's own agent for one ask (compare page: every column on the same harness)."""
    return (variant or profile.variant or "") in V3_VARIANTS


def harness_budget(profile: EndpointProfile, task_type: TaskType, variant: str | None = None) -> Budget | None:
    """None = the task type's default budget (call caps). v3 profiles get the round harness caps."""
    if not is_v3(profile, variant):
        return None
    base = DEFAULT_BUDGETS[task_type]
    return Budget(max_tool_calls=UNLIMITED_CALLS, max_turns=V3_MESSAGES, max_answer_tokens=base.max_answer_tokens,
                  max_context_tokens=V3_CONTEXT_TOKENS, max_commands_per_turn=V3_COMMANDS)


def apply_harness_knobs(profile: EndpointProfile, variant: str | None = None) -> None:
    """Per-task switches the bash tool reads (contextvars, so concurrent asks on other variants are unaffected)."""
    from codeqa.agent import shell
    on = is_v3(profile, variant)
    shell.HEAL_OVERRIDE.set(on)
    shell.PIPELINES_OVERRIDE.set(on)


def harness_row(profile: EndpointProfile) -> dict[str, Any]:
    """What the picker shows next to a profile: the variant and, for v3, its caps."""
    from codeqa.agent.variants import resolve as resolve_variant
    v = resolve_variant(profile.variant)
    row: dict[str, Any] = {"variant": v.name, "tools": list(v.tools)}
    if is_v3(profile):
        row["harness"] = {"rounds": True, "context_tokens": V3_CONTEXT_TOKENS, "messages": V3_MESSAGES, "commands_per_message": V3_COMMANDS}
    return row
