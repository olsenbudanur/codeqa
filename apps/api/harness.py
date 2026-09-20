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


def _run_of_sampler(model_path: str) -> str | None:
    """Which run produced a tinker:// sampler path (from data/logs/<run>/checkpoints.jsonl); None for base models."""
    import time
    from apps.api.workshop import read_jsonl
    from codeqa.shared import paths
    if not model_path.startswith("tinker://") or not paths.LOGS.exists():
        return None
    now = time.time()
    if now - _sampler_runs["at"] > 30:   # checkpoints.jsonl rows are appended inside existing run dirs, so rescan by time
        out: dict[str, str] = {}
        for run in paths.LOGS.iterdir():
            for r in read_jsonl(run / "checkpoints.jsonl"):
                if r.get("sampler_path"):
                    out[r["sampler_path"]] = run.name
        _sampler_runs.update(at=now, map=out)
    return _sampler_runs["map"].get(model_path)


_sampler_runs: dict[str, Any] = {"at": 0.0, "map": {}}


def run_caps(run: str | None) -> dict[str, int]:
    """The round-harness caps a run trained with. Arms record them as env vars in `scripts/arm.py` (`CODEQA_CAPS_*`);
    the run name is `<phase>_<arm>` (p6_bash_v4 -> bash_v4). Falls back to the process defaults (v3: 32k / 24 / 4)."""
    caps = {"context": V3_CONTEXT_TOKENS, "messages": V3_MESSAGES, "commands": V3_COMMANDS}
    if not run:
        return caps
    try:
        from scripts.arm import ARMS   # the experiment table, not core code; guarded so the API runs without it
    except Exception:  # noqa: BLE001
        return caps
    arm = run.split("_", 1)[1] if "_" in run else run
    env = ARMS.get(arm, (None, {}, 0))[1] if arm in ARMS else {}
    for key, name in (("context", "CODEQA_CAPS_CONTEXT"), ("messages", "CODEQA_CAPS_MESSAGES"), ("commands", "CODEQA_CAPS_COMMANDS")):
        if name in env:
            caps[key] = int(env[name])
    return caps


def harness_caps(profile: EndpointProfile) -> dict[str, int]:
    return run_caps(_run_of_sampler(profile.model))


def harness_budget(profile: EndpointProfile, task_type: TaskType, variant: str | None = None) -> Budget | None:
    """None = the task type's default budget (call caps). v3 profiles get the round harness caps their run trained with."""
    if not is_v3(profile, variant):
        return None
    base = DEFAULT_BUDGETS[task_type]
    caps = harness_caps(profile)
    return Budget(max_tool_calls=UNLIMITED_CALLS, max_turns=caps["messages"], max_answer_tokens=base.max_answer_tokens,
                  max_context_tokens=caps["context"], max_commands_per_turn=caps["commands"])


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
        caps = harness_caps(profile)
        row["harness"] = {"rounds": True, "context_tokens": caps["context"], "messages": caps["messages"], "commands_per_message": caps["commands"]}
    return row
