"""Agent variants for ablations: which tools, whether the repo map is in the prompt, which rules text.

`default` is the five-tool agent every lane runs today. Others are opt-in: pass `variant=` to RepoEnv or set
CODEQA_AGENT_VARIANT in the environment (so other lanes' CLIs pick it up without a flag). Rewards do not change:
grounding reads `files_read`, which every tool fills for the lines it showed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from codeqa.agent.prompts import SYSTEM_RULES, SYSTEM_RULES_BASH, SYSTEM_RULES_NOINDEX

ENV_VAR = "CODEQA_AGENT_VARIANT"


@dataclass(frozen=True)
class AgentVariant:
    name: str
    tools: tuple[str, ...]
    include_map: bool
    rules: str


VARIANTS: dict[str, AgentVariant] = {
    "default": AgentVariant("default", ("overview", "find_symbol", "grep", "read_file", "list_dir"), True, SYSTEM_RULES),
    "noindex": AgentVariant("noindex", ("grep", "read_file", "list_dir"), False, SYSTEM_RULES_NOINDEX),     # no map, no index tools
    "nomap": AgentVariant("nomap", ("overview", "find_symbol", "grep", "read_file", "list_dir"), False, SYSTEM_RULES),  # tools yes, map no
    "bash": AgentVariant("bash", ("bash",), True, SYSTEM_RULES_BASH),                                        # one shell tool, map kept
    "bash_nomap": AgentVariant("bash_nomap", ("bash",), False, SYSTEM_RULES_BASH),
}


def resolve(variant: str | AgentVariant | None) -> AgentVariant:
    if isinstance(variant, AgentVariant):
        return variant
    name = variant or os.environ.get(ENV_VAR) or "default"
    if name not in VARIANTS:
        raise KeyError(f"unknown agent variant {name!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[name]
