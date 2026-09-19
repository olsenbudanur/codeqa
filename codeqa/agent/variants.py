"""Agent variants for ablations: which tools, what repo map is in the prompt, which rules text.

`default` is the LEAN agent: four index/text tools (no `overview`, so no Haiku summaries are needed) and a ~1k-token
structural map built from the manifest + symbols alone. That is what every caller gets unless it passes `variant=` to
RepoEnv or sets CODEQA_AGENT_VARIANT. `full` is the day-one five-tool design with the summarised 3k map, kept for
ablation (needs `summaries.json` + `map.txt`). Rewards do not change: grounding reads `files_read`, which every tool
fills for the lines it showed. Map modes: none | tree (structural, MAP_TREE_TOKENS) | full (map.txt).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from typing import Literal

from codeqa.agent.prompts import SYSTEM_RULES, SYSTEM_RULES_BASH, SYSTEM_RULES_NOINDEX, rules_for

ENV_VAR = "CODEQA_AGENT_VARIANT"
MAP_TREE_TOKENS = int(os.environ.get("CODEQA_MAP_TOKENS", "1000"))
MapMode = Literal["none", "tree", "full"]
FIVE = ("overview", "find_symbol", "grep", "read_file", "list_dir")
FOUR = ("find_symbol", "grep", "read_file", "list_dir")


@dataclass(frozen=True)
class AgentVariant:
    name: str
    tools: tuple[str, ...]
    map: MapMode
    rules: str

    @property
    def include_map(self) -> bool:      # kept for callers written against the first version
        return self.map != "none"

    @property
    def needs_summaries(self) -> bool:  # only these need the Haiku indexing step
        return "overview" in self.tools or self.map == "full"


VARIANTS: dict[str, AgentVariant] = {
    "default": AgentVariant("default", FOUR, "tree", rules_for(overview=False, has_map=True)),       # lean: no Haiku anywhere
    "lean": AgentVariant("lean", FOUR, "tree", rules_for(overview=False, has_map=True)),             # alias of default
    "full": AgentVariant("full", FIVE, "full", SYSTEM_RULES),                                        # day-one design, for ablation
    "tree_overview": AgentVariant("tree_overview", FIVE, "tree", rules_for(overview=True, has_map=True)),
    "noindex": AgentVariant("noindex", ("grep", "read_file", "list_dir"), "none", SYSTEM_RULES_NOINDEX),   # no map, no index tools
    "nomap": AgentVariant("nomap", FOUR, "none", rules_for(overview=False, has_map=False)),          # tools yes, map no
    "nomap_full": AgentVariant("nomap_full", FIVE, "none", SYSTEM_RULES),                           # old `nomap`: five tools, no map
    "bash": AgentVariant("bash", ("bash",), "tree", SYSTEM_RULES_BASH),                              # one shell tool, tree map kept
    "bash_nomap": AgentVariant("bash_nomap", ("bash",), "none", SYSTEM_RULES_BASH),
}


def repo_map_text(repo_id: str, variant: AgentVariant) -> str:
    """The map string for the user prompt under this variant ('' for none)."""
    if variant.map == "none":
        return ""
    if variant.map == "full":
        from codeqa.agent.indexing.repomap import load_map
        text = load_map(repo_id)
        if not text:
            raise FileNotFoundError(f"no map.txt for {repo_id}; run `python -m codeqa.agent.indexing.cli map {repo_id}` "
                                    f"or use a variant with map=tree")
        return text
    from codeqa.agent.indexing.repomap import tree_map
    return tree_map(repo_id, max_tokens=MAP_TREE_TOKENS)


def resolve(variant: str | AgentVariant | None) -> AgentVariant:
    if isinstance(variant, AgentVariant):
        return variant
    name = variant or os.environ.get(ENV_VAR) or "default"
    if name not in VARIANTS:
        raise KeyError(f"unknown agent variant {name!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[name]
