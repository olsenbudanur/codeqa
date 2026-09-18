"""Group-level reward post-processing (gap_specs §4, §10). Pure functions; no cookbook types needed.

The cookbook computes total reward = sum of per-step rewards + the value returned by
`EnvGroupBuilder.compute_group_rewards`. Our env reward_fn returns 0.0 with metric `judge_error = 1`
when the judge failed, so here we add back the mean of the healthy siblings for those samples: their
total becomes exactly the group mean and their advantage is 0. If every sample errored, the group
stays constant (all zeros) and `remove_constant_reward_groups=True` drops it.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


NO_ANSWER_PENALTY = -0.1
NO_ANSWER_STOPS = ("budget", "max_turns")


def no_answer_penalty(stop_reason: str, gate_failed: str | None) -> float:
    """decisions.md (evening, #2): an episode that ends without a final answer gets -0.1; a format failure stays 0,
    so a bad answer still beats stalling. Only the format gate can fire on a no-answer episode, so the penalty applies
    exactly when the stop reason is budget|max_turns and the answer was empty (gate_failed == "format")."""
    return NO_ANSWER_PENALTY if (stop_reason in NO_ANSWER_STOPS and gate_failed == "format") else 0.0


def fill_judge_errors(rewards: Sequence[float], errored: Sequence[bool]) -> tuple[list[float], list[float]]:
    """Returns (group_reward_additions, final_totals). Errored samples get +mean(healthy); others +0."""
    healthy = [r for r, e in zip(rewards, errored) if not e]
    fill = statistics.fmean(healthy) if healthy else 0.0
    adds = [fill - r if e else 0.0 for r, e in zip(rewards, errored)]
    totals = [r + a for r, a in zip(rewards, adds)]
    return adds, totals


def advantages(totals: Sequence[float]) -> list[float]:
    """Group-centered advantages, as GRPO-style centering does."""
    if not totals:
        return []
    mu = statistics.fmean(totals)
    return [t - mu for t in totals]


def group_reward_std(totals: Sequence[float]) -> float:
    return statistics.pstdev(totals) if len(totals) > 1 else 0.0


def unique_tool_sequences(sequences: Sequence[Sequence[str]]) -> int:
    """Distinct tool-call name sequences in the group. Trending to 1 while reward plateaus = collapsed rollouts."""
    return len({tuple(s) for s in sequences})


def group_metrics(totals: Sequence[float], errored: Sequence[bool], sequences: Sequence[Sequence[str]]) -> dict[str, float]:
    n = max(len(totals), 1)
    return {
        "group_reward_std": group_reward_std(totals),
        "group_reward_mean": statistics.fmean(totals) if totals else 0.0,
        "unique_tool_sequences_per_group": float(unique_tool_sequences(sequences)),
        "judge_error_rate": sum(errored) / n,
        "group_all_judge_errors": 1.0 if errored and all(errored) else 0.0,
    }


def nan_safe(x: float) -> float:
    return 0.0 if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)
