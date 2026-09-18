"""Efficiency term and its variants (gap_specs §6). Run one uses `none` (eff = 1.0).

Budgets: tool calls from the task budget; prompt tokens from `max_tool_calls * TOKENS_PER_CALL`.
Usage is measured as a fraction of budget. The first half of the budget is free; usage beyond that
lowers eff linearly from 1.0 at 50% to 0.5 at 100% (and stays 0.5 beyond, which only other drivers can reach).

  none            eff = 1
  multiplicative  usage = max(calls_ratio, tokens_ratio); calls_ratio counts redundant reads twice
  hard_cap        eff = 1, but usage > 1 on either axis fails the `budget` gate (reward 0)
  token_cost      usage = tokens_ratio only
"""
from __future__ import annotations

from typing import Literal

from codeqa.shared.contracts import Budget, Span, TraceStats

Variant = Literal["none", "multiplicative", "hard_cap", "token_cost"]
VARIANTS: tuple[str, ...] = ("none", "multiplicative", "hard_cap", "token_cost")
TOKENS_PER_CALL = 3000
FREE_FRACTION = 0.5
FLOOR = 0.5


def redundant_reads(files_read: list[Span]) -> int:
    """Read spans that were already fully covered by earlier reads of the same file."""
    seen: dict[str, set[int]] = {}
    dup = 0
    for s in files_read:
        lines = set(range(s.start, s.end + 1))
        have = seen.setdefault(s.path, set())
        if lines and lines <= have:
            dup += 1
        have |= lines
    return dup


def token_budget(budget: Budget) -> int:
    return budget.max_tool_calls * TOKENS_PER_CALL


def shape(usage: float) -> float:
    """1.0 up to FREE_FRACTION of budget, then linear down to FLOOR at 100%."""
    if usage <= FREE_FRACTION:
        return 1.0
    over = min((usage - FREE_FRACTION) / (1.0 - FREE_FRACTION), 1.0)
    return 1.0 - (1.0 - FLOOR) * over


def usage_ratios(stats: TraceStats, budget: Budget) -> tuple[float, float]:
    calls = stats.tool_calls + redundant_reads(stats.files_read)
    return calls / max(budget.max_tool_calls, 1), stats.prompt_tokens / max(token_budget(budget), 1)


def efficiency(stats: TraceStats, budget: Budget, variant: str = "none") -> tuple[float, bool]:
    """(eff in [0.5, 1], budget_gate_failed). Only `hard_cap` can fail the gate."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown efficiency variant {variant!r}; choose from {VARIANTS}")
    if variant == "none":
        return 1.0, False
    calls_ratio, tokens_ratio = usage_ratios(stats, budget)
    if variant == "hard_cap":
        return 1.0, (stats.tool_calls > budget.max_tool_calls or stats.prompt_tokens > token_budget(budget))
    if variant == "token_cost":
        return shape(tokens_ratio), False
    return shape(max(calls_ratio, tokens_ratio)), False
