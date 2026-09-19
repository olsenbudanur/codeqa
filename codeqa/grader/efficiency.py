"""Efficiency term and its variants (gap_specs §6). Run one uses `none` (eff = 1.0).

Budgets: tool calls from the task budget; context tokens from `prefix_tokens + max_tool_calls * TOKENS_PER_CALL`, measured
against the FINAL context length (prompt of the last turn + its completion), not the sum over turns (each turn's prompt
already contains all previous turns, so the sum over-counts 3-4x).
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
TOKENS_PER_CALL = 1500
import os as _os
FREE_FRACTION = float(_os.environ.get("CODEQA_EFF_FREE_FRACTION", "0.5"))   # share of the call/token budget that is free of efficiency pressure
FLOOR = 0.5


def redundant_reads(files_read: list[Span]) -> int:
    """Multi-line read spans that were already fully covered by earlier reads of the same file (one-line grep spans are
    tool hits, not reads, and never count)."""
    seen: dict[str, set[int]] = {}
    dup = 0
    for s in files_read:
        lines = set(range(s.start, s.end + 1))
        have = seen.setdefault(s.path, set())
        if len(lines) > 1 and lines <= have:
            dup += 1
        have |= lines
    return dup


def token_budget(budget: Budget, prefix_tokens: int = 0) -> int:
    return prefix_tokens + budget.max_tool_calls * TOKENS_PER_CALL


def context_tokens(trace) -> tuple[int, int]:
    """(prefix_tokens, final_context_tokens) from per-message usage when the driver/trainer recorded it, else (0, 0).
    prefix = prompt tokens of the first assistant turn (system + map + question + tool specs);
    final = prompt tokens of the last assistant turn + its completion."""
    turns = [m for m in trace.messages if m.role == "assistant" and m.usage.get("prompt_tokens")]
    if not turns:
        return 0, 0
    first, last = turns[0], turns[-1]
    return int(first.usage.get("prompt_tokens", 0)), int(last.usage.get("prompt_tokens", 0)) + int(last.usage.get("completion_tokens", 0))


def shape(usage: float) -> float:
    """1.0 up to FREE_FRACTION of budget, then linear down to FLOOR at 100%."""
    if usage <= FREE_FRACTION:
        return 1.0
    over = min((usage - FREE_FRACTION) / (1.0 - FREE_FRACTION), 1.0)
    return 1.0 - (1.0 - FLOOR) * over


def usage_ratios(stats: TraceStats, budget: Budget, prefix_tokens: int = 0, final_tokens: int | None = None) -> tuple[float, float]:
    """(calls_ratio, tokens_ratio). Tokens use the final context when given; otherwise fall back to stats.prompt_tokens
    (cumulative, over-counts) against the same budget so the variant still runs on old traces."""
    calls = stats.tool_calls + redundant_reads(stats.files_read)
    tokens = stats.prompt_tokens if final_tokens is None else final_tokens
    return calls / max(budget.max_tool_calls, 1), tokens / max(token_budget(budget, prefix_tokens), 1)


def efficiency(stats: TraceStats, budget: Budget, variant: str = "none", prefix_tokens: int = 0,
               final_tokens: int | None = None) -> tuple[float, bool]:
    """(eff in [0.5, 1], budget_gate_failed). Only `hard_cap` can fail the gate."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown efficiency variant {variant!r}; choose from {VARIANTS}")
    if variant == "none":
        return 1.0, False
    calls_ratio, tokens_ratio = usage_ratios(stats, budget, prefix_tokens, final_tokens)
    if variant == "hard_cap":
        return 1.0, (calls_ratio > 1.0 or tokens_ratio > 1.0)
    if variant == "token_cost":
        return shape(tokens_ratio), False
    return shape(max(calls_ratio, tokens_ratio)), False


# ---------------------------------------------------------------------------
# v3 (2026-09-20): efficiency on the true cost, the sum of prompt + completion tokens over every turn (each turn
# re-prefills the whole context), excluding the final answer's own tokens so answer length carries no pressure.
# ---------------------------------------------------------------------------
TOKEN_SUM_BUDGET = int(_os.environ.get("CODEQA_EFF_TOKEN_BUDGET", "150000"))


def token_sum(trace, answer_tokens: int = 0) -> int:
    turns = [m for m in trace.messages if m.role == "assistant" and m.usage.get("prompt_tokens")]
    if turns:
        total = sum(int(m.usage.get("prompt_tokens", 0)) + int(m.usage.get("completion_tokens", 0)) for m in turns)
    else:
        total = int(trace.stats.prompt_tokens) + int(trace.stats.completion_tokens)
    return max(total - answer_tokens, 0)


def token_sum_efficiency(trace, answer_tokens: int = 0, budget_tokens: int | None = None) -> float:
    """shape(total tokens / budget): 1.0 up to FREE_FRACTION of the budget, then linear down to FLOOR at 100 %."""
    return shape(token_sum(trace, answer_tokens) / max(budget_tokens or TOKEN_SUM_BUDGET, 1))
