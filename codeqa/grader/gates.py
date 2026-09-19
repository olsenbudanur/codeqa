"""Hard gates: answer extraction, format validity, answer length, budget (C7 gates)."""
from __future__ import annotations

import math
import os
import re
from functools import lru_cache

from codeqa.shared.contracts import CITATION_RE, Budget, Message, Trace

# The policy model's tokenizer, so the answer cap means what the prompt tells the model ("under N tokens").
# Set CODEQA_GRADER_TOKENIZER=proxy to force the tokenizer-free proxy (tests do this for determinism).
GRADER_TOKENIZER = os.environ.get("CODEQA_GRADER_TOKENIZER", "Qwen/Qwen3.5-4B")

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def proxy_tokens(text: str) -> int:
    """Tokenizer-free proxy: max(words, chars/4). Within ~5-10 % of the Qwen count on real answers."""
    text = text.strip()
    if not text:
        return 0
    return max(len(text.split()), math.ceil(len(text) / 4))


@lru_cache(maxsize=1)
def _tokenizer():
    name = os.environ.get("CODEQA_GRADER_TOKENIZER", GRADER_TOKENIZER)
    if not name or name == "proxy":
        return None
    try:
        from codeqa.clients.tinker import tokenizer
        return tokenizer(name)
    except Exception:  # noqa: BLE001 - no cache, no network, no tinker: fall back to the proxy
        return None


def approx_tokens(text: str) -> int:
    """Token count with the policy tokenizer when loadable, else `proxy_tokens`."""
    text = text.strip()
    if not text:
        return 0
    tok = _tokenizer()
    if tok is None:
        return proxy_tokens(text)
    try:
        return len(tok.encode(text, add_special_tokens=False))
    except Exception:  # noqa: BLE001
        return proxy_tokens(text)


def strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks (closed or dangling) so hidden reasoning is never graded."""
    text = THINK_RE.sub("", text)
    text = OPEN_THINK_RE.sub("", text)
    return text.strip()


def final_assistant(trace: Trace) -> Message | None:
    for m in reversed(trace.messages):
        if m.role == "assistant":
            return m
    return None


def extract_answer(trace: Trace) -> str:
    """The graded text: the final assistant content only. Thinking and tool-call markup are dropped.

    Returns "" when the episode ended without an answer (last assistant turn still calling tools,
    a parse error, or an overflow).
    """
    if trace.answer.strip():
        return strip_thinking(trace.answer)
    last = final_assistant(trace)
    if last is None or last.tool_calls or last.parse_error:
        return ""
    text = strip_thinking(last.content)
    if TOOL_CALL_RE.search(text):
        return ""
    return text


VERBATIM_MAX_SHARE = 0.5     # answers whose lines are mostly pasted tool output are not answers
LENGTH_FLOOR = 0.1


def verbatim_share(answer: str, trace: Trace) -> float:
    """Share of the answer's substantive lines (>= 20 chars, citations stripped) that appear verbatim in a tool output
    of this episode. Copying tool output to hit rubric words is the threat the old hard length cap defended against."""
    tool_lines: set[str] = set()
    for m in trace.messages:
        if m.role == "tool":
            for ln in m.content.splitlines():
                ln = re.sub(r"^\s*L?\d+\s*\|\s?", "", ln).strip()      # drop the "  41 | " / "L41 | " prefixes read_file adds
                if len(ln) >= 20:
                    tool_lines.add(ln)
    lines = [re.sub(r"^\s*L?\d+\s*\|\s?", "", CITATION_RE.sub("", ln)).strip() for ln in answer.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 20]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if ln in tool_lines) / len(lines)


def length_factor(answer: str, budget: Budget) -> float:
    """Soft length term: 1.0 up to the cap, then cap / tokens (an answer twice the cap keeps half its reward), floored."""
    n = approx_tokens(answer)
    if n <= budget.max_answer_tokens:
        return 1.0
    return max(LENGTH_FLOOR, budget.max_answer_tokens / n)


def format_gate(answer: str, trace: Trace, budget: Budget) -> tuple[bool, str]:
    """Format is valid when there is a final answer, it stopped by answering, and it is not pasted tool output.
    Length is no longer a gate (decisions 2026-09-19): it scales the reward through `length_factor`."""
    if trace.stats.stop_reason in ("parse_error", "overflow", "error"):
        return False, f"stop_reason={trace.stats.stop_reason}"
    if not answer:
        return False, "no final answer"
    v = verbatim_share(answer, trace)
    if v > VERBATIM_MAX_SHARE:
        return False, f"{v:.0%} of the answer is pasted tool output"
    return True, ""


def citations_parse_gate(answer: str) -> tuple[bool, str]:
    if not CITATION_RE.search(answer):
        return False, "no [path:Lstart-Lend] citation in answer"
    return True, ""


def budget_gate(trace: Trace, budget: Budget) -> tuple[bool, str]:
    """Every tool call counts, errors included. The env enforces this in training; this catches other drivers."""
    if trace.stats.tool_calls > budget.max_tool_calls:
        return False, f"{trace.stats.tool_calls} tool calls > budget {budget.max_tool_calls}"
    return True, ""
