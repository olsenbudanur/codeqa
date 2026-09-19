"""v3 harness mechanics shared by the training env (tool_env.py) and the product driver (driver.py):
rounds (one message = one round, several commands), a context cap instead of a call cap, the remaining-budget
trailer after every round, and the forced final answer when the budget runs out. Pure functions; no I/O."""
from __future__ import annotations

FORCED_PROMPT = ("[Budget exhausted: no more tool calls. Give your final answer now from what you have already seen, "
                 "with citations only to lines that appeared in the output above. An answer with a tool call is discarded.]")
FORCED_MARK = "[Budget exhausted:"
COMMAND_CAP_NOTE = "\n[{dropped} more command(s) were not run: at most {cap} commands per message. Send them in your next message.]"
OUTPUT_CAP_NOTE = "\n(output for this message cut at {cap} chars; narrow the commands)"


def approx_tokens(text: str) -> int:
    """Cheap estimate for the trailer; the real count comes from the renderer or the sampler's usage."""
    return len(text) // 4


def trailer(context_tokens: int, context_cap: int, messages_left: int) -> str:
    return f"\n[context {context_tokens / 1000:.0f}k of {context_cap / 1000:.0f}k; {messages_left} message(s) left]"


def should_force(context_tokens: int, context_cap: int, turns_used: int, max_turns: int) -> bool:
    """Force the final answer when the next round would blow the context cap, or when exactly one message is left."""
    return context_tokens >= context_cap or turns_used >= max_turns - 1


def cap_calls(calls: list, cap: int | None) -> tuple[list, int]:
    """(calls to run, dropped count)."""
    if cap is None or len(calls) <= cap:
        return list(calls), 0
    return list(calls[:cap]), len(calls) - cap


def cap_outputs(texts: list[str], cap_chars: int) -> list[str]:
    """Total output of one round capped at cap_chars: later results are cut first, each keeps at least a first line."""
    total = sum(len(t) for t in texts)
    if total <= cap_chars:
        return list(texts)
    out: list[str] = []
    used = 0
    for i, t in enumerate(texts):
        room = max(cap_chars - used, 0)
        if len(t) <= room:
            out.append(t); used += len(t)
        else:
            head = t[:room] if room > 0 else t.splitlines()[0][:200] if t else ""
            out.append(head + OUTPUT_CAP_NOTE.format(cap=cap_chars)); used = cap_chars
    return out
