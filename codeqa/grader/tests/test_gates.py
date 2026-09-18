from codeqa.grader import gates
from codeqa.shared.contracts import Budget, Message, ToolCall, Trace, TraceStats


def _trace(last: Message, stop="answer", calls=0) -> Trace:
    return Trace(task_id="t", profile="p", messages=[Message(role="user", content="q"), last], stats=TraceStats(tool_calls=calls, stop_reason=stop))


def test_proxy_tokens_zero_and_scaling():
    assert gates.proxy_tokens("") == 0
    assert gates.proxy_tokens("a b c") == 3
    assert gates.proxy_tokens("x" * 400) == 100
    assert gates.approx_tokens("a b c") == 3          # conftest forces the proxy


def test_real_tokenizer_is_close_to_proxy(monkeypatch):
    monkeypatch.setenv("CODEQA_GRADER_TOKENIZER", "Qwen/Qwen3.5-4B")
    gates._tokenizer.cache_clear()
    try:
        tok = gates._tokenizer()
        if tok is None:
            import pytest
            pytest.skip("Qwen tokenizer not loadable here")
        text = "**Location:** `src/flask/json/provider.py:L144-L148` sets `ensure_ascii = True` by default.\n" * 20
        real, proxy = gates.approx_tokens(text), gates.proxy_tokens(text)
        assert real > 0 and real != proxy               # the real tokenizer is in use (path-heavy text tokenizes ~30 % above the proxy)
    finally:
        gates._tokenizer.cache_clear()


def test_strip_thinking_closed_and_dangling():
    assert gates.strip_thinking("<think>hidden</think>\nvisible") == "visible"
    assert gates.strip_thinking("visible <think>never closed") == "visible"


def test_extract_answer_prefers_trace_answer_then_final_content():
    t = _trace(Message(role="assistant", content="final [a.py:L1]"))
    assert gates.extract_answer(t) == "final [a.py:L1]"
    t.answer = "explicit"
    assert gates.extract_answer(t) == "explicit"


def test_extract_answer_empty_when_still_calling_tools_or_parse_error():
    t = _trace(Message(role="assistant", content="", tool_calls=[ToolCall(name="grep", args={})]))
    assert gates.extract_answer(t) == ""
    t = _trace(Message(role="assistant", content="garbage", parse_error="bad tool call"))
    assert gates.extract_answer(t) == ""
    t = _trace(Message(role="assistant", content="<tool_call><function=read_file></function></tool_call>"))
    assert gates.extract_answer(t) == ""


def test_format_gate_length_cap_and_stop_reason():
    b = Budget(max_answer_tokens=10)
    t = _trace(Message(role="assistant", content="short"))
    assert gates.format_gate("short", t, b) == (True, "")
    ok, why = gates.format_gate("word " * 20, t, b)
    assert not ok and "cap" in why
    t.stats.stop_reason = "overflow"
    assert not gates.format_gate("short", t, b)[0]


def test_budget_gate_counts_errors_as_calls():
    b = Budget(max_tool_calls=6)
    t = _trace(Message(role="assistant", content="x"), calls=7)
    t.stats.tool_errors = 6
    ok, why = gates.budget_gate(t, b)
    assert not ok and "7 tool calls" in why
