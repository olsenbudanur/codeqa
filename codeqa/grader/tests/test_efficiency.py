import pytest

from codeqa.grader.efficiency import TOKENS_PER_CALL, efficiency, redundant_reads, shape
from codeqa.shared.contracts import Budget, Span, TraceStats

B = Budget(max_tool_calls=10, max_answer_tokens=400)


def test_shape_free_zone_then_linear_to_floor():
    assert shape(0.0) == 1.0 and shape(0.5) == 1.0
    assert abs(shape(0.75) - 0.75) < 1e-9
    assert shape(1.0) == 0.5 and shape(3.0) == 0.5


def test_variants_on_the_same_trace():
    s = TraceStats(tool_calls=8, prompt_tokens=8 * TOKENS_PER_CALL // 2)        # calls 80% of budget, tokens 40%
    assert efficiency(s, B, "none") == (1.0, False)
    assert efficiency(s, B, "multiplicative") == (pytest.approx(0.7), False)    # max(0.8, 0.4) -> 1 - 0.5*0.6
    assert efficiency(s, B, "token_cost") == (1.0, False)                       # 0.4 is inside the free zone
    assert efficiency(s, B, "hard_cap") == (1.0, False)


def test_hard_cap_fails_gate_over_budget():
    assert efficiency(TraceStats(tool_calls=11, prompt_tokens=100), B, "hard_cap") == (1.0, True)
    assert efficiency(TraceStats(tool_calls=2, prompt_tokens=10 * TOKENS_PER_CALL + 1), B, "hard_cap") == (1.0, True)


def test_redundant_reads_count_and_penalize():
    reads = [Span(path="a", start=30, end=48), Span(path="a", start=35, end=42), Span(path="a", start=35, end=42), Span(path="a", start=1, end=48)]
    assert redundant_reads(reads) == 2
    clean = TraceStats(tool_calls=4, prompt_tokens=100, files_read=reads[:1])
    dup = TraceStats(tool_calls=4, prompt_tokens=100, files_read=reads)
    assert efficiency(dup, Budget(max_tool_calls=6), "multiplicative")[0] < efficiency(clean, Budget(max_tool_calls=6), "multiplicative")[0]


def test_unknown_variant():
    with pytest.raises(ValueError):
        efficiency(TraceStats(), B, "bogus")
