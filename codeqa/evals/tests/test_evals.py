import json

from codeqa.evals.plots import series
from codeqa.evals.report import table
from codeqa.evals.run import by_group, summarize
from codeqa.evals.sweqa_judge import parse_scores


def _row(tid, src, tt, reward, gate=None, stop="answer", calls=3, grounded=1.0):
    return {"task_id": tid, "source": src, "task_type": tt, "reward": reward, "gate_failed": gate, "stop_reason": stop,
            "metrics": {"reward": reward, "correctness": reward, "format_ok": 1.0, "citations_parse": 1.0, "citations_exist": 1.0,
                        "citations_grounded": grounded, "identifier_grounded": 1.0, "efficiency": 1.0, "judge_error": 0.0,
                        "tool_calls": calls, "tool_errors": 0.0, "prompt_tokens": 1000.0, "completion_tokens": 100.0,
                        "answer_tokens": 80.0, "turns": 2.0, "seconds": 5.0}}


def test_summarize_headline_ratios():
    rows = [_row("a", "sweqa", "explain", 1.0, calls=4), _row("b", "sweqa", "explain", 0.0, gate="format", stop="budget", calls=6, grounded=0.0),
            _row("c", "codescout", "locate", 0.5, calls=2)]
    s = summarize(rows)
    assert s["n"] == 3 and s["correct_rate"] == 2 / 3
    assert s["tool_calls_per_correct"] == (4 + 6 + 2) / 2
    assert s["stop_budget"] == 1 / 3 and s["gate_format"] == 1 / 3 and s["citation_valid"] == 2 / 3
    g = by_group(rows)
    assert set(g) >= {"sweqa/explain", "codescout/locate", "source:sweqa", "type:locate"}
    assert g["codescout/locate"]["n"] == 1


def test_summarize_empty_and_no_correct():
    assert summarize([])["n"] == 0
    s = summarize([_row("a", "sweqa", "explain", 0.0)])
    assert s["tool_calls_per_correct"] == float("inf")


def test_table_plain_and_markdown():
    rows = [{"profile": "p1", "reward": 0.5, "n": 3.0}, {"profile": "p2", "reward": float("inf"), "n": 120.0}]
    t = table(rows, ("reward", "n"), "profile")
    assert "p1" in t and "0.50" in t and "–" in t and "120" in t
    md = table(rows, ("reward", "n"), "profile", markdown=True)
    assert md.startswith("| profile | reward | n |") and "|---|---|---|" in md


def test_series_reads_steps_and_derives_tool_calls_per_correct():
    rows = [{"step": 0, "env/all/reward": 0.2, "env/all/tool_calls": 4.0}, {"step": 1, "env/all/reward": 0.5, "env/all/tool_calls": 3.0},
            {"step": 2, "optim/lr": 1e-4}]
    assert series(rows, "env/all", "reward") == ([0, 1], [0.2, 0.5])
    xs, ys = series(rows, "env/all", "tool_calls_per_correct")
    assert xs == [0, 1] and ys == [20.0, 6.0]
    assert series(rows, "eval/fast/env/all", "reward") == ([], [])


def test_parse_scores_validates_range():
    s = parse_scores('```json\n{"correctness": 12, "completeness": 10, "relevance": 18, "clarity": 15, "reasoning": 9}\n```')
    assert s["total"] == 64
    try:
        parse_scores(json.dumps({"correctness": 25, "completeness": 1, "relevance": 1, "clarity": 1, "reasoning": 1}))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_monitor_checks_flag_collapse_and_pass_healthy():
    from codeqa.evals.monitor import checks

    def row(step, reward, std, uniq, budget, fmt, calls):
        return {"step": step, "env/all/reward": reward, "env/all/group_reward_std": std, "env/all/unique_tool_sequences_per_group": uniq,
                "env/all/stop_budget": budget, "env/all/stop_max_turns": 0.0, "env/all/gate_format": fmt, "env/all/tool_calls": calls,
                "env/all/judge_error": 0.0}
    collapsed = [row(0, 0.15, 0.25, 3.5, 0.08, 0.0, 3.5), row(1, 0.0, 0.0, 4.0, 1.0, 1.0, 11.8), row(2, 0.0, 0.0, 4.0, 1.0, 1.0, 11.8)]
    w = checks(collapsed)
    assert any("group_reward_std" in x for x in w) and any("without answering" in x for x in w)
    healthy = [row(0, 0.18, 0.31, 3.7, 0.08, 0.25, 5.0), row(1, 0.25, 0.29, 3.8, 0.0, 0.56, 6.1), row(2, 0.28, 0.28, 4.0, 0.0, 0.5, 6.7)]
    assert checks(healthy) == []
