import math

from codeqa.grader.grade import grade, metrics
from codeqa.grader.judge import KeywordJudge

REQUIRED_METRICS = {"correct", "stalled", "reward", "format_ok", "citations_parse", "citations_exist", "citations_grounded", "identifier_grounded",
                    "correctness", "efficiency", "judge_error", "tool_calls", "prompt_tokens", "answer_tokens", "redundant_reads"}


async def test_metrics_keys_and_gate_one_hot(tasks, trace, repo):
    tr = trace("fabricated")
    r = await grade(tasks[tr.task_id], tr, repo=repo)
    m = metrics(r, tr, tasks[tr.task_id])
    assert REQUIRED_METRICS <= set(m)
    assert m["gate_citations"] == 1.0 and m["gate_format"] == 0.0 and m["stop_answer"] == 1.0


async def test_metrics_nan_reward_maps_to_zero_with_flag(tasks, trace, repo):
    tr = trace("good")
    r = await grade(tasks[tr.task_id], tr, judge_client=KeywordJudge(fail=True), repo=repo)
    m = metrics(r, tr, tasks[tr.task_id])
    assert math.isnan(r.reward) and m["reward"] == 0.0 and m["judge_error"] == 1.0


async def test_identifier_grounding_component(tasks, trace, repo):
    r = await grade(tasks["mini-locate"], trace("good_locate"), repo=repo)
    assert r.components.identifier_grounded == 1.0
    r = await grade(tasks["mini-enumerate"], trace("good_enumerate"), repo=repo)
    assert r.components.identifier_grounded == 1.0 and r.notes.startswith("symbols 2 gold")


async def test_explain_reference_mode_with_stand_in(tasks, trace, repo):
    """mini-explain has a rubric; drop it to exercise reference mode."""
    task = tasks["mini-explain"].model_copy(deep=True)
    task.grading.rubric = []
    tr = trace("verbatim").model_copy(deep=True)
    tr.answer = ("sign_token computes an HMAC-SHA256 of the payload with the secret and appends the hex digest after a dot "
                 "[src/miniapp/auth/tokens.py:L12-L15]. verify_token splits on the last dot and compares with compare_digest "
                 "[src/miniapp/auth/tokens.py:L18-L26].")
    tr.messages[-1].content = tr.answer
    r = await grade(task, tr, judge_client=KeywordJudge(), repo=repo)
    assert r.gate_failed is None and 0.0 < r.reward <= 1.0
