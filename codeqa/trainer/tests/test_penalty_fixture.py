"""The no-answer penalty end to end on fixture traces: stalled episode -> -0.1, format failure -> 0, good -> 1."""
from __future__ import annotations

import json
import os

import pytest

os.environ["CODEQA_GRADER_TOKENIZER"] = "proxy"

from codeqa.grader.grade import grade
from codeqa.grader.judge import KeywordJudge
from codeqa.grader.repo import FIXTURE_REPO_ID, load_repo
from codeqa.shared import paths
from codeqa.shared.contracts import Message, Task, ToolCall, Trace
from codeqa.shared.jsonl import read_all
from codeqa.grader import gates
from codeqa.trainer.group_rewards import advantages, no_answer_penalty


def _trace(name: str) -> Trace:
    return Trace.model_validate(json.loads((paths.FIXTURES / "traces" / f"{name}.json").read_text()))


async def _shaped(task: Task, trace: Trace) -> float:
    r = await grade(task, trace, judge_client=KeywordJudge(), repo=load_repo(FIXTURE_REPO_ID))
    return r.reward * gates.length_factor(gates.extract_answer(trace), task.effective_budget()) + no_answer_penalty(trace.stats.stop_reason, r.gate_failed)


async def test_stalled_vs_bad_format_vs_good():
    tasks = {t.task_id: t for t in read_all(paths.FIXTURES / "tasks.jsonl", Task)}
    stalled = _trace("good_locate").model_copy(deep=True)
    stalled.answer = ""
    stalled.messages[-1] = Message(role="assistant", content="", tool_calls=[ToolCall(name="grep", args={"pattern": "x"})])
    stalled.stats.stop_reason = "budget"
    assert await _shaped(tasks["mini-locate"], stalled) == pytest.approx(-0.1)
    padded = await _shaped(tasks["mini-trace"], _trace("padded"))               # answered, over the cap
    assert padded == 1.0                                                          # v2: no over-cap scaling (the judge's contradicted term polices padding)
    import os
    os.environ["CODEQA_LENGTH_CAP"] = "on"
    try:
        capped = await _shaped(tasks["mini-trace"], _trace("padded"))
        assert 0.0 < capped < 1.0                                                 # with the cap switched on it scales in training only
    finally:
        os.environ.pop("CODEQA_LENGTH_CAP", None)
    graded = await grade(tasks["mini-trace"], _trace("padded"), judge_client=KeywordJudge(), repo=load_repo(FIXTURE_REPO_ID))
    assert graded.reward == 1.0                                                   # ...and untouched in the grader
    assert await _shaped(tasks["mini-locate"], _trace("good_locate")) == 1.0
    adv = advantages([-0.1, 0.0, 1.0, 0.0])   # stall < bad answer < correct
    assert adv[0] < adv[1] < adv[2]                                            # stalling ranks below a bad answer
