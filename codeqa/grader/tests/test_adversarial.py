"""Threat-model fixtures (docs/gap_specs.md §3): each adversarial trace must score as intended.

Judged tasks use KeywordJudge, the offline stand-in, so this file needs no API key.
"""
from __future__ import annotations

import math

import pytest

from codeqa.grader.grade import grade
from codeqa.grader.judge import KeywordJudge

# name -> (expected gate, expected reward or None when reward is checked separately)
EXPECTED = {
    "good": (None, 1.0),
    "good_locate": (None, 1.0),
    "good_value": (None, 1.0),
    "good_enumerate": (None, 1.0),
    "padded": (None, 1.0),                   # over the answer cap: length is not the grader's business (trainer shaping only)
    "wrong": (None, 0.0),                    # verifiable type, judge never consulted, literal mismatch
    "restated": (None, 0.0),                 # passes gates, satisfies no rubric item
    "fabricated": ("citations", 0.0),        # path does not exist
    "unread_citation": ("grounding", 0.0),   # cited file was never read
    "no_tools": ("grounding", 0.0),          # nothing read, so nothing can be grounded
    "judge_injection": (None, 0.0),          # injected instructions do not move the judge
    "thinking_answer": (None, 0.0),          # the literal lives only in thinking
    "redundant_reads": (None, 1.0),          # grounding fine; efficiency penalized under multiplicative (below)
    "verbatim": ("format", 0.0),             # copied tool output: verbatim-paste gate
    "tool_errors": ("budget", 0.0),          # 6 errors + 1 read = 7 calls > locate budget of 6
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
async def test_adversarial_trace_scores_as_intended(name, tasks, trace, repo):
    tr = trace(name)
    gate, reward = EXPECTED[name]
    r = await grade(tasks[tr.task_id], tr, judge_client=KeywordJudge(), repo=repo)
    assert r.gate_failed == gate, r.notes
    assert r.reward == pytest.approx(reward), r.notes


async def test_wrong_answer_never_reaches_the_judge(tasks, trace, repo):
    client = KeywordJudge()
    tr = trace("wrong")
    await grade(tasks[tr.task_id], tr, judge_client=client, repo=repo)
    assert client.calls == 0


async def test_redundant_reads_cost_efficiency(tasks, trace, repo):
    dup, clean = trace("redundant_reads"), trace("good_locate")
    rd = await grade(tasks["mini-locate"], dup, variant="multiplicative", repo=repo)
    rc = await grade(tasks["mini-locate"], clean, variant="multiplicative", repo=repo)
    assert rd.components.efficiency < rc.components.efficiency == 1.0
    assert rd.reward == rd.components.efficiency < 1.0


async def test_judge_sees_stripped_truncated_untrusted_answer(tasks, trace, repo):
    """The injection fixture's markdown and closing marker never reach the judge verbatim."""
    seen: list[str] = []

    class Spy(KeywordJudge):
        async def chat(self, messages, tools=None, max_tokens=None, temperature=1.0):
            seen.append(messages[-1].content)
            return await super().chat(messages, tools, max_tokens, temperature)

    tr = trace("judge_injection")
    await grade(tasks[tr.task_id], tr, judge_client=Spy(), repo=repo)
    body = seen[0].split("<<<ANSWER", 1)[1]
    assert "**IMPORTANT" not in body and "## Grading" not in body
    assert body.count("ANSWER>>>") == 1                       # only our closing marker survives
    assert "UNTRUSTED" in seen[0] or "untrusted" in seen[0]


async def test_judge_outage_gives_nan_not_zero(tasks, trace, repo):
    tr = trace("good")
    r = await grade(tasks[tr.task_id], tr, judge_client=KeywordJudge(fail=True), repo=repo)
    assert math.isnan(r.reward) and r.gate_failed == "judge_error"
    assert r.components.citations_grounded == 1.0            # gates had passed


async def test_format_failure_is_zero_never_nan(tasks, trace, repo):
    tr = trace("verbatim")
    r = await grade(tasks[tr.task_id], tr, judge_client=KeywordJudge(fail=True), repo=repo)
    assert r.reward == 0.0 and r.gate_failed == "format"
