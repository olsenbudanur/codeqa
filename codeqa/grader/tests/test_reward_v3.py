"""Reward v3 (2026-09-20): correctness first, grounded fraction and token efficiency as secondary weights."""
from __future__ import annotations

import pytest

from codeqa.grader.grade import grade, v3_efficiency_weight


@pytest.mark.asyncio
async def test_v3_formula_before_and_after_the_efficiency_step(tasks, trace, repo, monkeypatch):
    t = trace("good_locate")
    task = tasks[t.task_id]
    v2 = await grade(task, t, repo=repo, reward="v2")
    assert v2.reward > 0
    c, g = v2.components.correctness, v2.components.citations_grounded
    monkeypatch.setenv("CODEQA_EFF_WEIGHT", "0.15"); monkeypatch.setenv("CODEQA_EFF_WEIGHT_FROM_STEP", "8"); monkeypatch.setenv("CODEQA_TRAIN_STEP", "0")
    assert v3_efficiency_weight() == 0.0
    r0 = await grade(task, t, repo=repo, reward="v3")
    assert r0.reward == pytest.approx(c * (0.75 + 0.25 * g))
    monkeypatch.setenv("CODEQA_TRAIN_STEP", "8")
    assert v3_efficiency_weight() == 0.15
    r8 = await grade(task, t, repo=repo, reward="v3")
    e = r8.components.efficiency
    assert 0.5 <= e <= 1.0
    assert r8.reward == pytest.approx(c * (0.6 + 0.25 * g + 0.15 * e))


@pytest.mark.asyncio
async def test_v3_wrong_answer_earns_nothing_from_secondary_terms(tasks, trace, repo, monkeypatch):
    t = trace("wrong")
    task = tasks[t.task_id]
    monkeypatch.setenv("CODEQA_TRAIN_STEP", "99"); monkeypatch.setenv("CODEQA_EFF_WEIGHT_FROM_STEP", "0")
    r = await grade(task, t, repo=repo, reward="v3")
    assert r.gate_failed is None or r.reward == 0.0
    if r.gate_failed is None:
        assert r.components.correctness == 0.0 and r.reward == 0.0


@pytest.mark.asyncio
async def test_v3_keeps_the_honesty_gates(tasks, trace, repo):
    for name in ("fabricated", "unread_citation", "no_tools"):
        t = trace(name)
        r = await grade(tasks[t.task_id], t, repo=repo, reward="v3")
        assert r.reward == 0.0 and r.gate_failed is not None, name


def test_env_pinned_reward_wins_over_env_var(monkeypatch):
    from codeqa.grader.grade import reward_version
    monkeypatch.setenv("CODEQA_REWARD", "v3")
    assert reward_version() == "v3"
