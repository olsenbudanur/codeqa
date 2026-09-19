"""Rewards must be computed from the trajectory's OWN history even when the cookbook returns envs in completion order
(RetryOnFailure does). 2026-09-19: index pairing scrambled 35-67 % of a group's rewards; run one (default strategy) was fine."""
import random
from types import SimpleNamespace

import pytest

from codeqa.shared import paths
from codeqa.trainer.dataset_builder import CodeQAGroupBuilder
from codeqa.shared.contracts import Task

FLASK = "pallets__flask__85c5d93"
pytestmark = pytest.mark.skipif(not (paths.repo_dir(FLASK) / "manifest.json").exists(), reason="flask snapshot not present")


def _traj(n_turns: int):
    tr = [SimpleNamespace(ob=SimpleNamespace(length=100), ac=SimpleNamespace(tokens=[1, 2, 3])) for _ in range(n_turns)]
    return SimpleNamespace(transitions=tr, stop_reason=None)


@pytest.mark.asyncio
async def test_trace_follows_the_env_that_arrived_with_the_trajectory():
    task = Task(task_id="t", repo_id=FLASK, question="Where is the session cookie signed?", task_type="locate", source="teacher")
    b = CodeQAGroupBuilder(task, "qwen4b-base", group_size=6)
    envs = list(await b.make_envs())
    # each env ran a different number of turns; history = system, user, then one assistant turn per "turn"
    for i, cb in enumerate(envs):
        renv = cb._codeqa_renv
        renv.captured_history = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}] + \
            [{"role": "assistant", "content": f"answer {i}"} for _ in range(i + 1)]
    order = list(range(len(envs))); random.Random(0).shuffle(order)           # completion order != creation order
    env_group = [envs[i] for i in order]
    trajs = [_traj(i + 1) for i in order]
    traces = [b._trace(getattr(cb, "_codeqa_renv", None), cb, tj) for cb, tj in zip(env_group, trajs)]
    for t, tj, i in zip(traces, trajs, order):
        assert t.stats.turns == len(tj.transitions) == i + 1
        assert t.answer == f"answer {i}"
