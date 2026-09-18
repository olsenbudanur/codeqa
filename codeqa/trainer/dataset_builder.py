"""Task -> EnvGroupBuilder -> RLDataset for the cookbook loop (C2).

One group = `group_size` fresh `RepoEnv`s for the same task. The env's reward_fn only captures the
cookbook history; all grading happens in `compute_group_rewards`, where the trajectories (token
counts) and the envs (files_read, tool calls) are both in hand, the 8 judge calls run concurrently,
and the judge-failure policy (NaN -> group mean, gap_specs §4) is applied in one place.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, Metrics, RLDataset, RLDatasetBuilder, Trajectory

from codeqa.agent.env import RepoEnv
from codeqa.grader.grade import grade, metrics as grade_metrics
from codeqa.grader.judge import JudgeClient, default_client
from codeqa.shared import paths
from codeqa.shared.contracts import GradeResult, Task, Trace
from codeqa.shared.jsonl import read_all
from codeqa.shared.profiles import get_profile
from codeqa.trainer.group_rewards import fill_judge_errors, group_metrics, nan_safe, no_answer_penalty

logger = logging.getLogger(__name__)

_JUDGE_CLIENTS: dict[str, JudgeClient] = {}
_JUDGE_SEMAPHORE = asyncio.Semaphore(16)      # concurrent Haiku calls across all groups


def judge_client_for(model: str | None, offline: bool = False) -> JudgeClient:
    """One shared client per judge model. `offline=True` uses the keyword stand-in (smoke tests only)."""
    key = "offline" if offline else (model or "default")
    if key not in _JUDGE_CLIENTS:
        if offline:
            from codeqa.grader.judge import KeywordJudge
            _JUDGE_CLIENTS[key] = KeywordJudge()
        else:
            _JUDGE_CLIENTS[key] = default_client(model) if model else default_client()
    return _JUDGE_CLIENTS[key]


def tool_sequence(trace: Trace) -> list[str]:
    return [tc.name for m in trace.messages if m.role == "assistant" for tc in m.tool_calls]


def trajectory_token_counts(traj: Trajectory) -> tuple[int, int]:
    prompt = sum(t.ob.length for t in traj.transitions)
    completion = sum(len(t.ac.tokens) for t in traj.transitions)
    return prompt, completion


class CodeQAGroupBuilder(EnvGroupBuilder):
    """Holds only strings and numbers so it stays pickleable; RepoEnvs are built in make_envs."""

    def __init__(self, task: Task, profile_name: str, group_size: int, variant: str = "none",
                 judge_model: str | None = None, offline_judge: bool = False):
        self.task = task
        self.profile_name = profile_name
        self.group_size = group_size
        self.variant = variant
        self.judge_model = judge_model
        self.offline_judge = offline_judge
        self._envs: list[RepoEnv] = []

    async def make_envs(self) -> Sequence[Env]:
        profile = get_profile(self.profile_name)
        self._envs = []
        out: list[Env] = []
        for _ in range(self.group_size):
            renv = RepoEnv(self.task, profile)
            renv.captured_history = None                          # set by _capture at episode end

            async def _capture(history, env=renv):
                env.captured_history = list(history)
                return 0.0, {}                                    # real reward comes from compute_group_rewards

            out.append(renv.make_cookbook_env(_capture))
            self._envs.append(renv)
        return out

    def _trace(self, renv: RepoEnv, cb_env: Env, traj: Trajectory) -> Trace:
        history = getattr(renv, "captured_history", None)
        if history is None:                                       # reward_fn skipped (parse error / overflow policy)
            inner = getattr(cb_env, "message_env", None)
            history = list(getattr(inner, "history", []) or [])
        trace = renv.trace_from_history(history)
        trace.stats.prompt_tokens, trace.stats.completion_tokens = trajectory_token_counts(traj)
        if traj.stop_reason and trace.stats.stop_reason == "answer" and not trace.answer:
            trace.stats.stop_reason = "parse_error" if "parse" in traj.stop_reason else trace.stats.stop_reason
        return trace

    async def _grade(self, trace: Trace) -> GradeResult:
        async with _JUDGE_SEMAPHORE:
            return await grade(self.task, trace, variant=self.variant, judge_client=judge_client_for(self.judge_model, self.offline_judge))

    async def compute_group_rewards(self, trajectory_group: list[Trajectory], env_group: Sequence[Env]) -> list[tuple[float, Metrics]]:
        traces = [self._trace(renv, cb, traj) for renv, cb, traj in zip(self._envs, env_group, trajectory_group)]
        results = await asyncio.gather(*(self._grade(t) for t in traces))
        penalties = [no_answer_penalty(t.stats.stop_reason, r.gate_failed) for t, r in zip(traces, results)]
        rewards = [nan_safe(r.reward) + p for r, p in zip(results, penalties)]
        errored = [math.isnan(r.reward) for r in results]
        _, totals = fill_judge_errors(rewards, errored)
        gm = group_metrics(totals, errored, [tool_sequence(t) for t in traces])
        out: list[tuple[float, Metrics]] = []
        for total, r, t, p in zip(totals, results, traces, penalties):
            m = grade_metrics(r, t, self.task)          # m["reward"] stays the grader's reward; the shaped total is separate
            m["no_answer_penalty"] = float(p != 0.0)
            m["reward_shaped"] = total
            m.update(gm)
            out.append((total, m))
        return out

    def logging_tags(self) -> list[str]:
        return [self.task.source, self.task.task_type]


class CodeQADataset(RLDataset):
    """Batches of `batch_size` groups; `epochs` passes over the tasks (so `max_steps` can exceed one pass)."""

    def __init__(self, builders: list[EnvGroupBuilder], batch_size: int, epochs: int = 1):
        self.builders = builders
        self.batch_size = batch_size
        self.epochs = epochs
        self.batches_per_epoch = max(math.ceil(len(builders) / batch_size), 1) if builders else 0

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        i = index % max(self.batches_per_epoch, 1)
        return self.builders[i * self.batch_size:(i + 1) * self.batch_size]

    def __len__(self) -> int:
        return self.batches_per_epoch * self.epochs


def load_tasks(path: Path, max_tasks: int | None = None, seed: int = 0, shuffle: bool = True) -> list[Task]:
    """Tasks whose repo has a map.txt (RepoEnv needs it). Others are skipped with a log line."""
    tasks = read_all(path, Task)
    ok, skipped = [], {}
    for t in tasks:
        if (paths.index_dir(t.repo_id) / "map.txt").is_file():
            ok.append(t)
        else:
            skipped[t.repo_id] = skipped.get(t.repo_id, 0) + 1
    if skipped:
        logger.warning("skipping %d tasks without map.txt: %s", sum(skipped.values()), skipped)
    if shuffle:
        random.Random(seed).shuffle(ok)
    return ok[:max_tasks] if max_tasks else ok


def builders_for(tasks: list[Task], profile_name: str, group_size: int, variant: str, judge_model: str | None,
                 offline_judge: bool = False) -> list[EnvGroupBuilder]:
    return [CodeQAGroupBuilder(t, profile_name, group_size, variant, judge_model, offline_judge) for t in tasks]


@chz.chz
class CodeQADatasetBuilder(RLDatasetBuilder):
    tasks_path: str
    profile_name: str
    group_size: int = 8
    groups_per_batch: int = 32
    variant: str = "none"
    judge_model: str | None = None
    offline_judge: bool = False
    max_tasks: int | None = None
    seed: int = 0
    epochs: int = 1

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tasks = load_tasks(Path(self.tasks_path), self.max_tasks, self.seed)
        if not tasks:
            raise RuntimeError(f"no usable tasks in {self.tasks_path}")
        builders = builders_for(tasks, self.profile_name, self.group_size, self.variant, self.judge_model, self.offline_judge)
        ds = CodeQADataset(builders, self.groups_per_batch, self.epochs)
        logger.info("dataset: %d tasks -> %d batches of up to %d groups x %d (%d epochs)", len(tasks), len(ds),
                    self.groups_per_batch, self.group_size, self.epochs)
        return ds, None
