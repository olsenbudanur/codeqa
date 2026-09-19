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
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, Metrics, RLDataset, RLDatasetBuilder, Trajectory

from codeqa.agent.env import RepoEnv
from codeqa.grader import gates
from codeqa.grader.grade import grade, metrics as grade_metrics
from codeqa.grader.judge import JudgeClient, default_client
from codeqa.shared import paths
from codeqa.shared.contracts import GradeResult, Task, Trace
from codeqa.shared.jsonl import read_all
from codeqa.shared.profiles import get_profile
from codeqa.trainer.group_rewards import (FORCED_ANSWER_FACTOR, fill_judge_errors, grounded_credit, group_metrics, nan_safe,
                                          no_answer_penalty)

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
                 judge_model: str | None = None, offline_judge: bool = False, grounded_credit: float = 0.05,
                 length_shaping: bool = True, reward_version: str | None = None):
        self.reward_version = reward_version        # None = CODEQA_REWARD; the held-out evaluator pins "v2"
        self.task = task
        self.grounded_credit = grounded_credit
        self.length_shaping = length_shaping
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

            cb = renv.make_cookbook_env(_capture)
            cb._codeqa_renv = renv            # the cookbook may return envs in COMPLETION order (RetryOnFailure); pair by object, never by index
            out.append(cb)
            self._envs.append(renv)
        return out

    def _trace(self, renv: RepoEnv, cb_env: Env, traj: Trajectory) -> Trace:
        renv = getattr(cb_env, "_codeqa_renv", renv)              # 2026-09-19: index pairing scrambled rewards across a group (LOG 23:05)
        history = getattr(renv, "captured_history", None)
        if history is None:                                       # reward_fn skipped (parse error / overflow policy)
            inner = getattr(cb_env, "message_env", None)
            history = list(getattr(inner, "history", []) or [])
        trace = renv.trace_from_history(history)
        trace.stats.prompt_tokens, trace.stats.completion_tokens = trajectory_token_counts(traj)
        assistants = [m for m in trace.messages if m.role == "assistant"]
        for m, t in zip(assistants, traj.transitions):          # per-turn context so efficiency can use the final context
            m.usage = {"prompt_tokens": int(t.ob.length), "completion_tokens": len(t.ac.tokens)}
        if traj.stop_reason and trace.stats.stop_reason == "answer" and not trace.answer:
            trace.stats.stop_reason = "parse_error" if "parse" in traj.stop_reason else trace.stats.stop_reason
        return trace

    async def _grade(self, trace: Trace) -> GradeResult:
        async with _JUDGE_SEMAPHORE:
            return await grade(self.task, trace, variant=self.variant, judge_client=judge_client_for(self.judge_model, self.offline_judge),
                               reward=self.reward_version)

    async def compute_group_rewards(self, trajectory_group: list[Trajectory], env_group: Sequence[Env]) -> list[tuple[float, Metrics]]:
        if len(env_group) != len(trajectory_group):
            raise RuntimeError(f"env/trajectory count mismatch: {len(env_group)} vs {len(trajectory_group)}")
        traces = [self._trace(getattr(cb, "_codeqa_renv", renv), cb, traj)
                  for renv, cb, traj in zip(list(self._envs) + [None] * len(env_group), env_group, trajectory_group)]
        mismatched = sum(abs(t.stats.turns - len(traj.transitions)) > 1 for t, traj in zip(traces, trajectory_group))   # +1 = a truncated turn the cookbook continued past; a scramble is off by many
        if mismatched:                                             # must stay 0: a trace that does not describe its trajectory = scrambled credit
            logger.warning("compute_group_rewards: %d/%d traces do not match their trajectory (turns vs transitions)", mismatched, len(traces))
        results = await asyncio.gather(*(self._grade(t) for t in traces))
        penalties = [no_answer_penalty(t.stats.stop_reason, r.gate_failed) for t, r in zip(traces, results)]
        credits = [0.0 if math.isnan(r.reward) else grounded_credit(r.reward, r.gate_failed, self.grounded_credit) for r in results]
        budget = self.task.effective_budget()
        lengths = [gates.length_factor(gates.extract_answer(t), budget, self.task.task_type) if self.length_shaping else 1.0 for t in traces]
        forced = [FORCED_ANSWER_FACTOR if t.stats.forced_answer else 1.0 for t in traces]     # v3: a forced final answer keeps 90 %
        # shaped = (grader reward + grounded credit) x length factor x forced factor + stall penalty; the grader's reward stays length-free
        rewards = [(nan_safe(r.reward) + c) * lf * ff + p for r, p, c, lf, ff in zip(results, penalties, credits, lengths, forced)]
        errored = [math.isnan(r.reward) for r in results]
        _, totals = fill_judge_errors(rewards, errored)
        gm = group_metrics(totals, errored, [tool_sequence(t) for t in traces])
        out: list[tuple[float, Metrics]] = []
        for total, r, t, p, c, lf in zip(totals, results, traces, penalties, credits, lengths):
            m = grade_metrics(r, t, self.task)          # m["reward"] stays the grader's reward; the shaped total is separate
            m["no_answer_penalty"] = float(p != 0.0)
            m["grounded_credit"] = float(c != 0.0)
            m["length_factor_applied"] = lf
            m["forced_factor_applied"] = float(t.stats.forced_answer)
            m["reward_shaped"] = total
            m["pairing_mismatch"] = float(mismatched) / max(len(traces), 1)   # 0.0 when every trace describes its own trajectory
            m.update(gm)
            out.append((total, m))
        return out

    def logging_tags(self) -> list[str]:
        return [self.task.source, self.task.task_type]


class CodeQADataset(RLDataset):
    """Batches of `batch_size` groups; `epochs` passes over the tasks (so `max_steps` can exceed one pass)."""

    def __init__(self, builders: list[EnvGroupBuilder], batch_size: int, epochs: int = 1, publish_step: bool = True):
        self.builders = builders
        self.batch_size = batch_size
        self.epochs = epochs
        self.publish_step = publish_step          # the held-out evaluator's dataset must not reset CODEQA_TRAIN_STEP
        self.batches_per_epoch = max(math.ceil(len(builders) / batch_size), 1) if builders else 0

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        i = index % max(self.batches_per_epoch, 1)
        batch = self.builders[i * self.batch_size:(i + 1) * self.batch_size]
        self._write_groups(index, batch)
        if self.publish_step:                                   # v3 reward: the efficiency weight ramps in by training step
            os.environ["CODEQA_TRAIN_STEP"] = str(index)
        return batch

    def _write_groups(self, index: int, batch: Sequence[EnvGroupBuilder]) -> None:
        """Sidecar `iteration_N/groups.json` (group_idx -> task) so the Workshop can show the question behind a rollout.
        Written only when CODEQA_GROUPS_DIR points at the run's log dir (scripts/arm.py sets it)."""
        import json
        root = os.environ.get("CODEQA_GROUPS_DIR")
        if not root:
            return
        try:
            p = Path(root) / f"iteration_{index:06d}" / "groups.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            rows = [{"group_idx": g, "task_id": b.task.task_id, "source": b.task.source, "task_type": b.task.task_type,
                     "repo_id": b.task.repo_id, "question": b.task.question} for g, b in enumerate(batch) if hasattr(b, "task")]
            p.write_text(json.dumps(rows))
        except Exception as e:  # noqa: BLE001
            logger.warning("groups.json not written for iteration %d: %s", index, e)

    def __len__(self) -> int:
        return self.batches_per_epoch * self.epochs


def load_tasks(path: Path, max_tasks: int | None = None, seed: int = 0, shuffle: bool = True) -> list[Task]:
    """Tasks whose repo is indexed for the active agent variant: symbols.json always, map.txt only when the variant's
    map is `full` (the lean default builds its tree map from the manifest + symbols). Others are skipped with a log line."""
    from codeqa.agent.variants import resolve as resolve_variant
    need_full_map = resolve_variant(None).map == "full"
    tasks = read_all(path, Task)
    ok, skipped = [], {}
    for t in tasks:
        idx = paths.index_dir(t.repo_id)
        if (idx / "symbols.json").is_file() and (not need_full_map or (idx / "map.txt").is_file()):
            ok.append(t)
        else:
            skipped[t.repo_id] = skipped.get(t.repo_id, 0) + 1
    if skipped:
        logger.warning("skipping %d tasks whose repo is not indexed for this variant: %s", sum(skipped.values()), skipped)
    if shuffle:
        random.Random(seed).shuffle(ok)
    return ok[:max_tasks] if max_tasks else ok


def _interleave(groups: dict[str, list]) -> list:
    """Proportional (Bresenham-style) interleave: at each step the group furthest behind its share goes next."""
    total = {k: len(v) for k, v in groups.items() if v}
    taken = {k: 0 for k in total}
    out: list = []
    for _ in range(sum(total.values())):
        k = min((k for k in total if taken[k] < total[k]), key=lambda k: (taken[k] + 1) / total[k])
        out.append(groups[k][taken[k]])
        taken[k] += 1
    return out


def stratified_order(tasks: list[Task]) -> list[Task]:
    """Two-level stratification so every batch has the global TASK-TYPE mix and, within a type, spreads REPOS
    (repo is the largest difficulty factor: 8 % pass on xgboost vs 93 % on fastai in p4_bash). Each type's own
    order is a repo-proportional interleave; the types are then interleaved proportionally."""
    from collections import defaultdict
    by_type: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        by_type[t.task_type].append(t)
    per_type: dict[str, list[Task]] = {}
    for tt, ts in by_type.items():
        by_repo: dict[str, list[Task]] = defaultdict(list)
        for t in ts:
            by_repo[t.repo_id].append(t)
        per_type[tt] = _interleave(dict(sorted(by_repo.items())))
    return _interleave(dict(sorted(per_type.items())))


def builders_for(tasks: list[Task], profile_name: str, group_size: int, variant: str, judge_model: str | None,
                 offline_judge: bool = False, grounded_credit: float = 0.05, length_shaping: bool = True,
                 reward_version: str | None = None) -> list[EnvGroupBuilder]:
    return [CodeQAGroupBuilder(t, profile_name, group_size, variant, judge_model, offline_judge, grounded_credit, length_shaping, reward_version)
            for t in tasks]


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
    grounded_credit: float = 0.05        # 0 disables the shaping floor for gate-passing wrong answers
    length_shaping: bool = True          # multiply the trained reward by min(1, cap / answer tokens); evals never do
    stratify: bool = True                # every batch gets the same task-type mix (2026-09-20: 8-task batches swung reward by ±0.15 on the draw alone)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tasks = load_tasks(Path(self.tasks_path), self.max_tasks, self.seed)
        if not tasks:
            raise RuntimeError(f"no usable tasks in {self.tasks_path}")
        if self.stratify:
            tasks = stratified_order(tasks)
        builders = builders_for(tasks, self.profile_name, self.group_size, self.variant, self.judge_model, self.offline_judge, self.grounded_credit, self.length_shaping)
        ds = CodeQADataset(builders, self.groups_per_batch, self.epochs)
        logger.info("dataset: %d tasks -> %d batches of up to %d groups x %d (%d epochs)", len(tasks), len(ds),
                    self.groups_per_batch, self.group_size, self.epochs)
        return ds, None
