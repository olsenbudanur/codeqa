"""A6: first end-to-end RL training through the cookbook loop on real tasks with the real grader.

10 graphiti tasks × group 4 × 3 steps → metrics.jsonl + rollout pages → sampler checkpoint → TinkerChatClient on the
checkpoint answers a flask question through run_episode → profiles.yaml + data/models/manifest.json records.
Lane C lifts TaskGroupBuilder / TaskDataset / the Config into codeqa/trainer (C2).

Run: uv run python -u -m scripts.smoke_train [run_name]   (~5-10 min; costs a few dollars of Tinker compute)
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook import checkpoint_utils, hyperparam_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder

from codeqa.agent.driver import run_episode, save_trace
from codeqa.agent.env import RepoEnv
from codeqa.clients.base import make_client
from codeqa.grader.grade import grade
from codeqa.grader.grade import metrics as grade_metrics
from codeqa.shared import paths
from codeqa.shared.contracts import CheckpointRecord, EndpointProfile, Task
from codeqa.shared.jsonl import read_all
from codeqa.shared.profiles import add_profile, get_profile
from codeqa.trainer.group_rewards import fill_judge_errors, group_metrics, nan_safe

MODEL = "Qwen/Qwen3.5-4B"


class TaskGroupBuilder(EnvGroupBuilder):
    """One task → group_size RepoEnvs; reward = real grader; judge errors filled to the group mean."""

    def __init__(self, task: Task, profile: EndpointProfile, group_size: int, variant: str = "none"):
        self.task, self.profile, self.group_size, self.variant = task, profile, group_size, variant
        self._envs: list[RepoEnv] = []

    async def make_envs(self):
        self._envs = [RepoEnv(self.task, self.profile) for _ in range(self.group_size)]
        return [env.make_cookbook_env(self._reward) for env in self._envs]

    async def _reward(self, history, env: RepoEnv):
        trace = env.trace_from_history(history)
        result = await grade(self.task, trace, self.variant)
        env.last_grade, env.last_trace = result, trace  # type: ignore[attr-defined]
        return nan_safe(result.reward), grade_metrics(result, trace, self.task)

    async def compute_group_rewards(self, trajectory_group, env_group):
        grades = [getattr(e, "last_grade", None) for e in self._envs]
        rewards = [nan_safe(g.reward) if g is not None else 0.0 for g in grades]
        errored = [bool(g is not None and g.gate_failed == "judge_error") for g in grades]
        adds, totals = fill_judge_errors(rewards, errored)
        seqs = [[tc.name for m in getattr(e, "last_trace").messages if m.role == "assistant" for tc in m.tool_calls]
                if hasattr(e, "last_trace") else [] for e in self._envs]
        gm = group_metrics(totals, errored, seqs)
        return [(a, dict(gm)) for a in adds]

    def logging_tags(self) -> list[str]:
        return [self.task.source, self.task.task_type]


class TaskDataset(RLDataset):
    def __init__(self, builders: list[EnvGroupBuilder], groups_per_batch: int, num_batches: int):
        self.builders, self.gpb, self.n = builders, groups_per_batch, num_batches

    def get_batch(self, index: int):
        start = (index * self.gpb) % len(self.builders)
        batch = self.builders[start:start + self.gpb]
        if len(batch) < self.gpb:  # wrap around
            batch = batch + self.builders[: self.gpb - len(batch)]
        return batch

    def __len__(self) -> int:
        return self.n


@chz.chz
class SmokeDatasetBuilder(RLDatasetBuilder):
    tasks_path: str
    profile_name: str = "qwen4b-base"
    group_size: int = 4
    groups_per_batch: int = 10
    num_batches: int = 3
    variant: str = "none"

    async def __call__(self):
        tasks = read_all(Path(self.tasks_path), Task)
        profile = get_profile(self.profile_name)
        builders = [TaskGroupBuilder(t, profile, self.group_size, self.variant) for t in tasks]
        return TaskDataset(builders, self.groups_per_batch, self.num_batches), None


def summarize_metrics(log_path: Path) -> list[dict[str, Any]]:
    rows = []
    mp = log_path / "metrics.jsonl"
    if not mp.exists():
        return rows
    keys = ["env/all/reward/total", "env/all/format_ok", "env/all/citations_grounded", "env/all/correctness", "env/all/tool_calls",
            "env/all/prompt_tokens", "env/all/turns", "env/all/group_reward_std", "env/all/unique_tool_sequences_per_group", "time/total"]
    for line in mp.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append({k: d.get(k) for k in ["step", *keys] if k in d or k == "step"})
    return rows


async def main(run_name: str) -> None:
    t0 = time.time()
    log_path = paths.LOGS / run_name
    tasks_path = paths.TASKS_TRAIN / "smoke_graphiti.jsonl"
    config = train.Config(
        learning_rate=hyperparam_utils.get_lr(MODEL),
        dataset_builder=SmokeDatasetBuilder(tasks_path=str(tasks_path), profile_name="qwen4b-base", group_size=4, groups_per_batch=10, num_batches=3),
        model_name=MODEL, recipe_name="codeqa_smoke", max_tokens=1024, log_path=str(log_path),
        eval_every=0, save_every=1, lora_rank=32, remove_constant_reward_groups=False, num_groups_to_log=2,
        rollout_json_export=True, max_steps=3, renderer_name="qwen3_5",
    )
    print(f"[train] {run_name}: lr={config.learning_rate:.2e} 10 tasks × group 4 × 3 steps → {log_path}", flush=True)
    await asyncio.wait_for(train.main(config), timeout=1800)
    print(f"[train] done in {time.time()-t0:.0f}s", flush=True)
    for r in summarize_metrics(log_path):
        print("  ", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    print("  files:", sorted(p.name for p in log_path.iterdir())[:20], flush=True)

    # --- checkpoint → profile → sample through the real driver
    recs = checkpoint_utils.load_checkpoints_file(str(log_path))
    with_sampler = [r for r in recs if r.sampler_path]
    assert with_sampler, f"no sampler checkpoint in {log_path}/checkpoints.jsonl: {recs}"
    ck = with_sampler[-1]
    name = f"qwen4b-{run_name}-step{ck.batch if hasattr(ck, 'batch') else len(with_sampler)}"
    profile = EndpointProfile(name=name, kind="tinker", model=ck.sampler_path, base_model=MODEL, renderer="qwen3_5", max_context=32768, max_generation_tokens=2048)
    add_profile(profile)
    print(f"[ckpt] {name} -> {ck.sampler_path}", flush=True)

    task = read_all(paths.TASKS_EVAL / "smoke_sweqa_flask.jsonl", Task)[1]
    client = await asyncio.wait_for(asyncio.to_thread(make_client, profile), timeout=120)
    env = RepoEnv(task, profile)
    trace = await asyncio.wait_for(run_episode(env, client), timeout=300)
    out = save_trace(trace, "dev")
    print(f"[sample] {task.task_id}: stop={trace.stats.stop_reason} turns={trace.stats.turns} calls={trace.stats.tool_calls} "
          f"prompt_tokens={trace.stats.prompt_tokens} -> {out}", flush=True)
    print("[sample] answer:", trace.answer[:400].replace("\n", " "), flush=True)

    manifest = paths.MODELS_MANIFEST
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    records.append(CheckpointRecord(name=name, run=run_name, step=int(getattr(ck, "batch", len(with_sampler))), created_at=datetime.now(timezone.utc),
                                    tinker_path=ck.sampler_path, profile=name, notes="A6 smoke: 10 graphiti tasks × 4 × 3 steps").model_dump(mode="json"))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=1))
    print(f"[manifest] {manifest} now has {len(records)} record(s)", flush=True)
    print(f"SMOKE_TRAIN PASSED in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "smoke1"))
