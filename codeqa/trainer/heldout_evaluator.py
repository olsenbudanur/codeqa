"""Held-out eval inside the training loop (gap_specs §5): the cookbook's RLTestSetEvaluator at temperature 0.2.

Lives in the trainer (not evals/) because trainer and evals may not import each other; evals reads the
`<name>/env/all/*` rows this writes to metrics.jsonl (name defaults to `eval/fast`, so keys read `eval/fast/env/all/reward`). Each eval task is one group of size 1, so the metrics
are plain per-task means: reward, correctness, format_ok, citations_grounded, tool_calls, prompt_tokens,
answer_tokens, stop_<reason>, judge_error (all from grader.metrics) plus the cookbook's token/turn stats.
"""
from __future__ import annotations

import os

from pathlib import Path

import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator

from codeqa.trainer.dataset_builder import CodeQADataset, builders_for, load_tasks

EVAL_TEMPERATURE = 0.2


class HeldoutEvaluator(RLTestSetEvaluator):
    def __init__(self, tasks_path: str | Path, profile_name: str, max_tokens: int, name: str = "eval/fast", variant: str = "none",
                 judge_model: str | None = None, offline_judge: bool = False, max_tasks: int | None = None,
                 temperature: float = EVAL_TEMPERATURE):
        tasks = load_tasks(Path(tasks_path), max_tasks, shuffle=False)
        builders = builders_for(tasks, profile_name, group_size=1, variant=variant, judge_model=judge_model, offline_judge=offline_judge, grounded_credit=0.0, length_shaping=False,
                                reward_version="v2")   # eval reports the unshaped, length-free phase-6 reward whatever the training reward is
        super().__init__(CodeQADataset(builders, batch_size=max(len(builders), 1), publish_step=False), max_tokens=max_tokens, name=name)
        self.temperature = temperature

    async def __call__(self, sampling_client: tinker.SamplingClient, *, rollout_summary_export=None, store=None) -> dict[str, float]:
        # Skip the step-0 call: it only re-measures the untrained model, which every run already has on record
        # (lead, 2026-09-19). Set CODEQA_SKIP_FIRST_EVAL=0 to keep it.
        self._calls = getattr(self, "_calls", 0) + 1
        if self._calls == 1 and os.environ.get("CODEQA_SKIP_FIRST_EVAL", "1") != "0":
            return {}
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens, temperature=self.temperature)
        return await self.eval_token_completer(policy, rollout_summary_export=rollout_summary_export, store=store)
