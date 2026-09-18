"""Build the cookbook `train.Config` for a run (C2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tinker_cookbook import hyperparam_utils
from tinker_cookbook.rl import train

from codeqa.clients.tinker import base_model_of
from codeqa.shared import paths
from codeqa.shared.profiles import get_profile
from codeqa.trainer.dataset_builder import CodeQADatasetBuilder


@dataclass
class RunSpec:
    tasks: str
    profile: str = "qwen4b-base"
    run_name: str = "dev"
    steps: int | None = None
    variant: str = "none"                  # grader efficiency variant: none | multiplicative | hard_cap | token_cost
    group_size: int = 8
    groups_per_batch: int = 32           # 16 -> 32 after the training-file probe (LOG 2026-09-19 15:40): ~4 groups/step carry a correctness signal
    lora_rank: int = 32
    learning_rate: float | None = None     # None -> hyperparam_utils.get_lr(base model, LoRA)
    eval_every: int = 10
    eval_tasks: str | None = None          # data/tasks/eval/fast.jsonl
    eval_max_tasks: int | None = None
    save_every: int = 10
    judge_model: str | None = None         # None -> grader default (Haiku 4.5)
    offline_judge: bool = False            # KeywordJudge; smoke tests only
    max_tasks: int | None = None
    seed: int = 0
    grounded_credit: float = 0.05          # shaping floor for gate-passing wrong answers (0 = off)
    epochs: int = 1                        # passes over the task file; steps = epochs * ceil(tasks / groups_per_batch) unless --steps caps it
    wandb_project: str | None = None
    wandb_name: str | None = None
    num_groups_to_log: int = 4
    extra: dict = field(default_factory=dict)

    @property
    def log_path(self) -> Path:
        return paths.LOGS / self.run_name


def build_config(spec: RunSpec) -> train.Config:
    profile = get_profile(spec.profile)
    if profile.kind != "tinker":
        raise ValueError(f"profile {spec.profile!r} is kind={profile.kind}; training needs a tinker profile")
    base = base_model_of(profile)
    lr = spec.learning_rate or hyperparam_utils.get_lr(base, is_lora=True)
    dataset_builder = CodeQADatasetBuilder(
        tasks_path=spec.tasks, profile_name=spec.profile, group_size=spec.group_size, groups_per_batch=spec.groups_per_batch,
        variant=spec.variant, judge_model=spec.judge_model, offline_judge=spec.offline_judge, max_tasks=spec.max_tasks, seed=spec.seed, epochs=spec.epochs, grounded_credit=spec.grounded_credit,
    )
    evaluator_builders = []
    if spec.eval_tasks:
        from codeqa.trainer.heldout_evaluator import HeldoutEvaluator
        evaluator_builders.append(lambda: HeldoutEvaluator(
            spec.eval_tasks, spec.profile, max_tokens=profile.max_generation_tokens, name="eval/fast", variant=spec.variant,
            judge_model=spec.judge_model, offline_judge=spec.offline_judge, max_tasks=spec.eval_max_tasks))
    extra = dict(spec.extra)
    if extra.get("kl_penalty_coef", 0) > 0 and "kl_reference_config" not in extra:
        extra["kl_reference_config"] = train.KLReferenceConfig(base_model=base)   # KL to the untrained base, logged as kl_ref/*
    return train.Config(
        model_name=base,
        recipe_name="codeqa_rl",
        renderer_name=profile.renderer,
        log_path=str(spec.log_path),
        dataset_builder=dataset_builder,
        learning_rate=lr,
        max_tokens=profile.max_generation_tokens,
        lora_rank=spec.lora_rank,
        max_steps=spec.steps,
        eval_every=spec.eval_every if spec.eval_tasks else 0,
        evaluator_builders=evaluator_builders,
        save_every=spec.save_every,
        remove_constant_reward_groups=True,
        rollout_json_export=True,
        num_groups_to_log=spec.num_groups_to_log,
        wandb_project=spec.wandb_project,
        wandb_name=spec.wandb_name or (spec.run_name if spec.wandb_project else None),
        load_checkpoint_path=extra.get("load_checkpoint_path"),
        **{k: v for k, v in extra.items() if k != "load_checkpoint_path"},
    )
