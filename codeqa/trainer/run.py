"""Run RL training (C2).

  uv run python -u -m codeqa.trainer.run --tasks data/tasks/train/run1.jsonl --profile qwen4b-base --run-name run1 \
      --eval-tasks data/tasks/eval/fast.jsonl --eval-every 10
  uv run python -u -m codeqa.trainer.run --tasks data/tasks/eval/smoke_sweqa_flask.jsonl --profile qwen4b-base \
      --steps 3 --group-size 4 --groups-per-batch 10 --run-name smoke

Outputs under data/logs/<run-name>/: metrics.jsonl (env/all/*, env/<source>/*, env/<task_type>/*, eval/fast/*),
checkpoints.jsonl (tinker:// paths), per-iteration rollout summaries.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from codeqa.grader.efficiency import VARIANTS
from codeqa.trainer.config import RunSpec, build_config


def parse(argv: list[str] | None = None) -> RunSpec:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--profile", default="qwen4b-base")
    ap.add_argument("--run-name", default="dev")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--variant", default="none", choices=VARIANTS)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--groups-per-batch", type=int, default=32)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-tasks", default=None)
    ap.add_argument("--eval-max-tasks", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--offline-judge", action="store_true", help="KeywordJudge instead of Haiku (smoke only)")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grounded-credit", type=float, default=0.05, help="reward floor for answers that pass every gate but score 0 (0 disables)")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--if-exists", default="delete", choices=["delete", "resume", "raise", "ask"])
    ap.add_argument("--load-checkpoint", default=None, help="tinker:// path to start from")
    ap.add_argument("--compute-post-kl", action="store_true", help="also log optim/post_kl (KL after the update; one extra forward pass per step)")
    ap.add_argument("--kl-penalty", type=float, default=0.0, help="KL-to-base penalty coefficient; > 0 also logs kl_ref/* each step")
    a = ap.parse_args(argv)
    extra = {"load_checkpoint_path": a.load_checkpoint} if a.load_checkpoint else {}
    if a.compute_post_kl:
        extra["compute_post_kl"] = True
    if a.kl_penalty > 0:
        extra["kl_penalty_coef"] = a.kl_penalty          # config.py adds the KLReferenceConfig for the profile's base model
    spec = RunSpec(tasks=a.tasks, profile=a.profile, run_name=a.run_name, steps=a.steps, variant=a.variant, group_size=a.group_size,
                   groups_per_batch=a.groups_per_batch, lora_rank=a.lora_rank, learning_rate=a.lr, eval_every=a.eval_every,
                   eval_tasks=a.eval_tasks, eval_max_tasks=a.eval_max_tasks, save_every=a.save_every, judge_model=a.judge_model,
                   offline_judge=a.offline_judge, max_tasks=a.max_tasks, seed=a.seed, epochs=a.epochs, grounded_credit=a.grounded_credit, wandb_project=a.wandb_project, extra=extra)
    spec.extra["_if_exists"] = a.if_exists
    return spec


def main(argv: list[str] | None = None) -> int:
    from codeqa.shared.runtime import maybe_redirect_to_modal
    maybe_redirect_to_modal("codeqa.trainer.run", argv)   # no-op unless CODEQA_RUNTIME=modal (decision #8)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout, force=True)
    spec = parse(argv)
    if_exists = spec.extra.pop("_if_exists", "delete")
    from tinker_cookbook import cli_utils
    from tinker_cookbook.rl import train

    from codeqa.trainer.metrics_patch import install
    install()                                   # optim/loss and friends in metrics.jsonl
    config = build_config(spec)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=if_exists)
    print(f"run {spec.run_name}: model={config.model_name} lr={config.learning_rate:.2e} lora_rank={config.lora_rank} "
          f"group={spec.group_size} groups/batch={spec.groups_per_batch} steps={spec.steps} variant={spec.variant} "
          f"eval_every={config.eval_every} log={config.log_path}", flush=True)
    asyncio.run(train.main(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
