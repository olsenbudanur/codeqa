# codeqa/trainer — the RL loop (C2)

Consumes **C5 tasks** (train file + optional eval file), a **C8 tinker profile**, lane A's `RepoEnv.make_cookbook_env`,
and the grader. Produces `data/logs/<run>/` (cookbook `metrics.jsonl`, `checkpoints.jsonl`, rollout summaries).

```
uv run pytest codeqa/trainer
uv run python -u -m codeqa.trainer.run --tasks data/tasks/eval/smoke_sweqa_flask.jsonl --profile qwen4b-base \
    --steps 3 --group-size 4 --groups-per-batch 10 --run-name smoke --eval-tasks tests/fixtures/... (any C5 file with map.txt)
```

| file | what |
|---|---|
| `dataset_builder.py` | `CodeQAGroupBuilder` (one task → `group_size` RepoEnvs; grading + NaN policy + diversity metrics in `compute_group_rewards`), `CodeQADataset`, `CodeQADatasetBuilder` (chz) |
| `group_rewards.py` | pure math: NaN → group mean, advantages, `group_reward_std`, `unique_tool_sequences_per_group` |
| `heldout_evaluator.py` | in-loop eval at temperature 0.2 over an eval task file; logs `eval/fast/env/all/*` every `eval_every` steps |
| `config.py` | `RunSpec` → cookbook `train.Config` (LoRA rank 32, lr from `hyperparam_utils.get_lr`, `remove_constant_reward_groups=True`) |
| `run.py` | CLI |

## How reward flows

1. The env's `reward_fn` returns 0 and just captures the cookbook history on the `RepoEnv`.
2. `compute_group_rewards` builds a C6 `Trace` per rollout (`RepoEnv.trace_from_history` + token counts from the trajectory), grades all
   of them concurrently (judge calls behind a semaphore of 16), and returns the reward as the group-level term.
3. No-answer penalty (decisions.md evening #2): an episode that ends at `budget` or `max_turns` without a final answer gets −0.1
   (`group_rewards.no_answer_penalty`); a format failure on an actual answer stays 0, so a bad answer still beats stalling.
   `env/all/reward` is the grader's reward, `env/all/reward_shaped` the trained total, `env/all/no_answer_penalty` the rate.
4. Judge failures: the grader returns NaN; here they become 0 + mean(healthy siblings), so their advantage is exactly 0. A group
   where every sample failed is constant and dropped by `remove_constant_reward_groups`.
5. Every grader metric is attached per trajectory; the cookbook means them into `env/all/<key>`, `env/<source>/<key>`,
   `env/<task_type>/<key>`. Group metrics (`group_reward_std`, `unique_tool_sequences_per_group`, `judge_error_rate`) ride along.

## Metric keys in metrics.jsonl

`env/all/reward/total` (cookbook), `env/all/reward`, `format_ok`, `citations_parse`, `citations_exist`, `citations_grounded`,
`identifier_grounded`, `correctness`, `efficiency`, `judge_error`, `tool_calls`, `tool_errors`, `prompt_tokens`, `completion_tokens`,
`answer_tokens`, `redundant_reads`, `turns`, `gate_<format|citations|grounding|budget|judge_error>`, `stop_<reason>`,
`reward_shaped`, `no_answer_penalty`, `group_reward_std`, `group_reward_mean`, `unique_tool_sequences_per_group`, `judge_error_rate`, `group_all_judge_errors`;
the same under `eval/fast/env/all/` for the held-out set (plus `eval/fast/env/all/by_group/*` from the cookbook); `optim/lr`, `progress/*`, `time/*` from the cookbook.

## Run two (from a run-one checkpoint)

`--load-checkpoint` takes the **state path** (`tinker://…/weights/<step>` from `data/logs/run1/checkpoints.jsonl`, key `state_path`),
not the sampler path: the cookbook calls `create_training_client_from_state_async(path)` (weights only, fresh optimizer).

```
uv run python -u -m codeqa.trainer.run --tasks data/tasks/train/run1_verifiable.jsonl --profile qwen4b-base --run-name run2 \
    --variant multiplicative --load-checkpoint tinker://<run1>/weights/<step> --lr 1e-4 --group-size 8 --groups-per-batch 16 \
    --steps 50 --eval-tasks data/tasks/eval/fast.jsonl --eval-every 10 --save-every 10
```

## Monitoring

`uv run python -m codeqa.evals.monitor --run run1` prints the per-step table, the held-out rows, the collapse checks
(group reward std, unique tool sequences, stop-by-budget/max-turns rate, format gate, judge errors) and writes `plots/`.
