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
4. Grounded-citation credit (lead, 2026-09-19): an answer that passes every gate but scores 0 gets a floor of +0.05
   (`group_rewards.grounded_credit`, `--grounded-credit`, 0 disables). Ladder: stall −0.1 < no citations 0 < grounded-but-wrong 0.05 < correct.
   The held-out evaluator reports the unshaped reward (`eval/fast/env/all/reward`); `env/all/reward_shaped` is what trains.
5. Length shaping (lead, 2026-09-19 night): the trained reward is `(grader reward + grounded credit) × length_factor + stall penalty`,
   where `length_factor = min(1, cap / answer tokens)` (floor 0.1). The grader and every eval are length-free (`length_shaping=False`
   in the held-out evaluator); `env/all/length_factor_applied` logs the multiplier.
6. Judge failures: the grader returns NaN; here they become 0 + mean(healthy siblings), so their advantage is exactly 0. A group
   where every sample failed is constant and dropped by `remove_constant_reward_groups`.
7. Every grader metric is attached per trajectory; the cookbook means them into `env/all/<key>`, `env/<source>/<key>`,
   `env/<task_type>/<key>`. Group metrics (`group_reward_std`, `unique_tool_sequences_per_group`, `judge_error_rate`) ride along.

## Metric keys in metrics.jsonl

`env/all/reward/total` (cookbook), `env/all/reward`, `format_ok`, `citations_parse`, `citations_exist`, `citations_grounded`,
`identifier_grounded`, `correctness`, `efficiency`, `judge_error`, `tool_calls`, `tool_errors`, `prompt_tokens`, `completion_tokens`,
`answer_tokens`, `redundant_reads`, `turns`, `gate_<format|citations|grounding|budget|judge_error>`, `stop_<reason>`,
`reward_shaped`, `no_answer_penalty`, `grounded_credit`, `group_reward_std`, `group_reward_mean`, `unique_tool_sequences_per_group`, `judge_error_rate`, `group_all_judge_errors`;
the same under `eval/fast/env/all/` for the held-out set (plus `eval/fast/env/all/by_group/*` from the cookbook); `optim/lr`, `progress/*`, `time/*` from the cookbook.

## Run one (as decided; the lead launches)

```
uv run python -u -m codeqa.trainer.run --tasks data/tasks/train/all.jsonl --profile qwen4b-base --run-name run1 \
    --lr 1e-4 --group-size 8 --groups-per-batch 32 --steps 50 --eval-tasks data/tasks/eval/fast.jsonl --eval-every 10 --save-every 10
```
32 groups/step because only ~12 % of groups carry a correctness signal at step 0 (LOG 2026-09-19 15:40); 1,489 tasks / 32 = 47 batches
per epoch, so 50 steps ≈ 1.07 epochs (`--epochs 2` if you go past 47 steps).

## SFT warm start (only if `env/all/citations_parse` is flat by step 10)

`codeqa/trainer/sft.py`: `build` turns Claude traces over TRAINING tasks into cookbook conversations rendered exactly as `RepoEnv`
renders them (tool prefix + prompt), keeping only gate-passing, reward ≥ 0.5 traces; `train` runs the cookbook supervised loop.
Generating ~200 traces costs ≈ $30 of Sonnet (60k input tokens per episode); see the module docstring for the three commands.

## Run two (from a run-one checkpoint)

`--load-checkpoint` takes the **state path** (`tinker://…/weights/<step>` from `data/logs/run1/checkpoints.jsonl`, key `state_path`),
not the sampler path: the cookbook calls `create_training_client_from_state_async(path)` (weights only, fresh optimizer).

```
uv run python -u -m codeqa.trainer.run --tasks data/tasks/train/run1_verifiable.jsonl --profile qwen4b-base --run-name run2 \
    --variant multiplicative --load-checkpoint tinker://<run1>/weights/<step> --lr 1e-4 --group-size 8 --groups-per-batch 16 \
    --steps 50 --eval-tasks data/tasks/eval/fast.jsonl --eval-every 10 --save-every 10
```

## Loss and optimizer metrics

Tinker computes the loss server-side and returns only per-token training logprobs, so the cookbook logs no loss.
`codeqa/trainer/metrics_patch.py` (installed by `run.py`) adds per step: `optim/loss` (importance-sampling policy-gradient
surrogate, `-mean(ratio × advantage)`; near 0 by construction because advantages are group-centered, so read its drift, not its
level), `optim/loss_abs` (learning-signal magnitude), `optim/advantage_std`, `optim/frac_tokens_with_advantage`,
`optim/importance_ratio_mean|max`, `optim/clip_fraction` (|ratio − 1| > 0.2), `optim/nll`, `optim/action_tokens`.
The monitor's optimizer table and `plots/optimizer.png` show them. Runs launched before 2026-09-20 (run one) do not have them.

## Monitoring

`uv run python -m codeqa.evals.monitor --run run1` prints the per-step table, the held-out rows, the optimizer table (lr, entropy, sampler-vs-trainer KL, post-update KL with `--compute-post-kl`, KL to base with `--kl-penalty`), the collapse checks
(group reward std, unique tool sequences, stop-by-budget/max-turns rate, format gate, judge errors) and writes `plots/`.
