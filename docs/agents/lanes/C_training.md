# Lane C — Training (grader, trainer, evals)

**Owner:** agent. **Status:** C1–C5 done on smoke; run-one kickoff waits on the lead's reward/lr decision (LOG 16:40). **Folders:** `codeqa/grader/**`, `codeqa/trainer/**`, `codeqa/evals/**`.

## Mission
Turn a trace into a reward the trainer can trust, configure the cookbook loop around lane A's environment, and produce the curves and tables for the talk.

## Consumes / produces
- Consumes: C5 tasks, C6 traces, C2 index (identifier grounding), C8 profiles, lane A's `RepoEnv.make_cookbook_env`, Haiku via `codeqa/clients/anthropic.py`.
- Produces: C7 `grade()` and `check_citations()`, trainer config + run, `data/logs/<run>/`, `data/evals/<profile>/<set>/`, plots.

## Checklist
- [x] **C1 Grader.** `gates.py` (format valid, citations parse, answer length ≤ budget), `citations.py` (`check_citations(answer, files_read, repo_id, expected_symbols)` → exists, grounded, `anchors_symbol` via the index), `verifiers.py` (exact path, normalized literal, symbol-set F1, path F1 when > 1 gold), `judge.py` (rubric mode: fraction of atomic items satisfied; fact-recall mode for DeepCodeBench facts; Haiku; markdown stripped; answer wrapped as untrusted data; 3 retries then `NaN`), `efficiency.py` (variants `none|multiplicative|hard_cap|token_cost`), `grade.py` → `GradeResult` with all components; `README.md` with the threat-model table (gap_specs §3). Adversarial traces: run 3 real episodes with lane A's driver (or start from `data/smoke_episode*.log` traces), mutate them into the 11 cases in `tests/fixtures/traces/`, and make `tests/test_adversarial.py` assert each scores as intended.
  Done when: adversarial tests pass; verifiable tasks grade with zero API calls; judge tested on 5 hand-graded answers.
- [x] **C2 Trainer.** Waits on A4/A6. `dataset_builder.py`: `Task` → `EnvGroupBuilder` whose `make_envs` returns `group_size` copies of `RepoEnv.make_cookbook_env(grade_fn)`; `compute_group_rewards` replaces `NaN` with the group mean; `logging_tags` by source and task_type. `config.py`: profile, `lora_rank=32`, lr from `hyperparam_utils.get_lr`, `group_size=8`, `groups_per_batch=32`, `eval_every=10`, `remove_constant_reward_groups=True`, `rollout_json_export`, `num_groups_to_log`, `wandb_*` optional, `reward_variant`. `run.py` CLI with `--tasks --profile --steps --variant --run-name`.
  Done when: 10 tasks × group 4 × 3 steps completes with the real grader; `metrics.jsonl` has `env/all/{reward,format_ok,citations_grounded,correctness,efficiency,tool_calls,prompt_tokens}` and `group_reward_std`, `unique_tool_sequences_per_group`.
- [x] **C3 Evals.** `heldout_evaluator.py` implementing the cookbook evaluator interface over `data/tasks/eval/fast.jsonl` (temperature 0.2); `run.py` (`--profile --tasks` → `data/evals/<profile>/<set>/per_task.jsonl` + `results.json`); `report.py` (table by source × task_type); `plots.py` (reward, correctness, citation validity, tool calls per correct, from `metrics.jsonl` and `data/evals/`); `sweqa_judge.py` (the five-dimension prompt from `docs/research/swe_qa_llm_as_a_judge.py`, Haiku or Sonnet, for the external number only, never for training reward). Update `CheckpointRecord.evals` in `data/models/manifest.json` after each run.
  Done when: `uv run python -m codeqa.evals.run --profile claude --tasks data/tasks/eval/smoke_sweqa_flask.jsonl` produces outputs and a plot.
- [x] **C4 Run two prep.** Efficiency variant tests on the same trace; document the formula in the grader README; a `--variant` flag path proven on the smoke.
- [x] **C5 Handoff.** Post metric key names and the manifest format in LOG.

## Waits on / provides
- Waits on: A4/A5/A6 for C2 and the in-loop evaluator; B2 for real eval files (use smoke files meanwhile).
- Provides: C1 unblocks B6 and D2's verified badges; C3 unblocks the talk.

## Commands
```
uv run pytest codeqa/grader
uv run python -m codeqa.trainer.run --tasks data/tasks/raw/smoke_deepcodebench.jsonl --profile qwen4b-base --steps 3 --run-name smoke
uv run python -m codeqa.evals.run --profile claude --tasks data/tasks/eval/smoke_sweqa_flask.jsonl
```

## Gotchas for this lane
- `RewardFn` is `async (history) -> (float, dict)`, called once at episode end with the full message list; `files_read` comes from the env object, not the history.
- Format failure → 0. Judge failure → `NaN` → group mean (advantage 0). Never group-average a format failure (SWE-QA-Pro did; it removes format pressure).
- Judge only ever sees the answer truncated to `max_answer_tokens`, with markdown stripped, inside delimiters, with an instruction that it is untrusted.
- Thinking is not graded; extract the answer from final content only.
- Every tool call counts toward the budget, errors included.

## Progress log (append-only)
Format: `- [YYYY-MM-DD HH:MM] C<n> done — one line with paths/numbers`
- [2026-09-18 12:55] pre-C: a 15-line stub reward (citations parsed + grounded + correct path) ran inside a real episode (`scripts/smoke_episode.py`); step-0 model fails only on citation format.
- [2026-09-18 21:40] lead's 4 items done — no-answer penalty in the group hook (+tests), baselines qwen4b-base/claude on fast (0.00/0.20) and sweqa_100 with Sonnet judge (40.6/73.1), `evals.monitor` with collapse checks, run-two command in trainer README (state path). Open for the lead: answer cap (LOG 21:00), Anthropic credit (LOG 20:10)
- [2026-09-18 16:45] C2–C5 done — trainer smoke 3 steps on real Tinker (data/logs/smoke3), evals harness + plots + SWE-QA judge live, variant flag proven on fixtures; collapse finding + proposal in LOG 16:40
- [2026-09-18 15:20] C1 done — codeqa/grader (56 tests, live judge test, CLI on fixtures, e2e via scripts/smoke_grade.py); see LOG 15:20
- [2026-09-18 14:10] pre-C: Tinker verified live (episode + LoRA training client + sampler export); cookbook loop API mapped, see LOG entries of 14:10. No code written yet.

## Open questions for the lead
- Confirm gating order and the run-two multiplicative formula (`eff` in [0.5, 1] from tool calls + prompt tokens vs budget).
- Judge model: Haiku for training reward, Sonnet for the final SWE-QA number?
