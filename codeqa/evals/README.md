# codeqa/evals — profile × task file → tables and plots (C3)

Consumes **C5 task files**, **C8 profiles** (`profiles.yaml`), lane A's `run_episode` (any client kind) and the grader.
Produces `data/evals/<profile>/<set>/{per_task.jsonl, results.json, traces/, sweqa_judge.json}`, `data/logs/<run>/plots/*.png`,
`data/evals/plots/eval_<set>.png`, and fills `CheckpointRecord.evals[<set>]` in `data/models/manifest.json` when a record's
`profile` matches.

```
uv run pytest codeqa/evals
uv run python -u -m codeqa.evals.run --profile qwen4b-base --tasks data/tasks/eval/smoke_sweqa_flask.jsonl          # episodes + grades
uv run python -m codeqa.evals.report --set smoke_sweqa_flask [--markdown]                                            # tables
uv run python -m codeqa.evals.plots --run smoke --run run1                                                           # training curves
uv run python -m codeqa.evals.plots --set fast --profiles qwen4b-base claude qwen4b-run1-step40                       # eval bars
uv run python -m codeqa.evals.sweqa_judge --profile qwen4b-base --set sweqa --model claude-sonnet-5                   # external number
```

| file | what |
|---|---|
| `run.py` | runs every task through `run_episode` (temperature 0.2 by default, concurrency 4, 240s per episode), grades, writes rows + summary by source × task_type, updates the manifest |
| `report.py` | headline table across profiles and the source × task_type table for one profile; plain text or markdown |
| `plots.py` | headline figure (reward, correctness, citation validity, tool calls per correct) and all curves from `metrics.jsonl` (`env/all/*` solid, `eval/fast/env/all/*` dashed); eval bars across profiles |
| `sweqa_judge.py` | the SWE-QA five-dimension strict judge (Sonnet 5 by default), 1–20 per dimension, total /100; writes `sweqa_judge.json` and `sweqa_total` into `results.json`. Never a training signal |

## results.json summary keys

`n reward correctness format_ok citations_parse citations_exist citations_grounded identifier_grounded efficiency judge_error
tool_calls tool_errors prompt_tokens completion_tokens answer_tokens turns seconds correct_rate citation_valid
tool_calls_per_correct prompt_tokens_per_correct stop_<reason> gate_<name>` (+ `sweqa_total`, `sweqa_correctness` after the judge).
`correct_rate` = share of tasks with reward > 0; `citation_valid` = `citations_grounded` (every citation exists and was read).

The in-loop held-out evaluator lives in `codeqa/trainer/heldout_evaluator.py` (trainer and evals may not import each other);
its rows land in `metrics.jsonl` under `eval/fast/env/all/*` and `plots.py` reads them from there.
