# codeqa/grader — task + trace → reward (C7)

Consumes a **C5 task** and a **C6 trace** (plus the **C1 snapshot** and **C2 symbols** of the task's repo, read from
`data/repos/<repo_id>/` and `data/index/<repo_id>/symbols.json`). Produces a **C7 `GradeResult`** and a flat metrics
dict for the trainer and evals. Imports only `codeqa.shared` and `codeqa.clients`.

```
uv run pytest codeqa/grader                                   # 55 offline tests, < 1s
uv run pytest codeqa/grader -m live                           # + Haiku on 5 hand-graded answers (needs ANTHROPIC_API_KEY)
uv run python -m codeqa.grader --tasks tests/fixtures/tasks.jsonl --traces tests/fixtures/traces --judge keyword
uv run python -m codeqa.grader --tasks data/tasks/eval/x.jsonl --traces data/traces/<run> --judge haiku --variant multiplicative --out grades.jsonl
```

## Reward

```
gates (in order, any failure → reward 0, gate_failed set):
  format      final answer exists, stop_reason not parse_error|overflow|error, answer ≤ max_answer_tokens (Qwen tokenizer; proxy fallback)
  citations   at least one [path:Lstart-Lend]; every cited file exists and the range is inside it
  grounding   every cited line was read this episode (union of files_read, per line)
  budget      tool_calls ≤ max_tool_calls (errors count); hard_cap variant also caps prompt tokens
correctness:
  locate | value | enumerate | codescout   verifiers, zero API calls: literal (whole-token, normalized) → symbol set → path set
                                           one gold: exact / any-of; several gold: F1. A symbol counts only if the answer names it
                                           AND a citation covers its definition line (identifier grounding, gap_specs §10)
  trace | explain with rubric or reference  Haiku judge: fraction of atomic items satisfied (rubric mode) or of ≤6 facts it
                                           derives from the reference (reference mode). 3 retries with backoff, then NaN.
efficiency (efficiency.py, `--variant`): none = 1.0 (run one) · multiplicative · hard_cap · token_cost (see below)
reward = correctness × efficiency    |    judge failure → reward NaN, gate_failed = judge_error (trainer maps NaN → group mean)
```

Only the final assistant content is graded; `<think>` blocks and the `thinking` field are dropped. A last turn that still
calls tools, a parse error, or an overflow means "no answer" and fails the format gate.

## Efficiency variants (gap_specs §6)

Budgets: `max_tool_calls` from the task; prompt-token budget = `max_tool_calls × 3000`. Usage is a fraction of budget.
The first half of the budget is free; beyond it, eff falls linearly from 1.0 at 50 % to 0.5 at 100 % and floors at 0.5.

| variant | usage | gate |
|---|---|---|
| `none` | – (eff = 1) | – |
| `multiplicative` | max(calls / budget, prompt_tokens / token_budget); a read span already fully covered by earlier reads counts as an extra call | – |
| `hard_cap` | – (eff = 1) | calls or prompt tokens over budget → `budget` gate, reward 0 |
| `token_cost` | prompt_tokens / token_budget only | – |

## Threat model (gap_specs §3) — `tests/test_adversarial.py`, fixtures in `tests/fixtures/traces/`

| Hack | Why it would score | Counter | Fixture | Scores |
|---|---|---|---|---|
| Padding, verbosity | judge length bias | `max_answer_tokens` gate; judge sees the answer truncated to the cap; atomic yes/no items; judge told not to prefer longer | `padded.json` | gate `format`, 0 |
| Confident wrong answer | judge fooled by tone | verifiable types never use the judge; judged types anchored to rubric/reference | `wrong.json` | 0, judge not called |
| Restating the question | judge partial credit | no item satisfied → 0 | `restated.json` | 0 |
| Fabricated citation | looks cited | `citations_exist` gate (file and line range checked against the snapshot) | `fabricated.json` | gate `citations`, 0 |
| Citing a plausible unopened file | looks grounded | `citations_grounded` gate against `files_read`, per line | `unread_citation.json` | gate `grounding`, 0 |
| Zero tool calls | efficiency would reward it | nothing read → nothing grounded → 0 | `no_tools.json` | gate `grounding`, 0 |
| Judge injection | judge obeys | markdown stripped, closing marker escaped, wrapped in delimiters, declared untrusted | `judge_injection.json` | 0 |
| Answer hidden in thinking | thinking not graded | answer extracted from final content only | `thinking_answer.json` | 0 |
| Repeated reads to bulk up `files_read` | grounding via volume | grounding is per cited line, so volume buys nothing; redundant reads count as extra calls under `multiplicative` | `redundant_reads.json` | 1.0 under `none`, eff < 1 under `multiplicative` |
| Tool errors not counted | free calls | every call counts, errors included | `tool_errors.json` | gate `budget`, 0 |
| Copying tool output verbatim | rubric hits by accident | answer cap; rubric items require synthesis | `verbatim.json` | gate `format`, 0 |

## Files

| file | what |
|---|---|
| `repo.py` | `RepoFiles`: snapshot line counts + `symbols.json` lookups; `mini__repo__0000001` maps to `tests/fixtures/` |
| `gates.py` | `extract_answer`, `format_gate`, `citations_parse_gate`, `budget_gate`, `approx_tokens` |
| `citations.py` | `parse_citations`, `check_citations(answer, files_read, repo_id, expected_symbols)` → `CitationReport` |
| `verifiers.py` | `literal_score`, `path_score`, `symbol_score`, `uses_judge`, `verify` |
| `judge.py` | prompts, `judge(...)` → `JudgeVerdict`, `KeywordJudge` offline stand-in, `default_client()` (Haiku 4.5) |
| `efficiency.py` | `efficiency(stats, budget, variant)`, `redundant_reads` |
| `grade.py` | `grade(task, trace, variant, judge_client, repo)`, `grade_sync`, `metrics(result, trace, task)` |
| `cli.py` | `python -m codeqa.grader` |

## Metrics emitted (`grade.metrics`)

`reward format_ok citations_parse citations_exist citations_grounded identifier_grounded correctness efficiency judge_error
tool_calls tool_errors prompt_tokens completion_tokens answer_tokens redundant_reads turns gate_<name> stop_<reason>`.
`reward` and `correctness` are reported as 0 when the judge failed; `judge_error` = 1 marks those rows. Under the cookbook
they aggregate as `env/all/<key>` and `env/<tag>/<key>`.

## Known limits

- Answer length uses the policy tokenizer (`Qwen/Qwen3.5-4B` via `clients/tinker.py`, cached) when loadable and otherwise the proxy `max(words, chars/4)`, which lands within ~5-10 % of it on real answers. `CODEQA_GRADER_TOKENIZER=proxy` forces the proxy (tests do).
- `KeywordJudge` is a test stand-in: it checks that backticked/quoted terms of each rubric item appear in the answer. Never use it for training.
- Reference mode lets the judge choose the atomic facts, so scores are less stable than rubric mode. Prefer rubrics (DeepCodeBench facts, teacher rubrics).
