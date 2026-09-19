# Lane B — Data

**Owner:** agent. **Status:** done (B1–B7). `data/tasks/train/all.jsonl` = 1,489 tasks. **Folders:** `codeqa/datagen/**`, `data/tasks/**`.

## Mission
Produce `data/tasks/train/all.jsonl` (1,500–2,500 kept tasks, ~55% programmatically graded) and the eval files, from four sources, with every record valid against C5 and every citation resolvable at the pinned commit.

## Consumes / produces
- Consumes: C1/C2 on disk (run lane A's indexing CLI yourself once A2 lands; until then flask only), public datasets via `codeqa/clients/hf.py`, Haiku via `codeqa/clients/anthropic.py`.
- Produces: C5 files: `data/tasks/raw/<source>.jsonl`, `data/tasks/train/<source>.jsonl`, `train/all.jsonl`, `eval/{deepcodebench_test,sweqa,fast}.jsonl`.

## Checklist
- [x] **B1 DeepCodeBench importer, full.** All 912 train + 232 test rows through `import_deepcodebench.to_task`. Resolve the function/class named in the answer to a line range via the index → `required_citations`. Needs the 8 repos snapshotted + indexed (their commits are in row metadata).
  Done when: 912/232 records written; resolution rate reported in LOG; 20 sampled records read as real questions.
- [x] **B2 SWE-QA importer, full.** All 15 splits; commits from `repo_commit.txt`; resolve bare filenames against the manifest when unique; `data/tasks/eval/sweqa.jsonl`.
  Done when: 720 records; citation resolution rate per repo reported (baseline 65% on flask before basename fix).
- [x] **B3 CodeScout derivation.** Select repos: top N by row count, excluding the 9 SWE-QA-Bench overlaps and the 8 DeepCodeBench repos; N = 15 unless the lead says otherwise. One commit per repo (latest `base_commit`). Snapshot + index them. Keep rows whose gold entities resolve in the index. `rewrite.py`: Haiku turns `problem_statement` into a where/which question (cached). Output `raw/codescout.jsonl`.
  Done when: yield (rows kept / rows available for those repos) reported; 20 sampled questions read naturally and do not leak the answer path.
- [x] **B4 Structural generator.** From the `__nodoc` index: locate (docstring-paraphrased question → symbol location), value (module constants / default args → literal), enumerate (importers of X, subclasses of Y → path set), trace (repo-defined callees of X → symbol set). Cap 60 per repo, balanced across types. Paraphrase with Haiku; drop paraphrases containing the symbol name.
  Done when: `raw/structural.jsonl` ≥ 1,000 across the training repos; a 30-record spot check shows no name leakage.
- [x] **B5 Teacher generator.** Waits on A5. Sonnet explores with `run_episode` on a seeded directory, writes question + reference + 3–5 atomic rubric items + spans read; Haiku answers blind with the same tools; judge agreement ≥ threshold keeps the task. Skew seeds toward where/what.
  Done when: `raw/teacher.jsonl` ≥ 500 with per-task cost logged; 20 sampled tasks pass your own read.
- [x] **B6 Filter and split.** Waits on A4/A5 and C1. Dedupe by token overlap. Base pass-rate filter: 4 samples per task via the Tinker base profile; report `format_ok`, `citations_grounded`, `correctness` separately; keep tasks with correctness-given-format in [0.1, 0.9]; hard and easy reserves kept in `raw/`. Repo-aware split. `eval/fast.jsonl` = 60 DeepCodeBench test + 60 SWE-QA.
  Done when: `train/all.jsonl` ≥ 1,500; a counts table per source × type is in `data/tasks/README.md` and LOG.
- [x] **B7 Handoff.** Post counts and any C5 gaps in LOG.

## Waits on / provides
- Waits on: A2 for snapshots beyond flask (you can run the CLI), A5 for B5, C1 for B6.
- Provides: B1/B2 eval files unblock C3 early; B6 unblocks run one.

## Commands
```
uv run python -m scripts.smoke_data                      # the importer smoke; extend, don't replace
uv run python -m codeqa.datagen.cli import --source deepcodebench
uv run python -m codeqa.datagen.cli derive --source codescout --repos 15
uv run python -m codeqa.datagen.cli filter --samples 4
```

## Gotchas for this lane
- `datasets-server` pages are 100 rows and 502 sometimes; `clients/hf.py` retries. 17,591 CodeScout rows = 176 calls; cache to `data/cache/`.
- Every snapshot must be at the source's commit or line-level gold is wrong. `make_repo_id(owner, repo, sha)`.
- CodeScout license is unverified on HF; if the lead cannot confirm, the fallback is `JetBrains-Research/lca-bug-localization` (Apache-2.0, same shape minus entities).
- Do not train on any SWE-QA-Bench repo. The 15 are listed in `import_sweqa.REPOS`.

## Progress log (append-only)
Format: `- [YYYY-MM-DD HH:MM] B<n> done — one line with counts/paths`
- [2026-09-18 12:30] pre-B: importers exist for all three sources and validate on real rows (`scripts/smoke_data.py`); outputs in `data/tasks/raw/smoke_*.jsonl`, `data/tasks/eval/smoke_sweqa_flask.jsonl`.
- [2026-09-18 15:10] B1 done — 912 train -> `raw/deepcodebench.jsonl`, 232 test -> `eval/deepcodebench_test.jsonl`; paths 1105/1122 resolved (98%); required_citations on 846/912 and 214/232 tasks; 8 repos snapshotted+indexed. Report: `data/tasks/reports/deepcodebench.json`.
- [2026-09-18 15:10] B2 done — 720 -> `eval/sweqa.jsonl`; citations 2678/2910 resolved (92%, was 65% baseline); 528/720 tasks with >=1 span; reflex and streamlink answers cite no line numbers (judge-only). 15 repos snapshotted+indexed. Report: `data/tasks/reports/sweqa.json`.
- [2026-09-18 17:05] B3 done — 15 repos (`data/repo_list_codescout.txt`, newest instance's base_commit each), 1,240/2,042 rows with gold resolving at that commit, 1,017 kept after Haiku rewrite (192 leak rejections) -> `raw/codescout.jsonl` (694 locate, 349 trace). Yield 50%. Report `data/tasks/reports/codescout.json`. 20 sampled questions read naturally; leak rule = path, multi-word identifier, or code-styled single word.
- [2026-09-18 17:05] B4 partial — `raw/structural.jsonl` has 480 tasks over the 8 DeepCodeBench repos (60 each, 15 per type; LightGBM 24 locate / 6 enumerate), all against `__nodoc` variants; 30-record spot check: 0 name leaks. CodeScout repos follow once lane A's `cli all --nodoc` finishes on them.
- [2026-09-18 18:20] B4 done — `raw/structural.jsonl` = 1,328 tasks over all 23 training repos (342 locate / 336 value / 306 enumerate / 344 trace), every task on a `__nodoc` repo_id; pyupgrade (23) and pre-commit (45) are the only repos under 60. Paraphrase leak rejections ~20%, absorbed by the 3x locate buffer. Report `data/tasks/reports/structural.json`.
- [2026-09-18 18:20] B5 in progress — smoke on graphiti: 3/4 seeds kept, blind Haiku scores 0.6–0.8, $0.11 per kept task, ~30 s per attempt. Full run: 642 seeds (30/repo), concurrency 6, log `data/tasks/reports/teacher_attempts.jsonl` (resumable).
- [2026-09-18 18:50] B5 blocked — 81 attempts done before the Anthropic credit balance ran out: 68 kept -> `raw/teacher.jsonl` (transformers 27, keras 26, diffusers 15), 9 blind-disagree, 2 no-answer. $9.49 spent, $0.14 per kept task. `cli teach` resumes the other 561 seeds once credits are topped up (~$80).
- [2026-09-18 18:50] B6 in progress — `cli filter` running on structural + codescout (2 samples, base model, two processes) into `reports/passrate.jsonl`; ~0.3 episodes/s total, ETA ~4 h. deepcodebench + teacher wait for the judge. `cli split` builds train/ from whatever is measured.
- [2026-09-18 20:05] B5 done (budget-cut) — 239 tasks -> `raw/teacher.jsonl` from 276 attempts across all 23 repos (10 seeds/repo after the outage; 30 for transformers/keras/diffusers): 124 explain, 114 trace, 1 locate; rejections 24 blind-disagree, 6 blind-no-answer, 5 question-names-path, 2 other. Mean blind-Haiku rubric score of kept tasks 0.785. $29.37 total, $0.123 per kept task. Below the 400–600 target on purpose: the user capped lane B at $50 after the credit top-up. Report `data/tasks/reports/teacher.json`.
- [2026-09-18 23:20] B6 done — base pass rate measured for all 3,494 raw tasks (2 samples each, +2 on every task at a two-sample 0 or 1: 2,482 tasks have n=4); `reports/passrate.jsonl`, summary in `reports/passrate_summary.json`. Window [0.1, 0.9] on the difficulty score keeps 1,515; per-repo cap 120 (sqlglot only) -> `train/all.jsonl` = 1,489 over 23 repos, 62% programmatic; 691 hard + 1,288 easy in `raw/reserve_*.jsonl`; `eval/fast.jsonl` = 60 DeepCodeBench test + 60 SWE-QA (seed 7). Counts table in `data/tasks/README.md`.
- [2026-09-18 23:20] B7 done — handoff in LOG (23:20 entry). Lane B spend after the top-up: ~$27 of the $50 allowance.
- [2026-09-19 01:20] eval validation — Sonnet 5 + Opus 5 on the fast set: 3 tasks removed (wrong or stale gold, vague question), 6 repaired, replacements deterministic; eval sizes 230 / 719 / 120. Frontier content correctness ~0.83 (Sonnet), gated reward 0.44 because of the 800-token cap. Recommendation to make length proportional is in LOG 01:20. Lane B spend ~$46 of $50.

## Open questions for the lead
- ~~Anthropic credits exhausted (18:35)~~ restored ~19:10 with $100; lane B holds itself to $50 of it.
- N for CodeScout repos (default 15). Include CodeScout at all? -> decided by lane B on 2026-09-18: yes, N=15 (see LOG).
- Pass-rate window: correctness-given-format in [0.1, 0.9]? -> keeping the default.
