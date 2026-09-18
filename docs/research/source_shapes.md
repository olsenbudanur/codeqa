# Source shapes and derivability (verified 2026-09-18 via HF datasets-server rows)

All counts below come from fetched rows/splits, not descriptions. "Sample" = the rows fetched.

## SWE-QA-Pro-Bench — TIGER-Lab/SWE-QA-Pro-Bench (MIT)
- 260 rows, split `test`. 26 repos × 10 questions. qa_type: What 51, Where 77, How 67, Why 65.
- Fields: `repo` (owner/name), `commit_id` (full sha), `cluster` {id,name}, `qa_type` {class_name, sub_class_name}, `question`, `answer`.
- Answers are free text (~1.4–2k chars) with paths and line ranges inline, e.g. "src/qibo/states.py: lines 6-247".
- Overlap with SWE-QA-Bench repos: sqlfluff, sphinx, xarray (commits may differ).
- Derivation to our task record: repo+commit → snapshot; question as-is; reference_answer = answer; required_citations = regex over answer (path + "lines a-b"); task_type from qa_type (Where→locate/trace, What→value/explain, How/Why→explain); rubric = none shipped, derive atomic facts with Claude or judge vs reference. **Effort: trivial.** Use: held-out eval (26 repos).

## DeepCodeBench — Qodo/deep_code_bench (Apache-2.0)
- 1,144 rows: train 912, test 232. 8 repos, one commit each: transformers 152, diffusers 181, keras 34, fastai 209, xgboost 131, LightGBM 89, qlib 160, graphiti 188.
- Fields: `question`, `answer` (short, names file + function, no line numbers), `facts` (list of atomic statements, min 1 / median 6 / max 35), `metadata` {commit, repo (git URL), difficulty easy 47 / moderate 938 / hard 159, scope deep 598 / broad 546, n_context_files, n_context_nodes, pr, includes_location_hints, is_core_question, type "open_question"}, `id`.
- No SWE-QA overlap.
- Derivation: repo+commit → snapshot (only 8 snapshots; big repos, map must be pruned); question as-is; reference_answer = answer; **rubric = facts as-is** (already atomic, exactly our rubric format); required_citations = resolve the named function/class in answer via our symbol index → line range (programmatic). task_type mostly explain/value. **Effort: trivial.** Caveat: 8 repos only → ~114 tasks per repo; holding out by repo costs 1/8 each.

## CodeScout — OpenHands/SWE-rebench-code-search (MIT)
- 17,591 rows, split `train`. Python.
- Fields: `instance_id`, `repo`, `base_commit`, `problem_statement` (issue text, median ~500 chars), `patch` (diff), `file_changes` [{file, changes {added_entities, added_modules, edited_entities, edited_modules}}] with entities like "sepal_ui/sepalwidgets/inputs.py:DatePicker.disable".
- Sample of 300: 98% have edited entity/module gold; median 2 files touched, 82% ≤ 3 files; 0 rows with no .py; 0 rows test-only; 10% of problem statements read as questions; 76 distinct repos in 300 rows (total distinct repos unverified); no SWE-QA overlap in sample.
- Derivation: question = LLM rewrite of problem_statement into "where is X handled / which component does Y" (1 Haiku call per row); expected_paths = files; expected symbols = edited_entities/modules (function-level gold, checkable with find_symbol); task_type locate/trace; programmatic grading, no judge. **Effort: one reduction prompt + one snapshot policy.**
- Snapshot cost is the constraint: every row has its own base_commit. Policy: one commit per repo (latest base_commit), keep rows whose gold entities still exist in the index at that commit. Yield must be measured, not assumed.
- Flavor: issue-derived. The model's task is still Q&A (locate), never patching.

## SWE-QA-Bench — swe-qa/SWE-QA-Benchmark (Apache-2.0)
- 720 rows, 15 repos × 48. Question first word: what 184, how 183, why 178, where 173.
- Fields: `question`, `answer` only. Commits in `repo_commit.txt` on GitHub.
- Over all 720 answers: 84% contain a file path, 79% contain line numbers, 76% both. Median answer ~2,050 chars.
- Derivation: eval only. required_citations = regex (76% coverage). Judge = their script (research/swe_qa_llm_as_a_judge.py). **Effort: trivial.**

## SWE-QA-Pro SFT trajectories — TIGER-Lab/SWE-QA-Pro-SFT-Trajectories (MIT)
- 1,000 rows. Fields: `tools` (3), `messages` (roles: system, user, assistant, tool_call, tool_response; hermes style).
- Tools: `view_codebase(path, view_range, concise, python_only)`, `semantic_search(term, path, python_only, max_files, include_lines)`, `execute_readonly_command(command)` — a read-only shell.
- Protocol: mandatory planning text each turn; multiple tool calls per turn; final answer inside `<finish>...</finish>` with evidence (paths + lines).
- Sample of 50: median 16 assistant turns (max 25), ~71k chars per trajectory. User message carries "Repository Path: repos_tmp/.../<repo>" and the question; **commit not in the row (unverified whether the card lists commits).**
- Derivation: NOT directly usable as SFT for our tool set (shell + semantic search ≠ our five index tools; `<finish>` format ≠ ours). Conversion is lossy (~half a day). Usable as: 1,000 extra questions with evidence-bearing answers (extract from user msg + final `<finish>` block) IF commits are recoverable; and as evidence that untrained-style trajectories run ~16 turns, far above our efficiency target.

## Tally (natural repo Q&A, commit-pinned, evidence-bearing), verified so far
- Eval-grade: SWE-QA-Bench 720 + SWE-QA-Pro-Bench 260 = 980 across 38 repos.
- Train-grade natural: DeepCodeBench 912 (8 repos) (+ 232 test).
- Possibly +1,000 from SWE-QA-Pro trajectories if commits are recoverable.
- Programmatic locate at scale: CodeScout 17,591 rows; usable count depends on snapshot policy (measure).
- Natural Q&A total ≈ 2.1k–3.1k. Does not reach 5k without CodeScout-derived or self-generated tasks.
- Pending: dataset-census agent (CodeRepoQA 585k claim, LoCoBench 8k claim, others).

## Additions from the follow-up (verified from SWE-QA-Pro repo source, paper, and parquet files)

- **SWE-QA-Pro RL train set** (in their GitHub repo under `train/RL/verl-tool/dataset/`, MIT): 464 rows, 464 distinct repos, every row has `extra_info.commit_id` and `repo_name`; test 48 rows. Overlap with SWE-QA-Bench: astropy, conan (1 each). **Train-usable natural Q&A with pinned commits and wide repo diversity.**
- **SWE-QA-Pro trajectories**: 999 distinct repo names, one question per repo, no owner and no commit id in rows → not commit-pinned; skip as a training source unless a repo→commit mapping is recovered.
- **SWE-QA-Pro reward** (`SWEQAProRewardManager`): judge = GPT-4o, five 1–10 dims; `reward = (0.3*correctness + 0.2*completeness + 0.2*relevance + 0.1*clarity + 0.2*reasoning)/10`. Format gate = answer inside `<finish>`; on format or judge failure the sample receives the group-average scores. No citation check, no efficiency/turn penalty, no invalid-tool penalty.
- **SWE-QA-Pro tools**: `view_codebase(path, view_range, concise, python_only)` numbered lines, 6,000-char cap; `semantic_search(term, path, ...)` is plain substring match, not embeddings; `execute_readonly_command(command)` allowlisted shell. Usage over 1,000 SFT trajectories: view 14,422 / shell 4,952 / search 3,589 calls; mean 15 turns and 23 tool calls per trajectory.
- **SWE-QA-Pro training**: SFT full FT Qwen3-8B (teacher Claude Sonnet 4.5), then GRPO full FT, batch 8, n=8, temp 1.0, max_turns 25, KL 0.02, ~58 steps. Results (max 50): direct 26.61 → +agent 30.03 → SFT 34.34 → SFT+RL 35.39; GPT-4o+agent 33.08; Claude Sonnet 4.5+agent 40.67. RL added +1.05 over SFT.
- **DeepCodeBench grading**: fact recall — LLM checks each gold fact against the prediction, score = fraction recalled. Mean 6.6 facts/row.
- **CodeScout totals**: SWE-rebench-code-search 17,591 rows over **3,103 repos**, 2.2 files and 4.8 entities per row; overlap with SWE-QA-Bench = 9 repos (astropy, conan, matplotlib, xarray, scikit-learn, sphinx, sqlfluff, streamlink, sympy) → exclude. SWE-Gym-code-search 2,317 rows / 11 repos. SWE-smith-py 39,287 rows but base_commit null and synthetic mirror repos → skip. **HF license field is unset on the CodeScout sets → license unverified.**

## Updated tally
- Natural, commit-pinned, evidence-bearing: 720 + 260 + 512 + 1,144 = **2,636**. Train-usable natural: 912 + 464 = **1,376 across 472 repos**.
- Programmatic locate raw: CodeScout 17,591 (3,103 repos) + SWE-Gym 2,317 (11 repos).
- Natural Q&A does not reach 5k. Verifiable material exceeds 5k only via CodeScout-derived tasks and/or our structural generator.
