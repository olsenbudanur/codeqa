# Census of public repository-level code Q&A data (verified 2026-09-18)

Question: do 5,000+ repository-level code question-answer items exist online today that we could use to RL-train a code Q&A agent?

Ideal record: natural-language question about a specific codebase + repo URL + pinned commit/tag + reference answer + (ideally) file paths / line spans.

Verification methods used: Hugging Face `datasets-server` `/size`, `/splits`, `/rows` endpoints; HF `/api/datasets` metadata (license, gated); dataset-card READMEs (`/raw/main/README.md`); `gh api` for GitHub READMEs, trees, and raw files; direct download and inspection of zips/parquets/CSVs where the row API failed; arXiv abstracts; PDF text extraction for two papers. Anything not obtained one of these ways is marked **unverified**.

Conventions: "commit pinned" = each record (or a per-repo manifest shipped with the data) carries a commit SHA or tag. "Evidence spans" = structured file path and/or line range fields, not just paths mentioned inside answer prose.

---

## Summary table

| # | Dataset | Type | Verified count | Commit pinned | Evidence spans | License | Usable as-is |
|---|---|---|---|---|---|---|---|
| 1 | SWE-QA (SWE-QA-Bench, ACL 2026 Findings) | natural repo QA, free text | 720 (15 repos x 48) | yes (per-repo short SHA manifest; mirror has `base_commit` col) | no (paths/lines in answer prose only) | Apache-2.0 | yes (eval set; contamination caveat) |
| 2 | SWE-QA-Pro Bench (TIGER-Lab) | natural repo QA, free text | 260 (26 repos x 10) | yes (`commit_id` full SHA per row) | no (in prose) | MIT | yes (eval set) |
| 3 | SWE-QA-Pro RL train/test parquets (GitHub) | synthetic repo QA, free text | 464 train + 48 test | yes (`extra_info.commit_id`, 464/464) | no (in prose; 397/464 answers cite .py files) | MIT (repo) | yes |
| 4 | SWE-QA-Pro SFT Trajectories | agent trajectories ending in QA answer | 1,000 | partial (repo name in user turn; no SHA in row) | no | MIT | partial (need to recover commit) |
| 5 | Code-QA-Bench (Lens-Frontier) | natural repo QA, free text + rubric | 528 code-derivable + 100 doc-dependent = 628 | yes (`repos.json` full SHA per repo) | file-level (`key_files` list) | MIT | yes (eval set) |
| 6 | SWE-Bench-Pro Interactive Issue+QA (Anonym01048) | repo QA with executable proofs | 467 (Sol pool) + 118 (Terra pool) = 585 unique questions over 322 instances | yes (SHAs embedded in `instance_id`) | proof scripts + `facts`; no line spans | "other" (derived from SWE-Bench Pro) | partial (license) |
| 7 | swe-qa-episodes (aviralku) | synthetic repo QA, free text, with inline repo files | 564 QAs over 330 repos | no | no (in prose) | none stated | no (no license, no commit) |
| 8 | StackRepoQA (FSE/PROMISE 2026) | natural SO questions mapped to repos | 1,318 (134 Java repos) | no | no | none in repo (paper CC-BY-4.0) | no as-is (answers are SO posts; no commit) |
| 9 | CoReQA (arXiv 2501.03447) | natural repo QA from issues | 1,563 claimed; **no public release found** | unverified | unverified | unverified | no (not released) |
| 10 | CodeRepoQA (arXiv 2412.14764) | raw GitHub issue threads | 585,687 issue JSONs claimed (30 repos); 155-file sample verified | no | no | none stated | no (raw issues; reducible) |
| 11 | SWE-QA LREC-2026 MCQ (lailaelkoussy/swe-qa) | templated MCQ | 9,072 (x2 splits) | no (repo zips shipped, no SHA) | chunk `file_path` | Apache-2.0 | weak (MCQ, templated) |
| 12 | LongCodeQA (LongCodeBench) | MCQ from GitHub issues | 443 | no (`repo` only) | no | Apache-2.0 | weak (MCQ) |
| 13 | LoCoBench (Salesforce) | synthetic codebases, free-text analysis tasks | 8,000 (1,000 x 8 categories) | n/a (synthetic codebases shipped in zip) | `context_files` list | Apache-2.0 | partial (synthetic code; ~2,000 QA-shaped) |
| 14 | RepoQA (evalplus) | needle-function retrieval | 500 | yes (repos vendored in package) | function | Apache-2.0 | no (not QA) |
| 15 | CodeQueries (thepurpleowl) | templated CodeQL-query spans | 102,962 train / 11,183 val / 57,201 test | no | yes (line/col spans) | Apache-2.0 | no (templated, file-level) |
| 16 | RLM-Bench (mohit-awana) | hand-written repo QA | 20 (httpx 0.28.1) | yes (version tag) | no | MIT | trivial size |
| 17 | MultiRepoQA (NLPForUA/multilingual-repo-qa) | repo QA, EN/DE/UK | 783 canonical / 2,349 examples claimed on card; rows gated | unverified | unverified | CC-BY-4.0 | unverified ("release in progress") |
| 18 | DeepRepoQA questions | re-hosted SWE-QA | 575 + 143 live = 718 | no SHA in file | `relative_code_list` | Apache-2.0 | duplicate of #1 |
| 19 | vm2825 synthetic multi-file QA | synthetic QA over inline files | 5,088 + 4,995 | no | file names inline | none stated | no (no repo/commit/license) |
| 20 | Long Code Arena bug localization | issue -> changed files | 7,479 unique (java 2,522 / kt 618 / py 4,339) | yes (`base_sha`, `head_sha`) | `changed_files` + diff | Apache-2.0 | reducible to localization QA |
| 21 | Long Code Arena module summarization | code -> doc generation | 216 | no | `relevant_code_files` | Apache-2.0 | no (not QA) |
| 22 | SWE-rebench / V2 / V2-PRs (nebius) | issue-to-patch | 21,336 (6,542 filtered) / 32,079 / 126,300 | yes (`base_commit`) | patch diff | CC-BY-4.0 | reducible |
| 23 | SWE-Gym / SWE-Gym-Raw | issue-to-patch | 2,438 / 64,689 | yes (`base_commit`) | patch diff | MIT / none | reducible |
| 24 | SWE-smith | synthetic bug-to-patch | 59,136 (py 50,908) | partial (short SHA in mirror repo name) | patch diff | MIT | reducible |
| 25 | R2E-Gym-V1 | issue-to-patch | 8,101 | yes (`commit_hash`) | `modified_files`, `relevant_files` | Apache-2.0 | reducible |
| 26 | Multi-SWE-RL / Multi-SWE-RL-Verified / Multi-SWE-bench | issue-to-patch, 7 langs | 4,723 (paper) / 2,232 / 1,632 (paper) | yes (`base` dict) | patch diff | CC0 (card) / other | reducible |
| 27 | SWE-bench Multilingual | issue-to-patch | 300 | yes (`base_commit`) | patch diff | MIT | reducible |
| 28 | OpenHands/SWE-rebench-code-search | issue -> edited entities | 17,591 | yes (`base_commit`) | `file_changes` entities | none stated | reducible to localization QA |
| 29 | bigcode/the-stack-github-issues | raw issues | gated; access denied with available token; 127 parquet shards | no | no | Stack ToU | unverified |
| 30 | open-index/open-github-issues | raw issues + PRs | issues 1,224,993; comments 6,907,417; PRs 717,815 | no | `pr_files` config | ODC-BY | reducible |
| 31 | blasi03/github-issues-qa-dataset | issue -> LLM CoT answer | 1,200 | no | no | none | no |
| 32 | InfiBench | SO-style free-form QA | 234 | n/a | n/a | CC-BY-SA-4.0 | function-level |
| 33 | ProCQA (jordane95 subset) | SO Q/A pairs | 4,612 (subset) | n/a | n/a | Apache-2.0 | SO-style |
| 34 | CoSQA+ | code search | rows API 500; card only | n/a | n/a | CC-BY-4.0 | function-level |
| 35 | CodeSearchNet | doc-code pairs | 1,880,853 train / 100,529 test | no | function | other | function-level |
| 36 | CodeQA 2021 + CS1QA (datapaf/CodeQuestionAnswering) | function-level QA | 63,776 / 8,228 / 8,228 | no | function inline | none | function-level |
| 37 | PrimeIntellect/synthetic-code-understanding | output prediction | 60,621 | n/a | n/a | none | function-level |

---

## 1. SWE-QA (SWE-QA-Bench) — ACL 2026 Findings

- Links: https://github.com/peng-weihan/SWE-QA-Bench ; https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark ; mirror with commit column https://huggingface.co/datasets/Cipherxzc/SWE-QA-Bench ; paper https://arxiv.org/abs/2509.14635
- Verified by: HF `/size` and `/rows`; GitHub README, `repo_commit.txt`, per-repo JSONL line counts.
- Count: 720. HF `default` split = 720; 15 per-repo splits x 48. GitHub `Benchmark/*.jsonl` = 15 files x 48 lines.
- Sample (HF, matplotlib): Q "What is the synchronization mechanism of the _expect method that integrates with the broader LatexManager architecture..." A "The synchronization mechanism uses character-by-character blocking reads ... (from `backend_pgf.py` lines 240-252) ..."
- Fields (official HF): `question`, `answer` only. Mirror `Cipherxzc/SWE-QA-Bench`: `instance_id`, `repo`, `repo_name`, `base_commit` (short, e.g. `0a041d3`), `problem_statement`, `answer`.
- Repo + commit pinned: yes, via `repo_commit.txt` (15 URLs + short SHAs, e.g. `https://github.com/astropy/astropy 0a041d3`) and `clone_repos.sh`.
- Evidence spans: no structured field; answers embed file paths and line numbers as prose.
- License: Apache-2.0 (GitHub LICENSE and HF card).
- Classification: natural repo QA (LLM-generated from issues, human-validated), free text.
- Usable for training: yes technically, but it is the field's main eval set; training on it burns the benchmark.

## 2. SWE-QA-Pro Bench (TIGER-Lab)

- Links: https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-Bench ; https://github.com/TIGER-AI-Lab/SWE-QA-Pro ; paper https://arxiv.org/abs/2603.16124
- Verified by: HF `/size`, `/rows`, card README.
- Count: 260 (`test` split). Card: "260 high-quality QA pairs from 26 repositories (10 per repository)".
- Sample: repo `qiboteam/qibo`, commit `2f98679c4a738d5a59de17eccd4712659d52a461`, qa_type "Where (Localization Queries)", Q "Where does measurement data flow from quantum state tensors through probability calculations..." A "...The CircuitResult class (src/qibo/states.py: lines 6-247) orchestrates..."
- Fields: `repo`, `commit_id`, `cluster{id,name}`, `qa_type{class_name,sub_class_name}`, `question`, `answer`.
- Repo + commit pinned: yes (full SHA per row).
- Evidence spans: no structured field; file:line refs in prose.
- License: MIT.
- Classification: natural repo QA, free text, seeded from real issues, human-verified.
- Usable: yes (eval set).

## 3. SWE-QA-Pro RL training data (GitHub parquet)

- Links: https://github.com/TIGER-AI-Lab/SWE-QA-Pro/tree/main/train/RL/verl-tool/dataset (`train.parquet` 1.0 MB, `test.parquet` 158 KB)
- Verified by: downloaded both parquets and read with pyarrow.
- Count: train 464 rows over 464 distinct repos; test 48 rows over 23 repos.
- Sample: `extra_info` = {repo_name `osbuild/osbuild`, commit_id `144b0563d63034c086ed375a821b1831cd04b5e7`, question "Where does Azure cloud-init configuration data flow through the validation and processing pipeline...", answer "...schema validation defined inline in the cloud-init stage module (stages/org.osbuild.cloud-init: line 25-142)..."}; `reward_model.ground_truth` = same answer; `prompt` = system+user turns for the verl-tool agent.
- Fields: `session_id`, `data_source`, `prompt`, `use_tool`, `ability`, `reward_model{ground_truth}`, `extra_info{answer, commit_id, index, question, repo_name, repo_path, split}`; test adds `cluster`, `qa_type`.
- Repo + commit pinned: yes, 464/464 non-empty `commit_id`.
- Evidence spans: none structured; 397/464 answers mention a `.py` path.
- License: MIT (repo).
- Classification: synthetic repo QA (LLM-generated, "grounded supervision"), free text, long-tail repos.
- Usable: yes, this is literally RL training data for the same task.

## 4. SWE-QA-Pro SFT Trajectories

- Link: https://huggingface.co/datasets/TIGER-Lab/SWE-QA-Pro-SFT-Trajectories (harmonized mirror: Biomechanist/SWE-QA-Pro-SFT-Trajectories-harmonized, rows API 500)
- Verified by: HF `/size`, `/rows` (3 rows parsed), card.
- Count: 1,000 (`train`).
- Sample user turn: "Repository Path: repos_tmp/worker_646683/python-neo Question: Why does the AnalogSignal class's times property computation become a performance bottleneck..." Final assistant turn holds `<finish>...</finish>` answer with file/line refs. 27 to 93 messages per trajectory; roles system/user/assistant/tool_call/tool_response; tools view_codebase, semantic_search, execute_readonly_command. Generated by Claude Sonnet 4.5.
- Fields: `tools`, `messages`.
- Repo + commit pinned: partial (repo folder name only; no SHA in row).
- Evidence spans: no.
- License: MIT.
- Classification: agent trajectories for repo QA (synthetic questions).
- Usable: partial. Questions and answers are extractable, but commits must be recovered and the questions likely overlap with #3.

## 5. Code-QA-Bench (Lens-Frontier)

- Links: https://github.com/Lens-Frontier/code-qa-bench ; paper https://arxiv.org/abs/2605.29277
- Verified by: `gh api` metadata, README, downloaded `tasks/tasks.json` (528) and `tasks/tasks_doc_dependent.json` (100), `repos.json`.
- Count: 628 (528 code-derivable + 100 doc-dependent) over 10 SWE-bench Python repos.
- Sample: id `django_gen_01`, Q "What is the full class hierarchy and architectural design behind how the Polygon geometry class supports list-like ring mutation through GEOS C bindings?", gold_answer with class hierarchy, `key_files` [6 django/contrib/gis/geos paths], rubric list, `source_doc`, verification metadata.
- Fields: `id, repo, question, category, sub_type, gold_answer, rubric, key_files, source_doc, verification_verdict, verification_issues, strip_verify_leakage, strip_verify_summary, generation_status, generation_error`.
- Repo + commit pinned: yes, `repos.json` maps each repo key to URL + full SHA `ref`.
- Evidence spans: file-level (`key_files`); no line ranges.
- License: MIT.
- Classification: natural-style repo QA, LLM-agent-generated from doc chunks, free text + rubric.
- Usable: yes (rubrics make it reward-friendly); eval-set caveat.

## 6. SWE-Bench Pro Interactive Issue+QA (Anonym01048)

- Link: https://huggingface.co/datasets/Anonym01048/SWE-Bench-Pro-interactive-issue-qa
- Verified by: HF `/size`, `/rows`, card.
- Count: 9 configs; unique questions per card table: Sol pool 467 (configs qa-first/last/random/only-sol, 255 instances each), Terra pool 114 to 118 (64 to 67 instances), main `qa-tfas-optimized-sol` 701 rows / 454 questions. Unique = 467 + 118 = 585.
- Sample: instance `instance_qutebrowser__qutebrowser-46e6839e...-v2ef375ac...`, Q "How does a canonical qute://version/ request become the returned HTML response...", golden answer + `golden_proof_scripts` (bash proof run in a clean repo copy) + `facts`.
- Fields: `instance_id, problem_statement, subset, phase_order_config, stable_question_ids, record_sha256, bundle_sha256, hybrid, questions, golden_proof_scripts, golden_proof_executions, facts, documentation, bundle_metadata`.
- Repo + commit pinned: yes (SWE-Bench Pro instance ids carry base and target SHAs).
- Evidence spans: executable proof scripts and mined facts; no line spans.
- License: "other" (derived from SWE-Bench Pro; check upstream terms).
- Classification: natural-style repo QA with verifiable proofs, free text.
- Usable: partial (license unclear, anonymous release).

## 7. swe-qa-episodes (aviralku)

- Link: https://huggingface.co/datasets/aviralku/swe-qa-episodes (single 670 MB JSONL)
- Verified by: streamed the whole file and counted.
- Count: 330 records (one per repo), 564 QA pairs total, 330 distinct `passage_id` repos (long-tail Python: `15five/scim2-filter-parser`, `Krukov/cashews`, ...).
- Sample: passage_id `PennyLaneAI/pennylane`; `inner_docs` = 128 `<file path=...>` blocks; `qas` [{question "Where do measurement results flow through the batch execution pipeline...", answer "...QNode.call in pennylane/qnode.py ... lines 963 to 972..."}] (10 QAs for this repo).
- Fields: `passage_id, passage, inner_docs, qas[{question, answer}]`.
- Repo + commit pinned: no.
- Evidence spans: no (prose).
- License: none stated.
- Classification: synthetic repo QA in SWE-QA-Pro style, free text, with repo snapshot inlined.
- Usable: no as-is (no license, no commit); content quality looks similar to #3.

## 8. StackRepoQA (FSE/PROMISE 2026)

- Links: https://github.com/code-world-no-blanket/StackRepoQA (data/questions.csv) ; paper https://arxiv.org/abs/2603.26567
- Verified by: downloaded `questions.csv` and parsed.
- Count: 1,318 rows (data README says 1,328; CSV has 1,318), 134 distinct `Repo Full Name`, all Java.
- Sample: repo `apache/hadoop`, SO title "(Hadoop): reduce method is not getting executed/called while running mapreduce job", SO body, accepted answer HTML, `Question Related? = Yes`.
- Fields: 30 columns: repo metadata (name, URL, license, stars, commit count...), `SO Question Id/Title/Body/Link/Date/Tags`, `SO Accepted Answer ID/Body`, `Question Related?`, author ids.
- Repo + commit pinned: no (repo URL only; no SHA).
- Evidence spans: no.
- License: no LICENSE file; paper is CC-BY-4.0; SO content is CC-BY-SA.
- Classification: natural developer questions, but answers are SO posts, not repository-grounded.
- Usable: no as-is; could be re-grounded with effort.

## 9. CoReQA (arXiv 2501.03447)

- Verified by: HF hub search "coreqa" (0 hits), `gh search repos CoReQA` (no relevant repo), full-text extraction of the arXiv PDF (only URLs are references to Copilot, The Stack, Zenodo for another tool).
- Paper claims: 1,563 QA pairs from 190 repositories (abstract says 176) across four languages, built from issues with 3+ positive comments; answers derived from issue comments.
- Count: **unverified, no public release found**.
- Commit / spans / license: unverified.
- Usable: no.

## 10. CodeRepoQA (arXiv 2412.14764)

- Links: https://github.com/kinesiatricssxilm14/CodeRepoQA (README only) ; data on Google Drive folder `19-7gqlcYuwbbHAqYyzMMov7tuTTwHfcY` (30 zips, one per repo).
- Verified by: README table; Drive folder listing (30 zip names/ids, e.g. aiohttp.zip 11.8 MB, angular.zip 129 MB, ansible.zip 175 MB); downloaded `py-tree-sitter.zip` (449 KB) and inspected 156 files.
- Count: README total 585,687 issues (e.g. vscode 148,293; home-assistant/core 50,540; kubernetes 44,567; pytorch 42,408). py-tree-sitter.zip: 155 issue JSONs matches README's 155.
- Sample record (`QA_data/py-tree-sitter/115.json`): standard GitHub issue JSON (`title` "Help: 'tree.edit(...)' syntax highlighting", `body`, `html_url`, `state`, `comments`) plus `comments_details[]` (author_association, body), `issue_or_pr`, `cite`, `cited_by`, `fixed_by`, `duplicate`.
- Repo + commit pinned: no (repo implied by folder; no SHA anywhere in the record).
- Evidence spans: no.
- License: none stated.
- Classification: raw multi-turn issue dialogue (avg 6.62 turns per paper), not QA; answers are whatever maintainers replied.
- Usable: no as-is. Reducible raw material if you filter question-like issues and re-ground answers against a commit near `created_at`.

## 11. SWE-QA LREC-2026 MCQ (lailaelkoussy/swe-qa)

- Links: https://huggingface.co/datasets/lailaelkoussy/swe-qa ; https://github.com/lailanelkoussy/swe-qa (datasets/mcq_dataset.zip, datasets/repos/*.zip)
- Verified by: HF `/size`, `/rows`, card; GitHub tree.
- Count: 9,072 in `oracle` split and 9,072 in `noisy_oracle` (same questions, more distractor chunks). 12 Python repos from SWE-bench.
- Sample: Q "Are DurationField and SmallIntegerField designed to be used together in a model...?" options A-D, `correct_answer` D, `repo` django/django, `category` interacting_entities, `chunks[{file_path, content, declared_entities, called_entities}]`.
- Fields: `question, options, code, correct_answer, repo, category, chunks, entities`.
- Repo + commit pinned: no SHA; repo snapshots are zipped in the GitHub repo.
- Evidence spans: chunk `file_path` (file-level).
- License: Apache-2.0.
- Classification: templated MCQ (two generation categories: entity declaration/call, interacting entities).
- Usable: weak. MCQ and templated; a large count but low question naturalness.

## 12. LongCodeQA (LongCodeBench)

- Links: https://huggingface.co/datasets/Steefano/LCB (LongCodeQA.zip) ; https://huggingface.co/datasets/agu18dec/LongCodeQA (443 rows, texts only) ; https://huggingface.co/datasets/SophieWu/LongCodeBench_split (3,723 rows, one per question x context-length prompt) ; paper https://arxiv.org/abs/2505.07897
- Verified by: downloaded LongCodeQA.zip (138 MB) and parsed all six JSON files.
- Count: 443 unique MCQs: 32K 113, 64K 76, 128K 92, 256K 65, 512K 47, 1M 50.
- Sample: repo `pallets/markupsafe`, Q "What is the expected behavior of the `html.escape` function when applied to a MarkupSafe `Markup` object in MarkupSafe version 3.0.0? A)... D)", `correct_letter` D.
- Fields: `prompt, repo, question, correct_letter, repo_text, prompt_goal, is_hard`.
- Repo + commit pinned: no (`repo` name only; `repo_text` is the snapshot).
- Evidence spans: no.
- License: Apache-2.0.
- Classification: MCQ derived from GitHub issues.
- Usable: weak (MCQ, tiny).

## 13. LoCoBench (Salesforce, arXiv 2509.09614)

- Links: https://huggingface.co/datasets/jasonqiu/LoCoBench (data.zip 250 MB) ; https://github.com/SalesforceAIResearch/LoCoBench
- Verified by: downloaded data.zip, listed 181,578 files, counted scenario JSONs, parsed two scenarios; read GitHub README and `LoCoBench_generation.md`.
- Count: exactly 8,000 scenario JSONs, 1,000 per category (architectural_understanding, bug_investigation, code_comprehension, cross_file_refactoring, feature_implementation, integration_testing, multi_session_development, security_analysis); 1,000 generated codebases across 10 languages.
- Codebases are synthetic: generation guide Phase 1 "Project Specification Generation", Phase 2 "Codebase Generation" (LLM), Phase 3 scenario creation, Phase 4 validation. No real GitHub repos.
- Sample (`typescript_api_microservice_easy_080_code_comprehension_medium_01`): `task_prompt` "Provide a step-by-step written analysis of the entire request lifecycle for creating a new invoice via the `POST /invoices` endpoint...", `ground_truth` free-text expected analysis, `evaluation_criteria` list, `context_files` 6 paths, `context_length` 45,711.
- Fields: `id, task_category, difficulty, title, description, context_files, context_length, task_prompt, expected_approach, ground_truth, evaluation_criteria, metadata`.
- Repo + commit pinned: n/a (synthetic codebases shipped in the zip, so reproducible).
- Evidence spans: file-level (`context_files`).
- License: Apache-2.0.
- Classification: synthetic codebases; the code_comprehension and architectural_understanding categories (2,000) are QA-shaped free-text tasks with rubric-style ground truth; others are implementation/refactoring tasks.
- Usable: partial. Large and rubric-graded, but not real repositories; distribution shift risk.

## 14. RepoQA (evalplus)

- Link: https://github.com/evalplus/repoqa ; paper https://arxiv.org/abs/2406.06025
- Verified by: README.
- Count: 500 (5 languages x 10 repos x 10 needle functions).
- Format: NL description of a function -> model must retrieve the function from a long code context; graded by syntactic similarity.
- License: Apache-2.0.
- Classification: needle-function retrieval, not QA.

## 15. CodeQueries (thepurpleowl/codequeries)

- Link: https://huggingface.co/datasets/thepurpleowl/codequeries
- Verified by: HF `/size`, `/rows`.
- Count: `ideal` config train 102,962 / validation 11,183 / test 57,201; `file_ideal` test 44,421; `prefix` test 44,423.
- Sample: `query_name` "Unused import", `code_file_path` `rcbops/glance-buildpackage/glance/tests/unit/test_db.py`, `answer_spans` [{span, start_line 19, ...}], `supporting_fact_spans`, `single_hop`.
- Repo + commit pinned: no (repo path prefix only).
- Evidence spans: yes (line/column).
- License: Apache-2.0.
- Classification: templated (52 CodeQL query names as "questions"), file-level.
- Usable: no for natural QA.

## 16. RLM-Bench (mohit-awana/rlm-bench)

- Link: https://github.com/mohit-awana/rlm-bench
- Verified by: README.
- Count: 20 hand-written cross-file questions on `httpx` 0.28.1; questions in `benchmark/questions.py`. MIT. Trivial size.

## 17. MultiRepoQA (NLPForUA/multilingual-repo-qa)

- Link: https://huggingface.co/datasets/NLPForUA/multilingual-repo-qa
- Verified by: HF API metadata and card summary only (gated=manual; rows/size return 401 even with the cached token; card says "Release in progress").
- Card claims: 783 validated canonical questions across 30 repositories, aligned EN/DE/UK (2,349 examples); configs `questions`, `repositories`, `simal_schemas`, model-run outputs.
- License: CC-BY-4.0.
- Count/commit/spans: **unverified**.

## 18. DeepRepoQA (peng-weihan/DeepRepoQA)

- Link: https://github.com/peng-weihan/DeepRepoQA ; paper https://arxiv.org/abs/2608.24221
- Verified by: tree and line counts of `dataset/questions/*.jsonl`.
- Count: 12 files = 575 (astropy 47, others 48) + `live/` conan 47, reflex 48, streamlink 48 = 718. Fields `question, answer, relative_code_list, ground_truth, score`. These are the SWE-QA questions (first astropy question identical to #1). Apache-2.0. Not new data.

## 19. vm2825 synthetic multi-file QA

- Links: https://huggingface.co/datasets/vm2825/all_1000_multifile_generated_qa_pairs_code_qa-dataset (5,088 rows) ; vm2825/small_repos_multi_file_chatgpt_5_qas_code_qa_1k-dataset (4,995 rows)
- Verified by: HF `/size`, `/rows`.
- Sample: `code` = several files inlined with `<start_file_name0>/crawl4ai/async_webcrawler.py...` (262 KB), Q "When AsyncWebCrawler.arun() is called for an HTTP/HTTPS URL, which objects across files are involved...", A short prose.
- Fields: `code, question, answer`. No repo URL, no commit, no license, no card.
- Classification: synthetic multi-file QA with code inlined; not repo-level in the required sense.

## 20. Long Code Arena bug localization (JetBrains-Research/lca-bug-localization)

- Link: https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization
- Verified by: HF `/size`, `/rows`.
- Count: java train 2,472 / dev 2,522 / test 50; kt 568 / 618 / 50; py 4,289 / 4,339 / 50. dev = train + test, so unique = 2,522 + 618 + 4,339 = 7,479.
- Sample: `square/okhttp`, issue "SpdyConnection.pushExecutor has zero keep-alive time", `base_sha`, `head_sha`, `diff`, `changed_files` ['...Connection.java'], repo stats.
- Fields: 41 incl. `repo_owner, repo_name, issue_url, pull_url, issue_title, issue_body, base_sha, head_sha, diff, changed_files, changed_files_count, repo_license`.
- Repo + commit pinned: yes.
- Evidence: changed files + diff.
- License: Apache-2.0 (dataset); per-repo licenses recorded.
- Classification: issue -> file localization; reducible to "which files implement/are affected by X" QA.

## 21. Long Code Arena module summarization

- Link: https://huggingface.co/datasets/JetBrains-Research/lca-module-summarization
- Count: 216 (`test`). Fields `repo, docfile_name, doc_type, intent, license, path_to_docfile, relevant_code_files, relevant_code_dir, target_text, relevant_code_context`. No commit. Apache-2.0. Doc generation, not QA.

## 22. Issue-to-patch raw material (NOT Q&A; reducible)

All verified via HF `/size` and `/rows` unless noted.

- **nebius/SWE-rebench**: test 21,336, filtered 6,542; Python; CC-BY-4.0; `base_commit` yes; fields incl. `problem_statement, patch, test_patch, FAIL_TO_PASS, install_config, license`.
- **nebius/SWE-rebench-V2**: 32,079; multi-language (`language` field, e.g. ts, python); CC-BY-4.0; `base_commit` yes; also `pr_description`, `interface`, per-instance `license`.
- **nebius/SWE-rebench-V2-PRs**: 126,300; CC-BY-4.0; `base_commit` yes (PR-derived, less test validation).
- **nebius/SWE-bench-extra**: 6,376; CC-BY-4.0; `base_commit` yes.
- **SWE-Gym/SWE-Gym**: 2,438; MIT; `base_commit` yes. **SWE-Gym-Raw**: 64,689; no license field; `base_commit` yes.
- **SWE-bench/SWE-smith**: 59,136 (SWE-smith-py 50,908; java/js/ts/cpp/go/rs/php variants exist); MIT; repo is a mirror `swesmith/oauthlib__oauthlib.1fd52536` (short SHA in name); synthetic bugs, so `problem_statement` is LLM-written.
- **R2E-Gym/R2E-Gym-V1**: 8,101; Apache-2.0; `commit_hash` yes; `modified_files`, `relevant_files`, `modified_entity_summaries`.
- **ByteDance-Seed/Multi-SWE-RL**: rows API 500; paper abstract (2504.02605) says 4,723 instances, 7 languages; card says CC0. **PrimeIntellect/Multi-SWE-RL-Verified**: 2,232 rows verified (`org, repo, number, base, fix_patch, test_patch, f2p_tests...`). **Multi-SWE-bench**: 1,632 (card + paper).
- **SWE-bench/SWE-bench_Multilingual**: 300; MIT; `base_commit` yes.
- **OpenHands/SWE-rebench-code-search**: 17,591; no license stated; `base_commit` yes; `file_changes` lists edited/added entities per file (ready-made localization labels).

## 23. Raw GitHub issue corpora

- **bigcode/the-stack-github-issues**: gated (auto-approval ToU); the cached local token is not on the authorized list, so `/size`, `/rows` and README return 401/403. Verified only: 127 parquet shards in `data/`, card task = language-modeling. Size and fields **unverified**.
- **open-index/open-github-issues**: ODC-BY; configs and rows: issues 1,224,993; comments 6,907,417; pull_requests 717,815; pr_files 5,304,006; review_comments 1,754,513; reviews 1,242,158; timeline_events 5,082,097; commit_statuses 799,067. Issue fields `owner, repo, id, issue_number, author, body, created_at, updated_at, reactions, author_association` (title not in the parsed feature list shown; check). No commit per issue, but `pr_files` + `pull_requests` allow linking issues to fixing PRs.
- **blasi03/github-issues-qa-dataset**: 1,200; `id, url, cot_answer, instruction` (issue URL + LLM CoT answer); no license.
- **acampi23/github-issue-pr-resolution-10k**: rows not fetched (cut off); unverified.
- **lewtun/github-issues** and similar HF course datasets: small single-repo dumps; not inspected further.

## 24. Function-level / StackOverflow-style (for classification only)

- **InfiBench** (llylly001/InfiBench): 234 rows in `suite_v2.1_data.csv`; CC-BY-SA-4.0; free-form SO-style questions; not repo-level.
- **ProCQA** (jordane95/procqa): 4,612 rows in this HF subset (`question_id, answer_id, title, question, answer`), SO Q/A; Apache-2.0. Full ProCQA is larger (unverified here).
- **CoSQA / CoSQA+**: code search (query -> code); CoSQA_Plus rows API 500; CC-BY-4.0. Function-level.
- **CodeSearchNet**: 1,880,853 train / 100,529 test (config `all`); doc-code pairs; license "other".
- **CodeQA (2021) + CS1QA** as re-hosted in datapaf/CodeQuestionAnswering: 63,776 / 8,228 / 8,228 with `src` (cs1qa seen); function inline; no license. vm2825/CodeQA-dataset: 76,840 templated "What does the code make?" rows.
- **PrimeIntellect/synthetic-code-understanding**: 60,621 output-prediction tasks; not QA.

---

## Bottom line

Verified natural, free-text, repository-level Q&A with a pinned commit:

| Source | Count |
|---|---|
| SWE-QA (v1+v2) | 720 |
| SWE-QA-Pro Bench | 260 |
| SWE-QA-Pro RL train + test | 512 |
| Code-QA-Bench | 628 |
| SWE-Bench-Pro Interactive Issue+QA (license "other") | 585 |
| **Total** | **2,705** |

Adding partially-pinned or unlicensed free-text sets (SWE-QA-Pro SFT trajectories 1,000, swe-qa-episodes 564) reaches 4,269, still under 5,000, and #3/#4/#7 likely overlap in origin. Adding MCQ (LREC SWE-QA 9,072; LongCodeQA 443) or synthetic-codebase tasks (LoCoBench 2,000 QA-shaped) crosses 5,000 only by accepting templated multiple choice or non-real code.

Not released or not verifiable: CoReQA (1,563 claimed, no release found), MultiRepoQA (783 claimed, gated, "release in progress").

Largest reducible raw sources with pinned commits: SWE-rebench-V2-PRs 126,300; SWE-smith 59,136; SWE-Gym-Raw 64,689; SWE-rebench-V2 32,079; SWE-rebench 21,336; OpenHands/SWE-rebench-code-search 17,591 (issue -> edited entities); R2E-Gym-V1 8,101; Long Code Arena bug localization 7,479 (issue -> changed files). Without commits: CodeRepoQA 585,687 issue threads (30 repos, Google Drive); open-github-issues 1.22M issues + 5.3M PR files.

---

## Addendum (resumed run, 2026-09-18)

- **open-index/open-github-issues `issues` config, fields verified via `/rows`:** `owner, repo, number, node_id, is_pull_request, title, body, state, state_reason, author, created_at, updated_at, closed_at, labels, assignees, milestone_title, milestone_number, reactions, comment_count, locked, lock_reason, detail_fetched_at`. Card: full development metadata of **7 public repositories** only (ClickHouse etc.), 6.0M rows, 699.6 MB, ODC-BY. So it is deep but narrow: not a broad multi-repo issue corpus. No commit per issue; `pr_files` config links PR file changes.
- **RepoSearch-R1 / RepoQA-Agent (arXiv 2510.26287):** WebSearch found the paper only; no dataset or code release located. Unverified/unreleased.
- **Additional HF hub searches** ("SWE-QA-Agent", "repoqa rl", "code agent qa trajectories", "repository question", "codebase understanding", "repo understanding", "code-repo-qa", "RepoQA-Agent", "repo-level qa"): zero new repository-level QA datasets. The only hits were SousiOmine/codeqa-agent-distill-* (Japanese pi-coding-agent trajectories, 631 rows in the 260709 set, Apache-2.0, questions about a single repo, no commit field in metadata sample).
