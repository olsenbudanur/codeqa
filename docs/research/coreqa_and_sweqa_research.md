# CoReQA usability, and how CoReQA / SWE-QA-Bench built their Q&A

Date: 2026-09-18. Sources: arXiv 2501.03447 (PDF text extracted locally), arXiv 2510.26287 (RepoSearch-R1), github.com/LingmaTongyi/RepoSearch-R1 (now "RepoNav"), github.com/peng-weihan/SWE-QA-Bench (raw `repo_commit.txt` and `Benchmark/flask.jsonl`), arXiv 2509.14635 HTML, HF dataset cards, GitHub/HF search APIs.

## 0. Verdict on CoReQA

**No, CoReQA is not usable for training with exact-commit snapshots.**

1. **No public release exists.** The paper has no data-availability statement and no GitHub/HF/Zenodo link for the benchmark (the only URLs in the PDF are references: The Stack, LangChain, Copilot, tokenizer, etc.). The GitHub search API for "CoReQA" returns only unrelated repos (`techthiyanes/CoReQA` is a 2023 "contextual retrieval" project, not this dataset). The HF datasets API search for "coreqa" returns `[]`. The paper is at v1 only (2025-01-07), authors Chen, Zhao, Liu, Peng, Liu, Zhu, Gao, Yang, Deng (ByteDance / Zhejiang University affiliations per the PDF header).
2. **Records do not pin a commit.** Each record stores issue URL, title, description, repository name, comments with reaction tags, the rewritten question (with the issue's original code snippets attached), the generated answer, and 10 BM25-retrieved chunks. The only temporal signal is the issue creation date (used for a timeline analysis). No commit hash, tag, or version.
3. **Answers are not grounded in the code.** Reference answers are LLM-generated from the issue thread (question + issue tag + title + description + comments). Answers "may only include natural language response, code snippets, or the mixed format"; file paths and code spans are not guaranteed. Repository chunks are attached only as retrieval context for evaluating models, not for constructing answers.
4. **The RepoSearch-R1 500/160/170 split is not released.** RepoSearch-R1 (2510.26287, v1 2025-10-30) says it filtered the 1,563 pairs to 830 "high-quality" pairs then, via curriculum filtering, to 500 train / 160 validation / 170 evaluation; it does not say how repo snapshots were obtained. The GitHub URL it cites (`github.com/LingmaTongyi/RepoSearch-R1`) now hosts "RepoNav" (Apache-2.0, veRL-based) whose README states: "Datasets, repository snapshots, checkpoints, merged models, trajectories, logs, analysis results, and paper artifacts are not distributed in this repository." Its launchers expect `data/swe_qa/split/{train,test}.parquet`, i.e. SWE-QA, not CoReQA. The only data-prep script is `scripts/reponav/data_preparation/decompose_claims.py` (claim-decomposition rewards). Tools live in `verl/tools/repoqa_tool.py`; config in `examples/repoqa/config/tool_config/repoqa_tool_config.yaml`; training scripts `scripts/reponav/training/swe_qa_mcts_agent_iter7_minimal{,_4b}.sh` for Qwen3-8B / Qwen3-4B.
5. **Overlap with SWE-QA-Bench repos cannot be checked**: the 176 repositories are never named (only "top 500 by stars" per language, then filtered). Given 46 Python repos drawn from the star-ranked top 500, overlap with django/flask/requests/pytest/sympy/matplotlib/scikit-learn/sphinx is likely but unverifiable.

Size and shape for the record: 1,563 QA pairs, 176 repos, four languages: Python 46 repos / 439 pairs, Java 33 / 209, Go 48 / 371, TypeScript 49 / 544; 44 questions flagged for long-context evaluation. License of the data itself: unstated (paper says repos were chosen "with permitted license" per The Stack's license list).

## 1. How CoReQA constructed its Q&A (exact pipeline)

Repository selection
- Start from the top 500 GitHub repos by stars for each of Python, Java, Go, TypeScript.
- Drop repos whose closed issues + PRs count is < 1,000 or > 50,000. Result: 890 repos.
- Apply a license filter (permissive, per The Stack's `licenses.json`) and repo-size filter.
- Estimate repo token length as chars/4; per language keep 50 repos with > 200K tokens and 5 with < 200K. Result: 218 candidate repos with 1,623,624 issues.

Issue collection and filtering (all closed issues with comments, "IwC")
- Tag filter on "feat", "bug", "fix", "error" (the paper text says it filters out issues that do not carry these tags, then notes these tags usually mark fix requests without direct solutions in comments; the direction is ambiguous in the text).
- Exclude issues/comments containing images.
- Keep issues whose description contains code snippets.
- Require at least three "positive" comment reactions (+1, laugh, hooray, heart, rocket).
- Exclude issues whose positive comments link to other issues or promise a commit/fix.
- Result: 8,977 issues from 213 repos; cap 300 issues per repo.
- LLM pre-filter with a bespoke prompt for "suitability as a QA pair". Result: 2,127 issues from 190 repos.

Question generation
- An LLM rewrites issue title + description into a repository-related question, using Chain-of-Thought and Self-Consistency-style output formatting so questions are extractable; the issue body's original code snippets are appended to the question. The specific generator model is not named (evaluation models were GPT-4o via Azure, DeepSeek-V2, Gemini-1.5).

Reference answer construction
- Prompt (Fig. 3) takes the question, issue tag, title, description and comments; role assignment + CoT; formatted output. Answers therefore come from the discussion thread, not from reading the repository.

Retrieval context
- BM25 over repository chunks; top-5 chunks for the issue snippet and top-5 for the question = 10 "reference content" chunks per pair, used in the RAG evaluation setting.

Human verification
- "Rigorous in-house human inspection"; "All annotators in CoReQA are authors of this work"; "manual sampling and verification on a subset of the annotated questions and answers." No counts, criteria, or removal rates are reported.

Evaluation
- LLM-as-judge (four scored aspects, 1-10, five levels): Accuracy (also "checks the accuracy of quoted sources"), Completeness, Relevance, Clarity; plus Pairwise Comparison Evaluation (PCE, Elo, run twice with swapped order). The abstract's "five aspects" = these four scores + PCE. Judge std ~0.55 over five runs.

Takeaway for us: the pipeline is issue-thread QA, exactly the style our SWE-rebench/SWE-Gym rows enable, but it never grounds answers in the code; our answer-first, citation-first generation is a different (better) design.

## 2. How SWE-QA-Bench built its 720 questions (2509.14635)

Repositories and pinned commits (from `repo_commit.txt`, Apache-2.0 repo, `clone_repos.sh` checks these out):
- astropy/astropy 0a041d3, matplotlib/matplotlib a5e1f60, scikit-learn/scikit-learn adb1ae7, sympy/sympy 3c817ed, pydata/xarray 40119bf, django/django 14fc2e9, pallets/flask 85c5d93, pylint-dev/pylint 44740e5, pytest-dev/pytest 5989efe, sphinx-doc/sphinx 6c9e320, sqlfluff/sqlfluff db9801b, psf/requests 46e939b, conan-io/conan 52f43d9, reflex-dev/reflex fe0f946, streamlink/streamlink ab1f365.
- 12 from SWE-bench + 3 from SWE-bench-Live (conan, reflex, streamlink). The earlier version had 576 questions over the 12 SWE-bench repos; DeepRepoQA (same first author) notes the expansion to 720.

Question pipeline
- Crawl 77,100 GitHub issues from the 12 SWE-bench repos; keep 41,955 with body >= 1,000 chars.
- LLM parses issues into 127,415 distinct questions (~3.04 per issue).
- Two annotators manually code 1,000 questions to derive a two-level taxonomy: Level 1 What 13.3% / Why 23.1% / Where 28.4% / How 35.2%; Level 2 (12 intentions): Architecture exploration, Concept/Definition, Dependency tracing, Design rationale, Purpose exploration, Performance, Data/Control-flow, Feature location, Identifier location, System design, Algorithm implementation, API/Framework support. GPT-5 classifies the remaining 126,415.
- Abstract seed templates are written per intention; tree-sitter parses each repo into a structure graph; templates are instantiated on a "compact subgraph around a focal element"; manual screening/refinement; final 720 = 48 per repo, balanced over Level-1 types.

Answer pipeline
- RAG pipeline (semantic similarity + dependency analysis) feeding "a powerful LLM", with Cursor used as a tool, assisted by 4 human experts (>= 3 years experience).
- Two experts independently answer each question; disagreements go to a third expert for consensus; ambiguous questions and ungrounded/incorrect answers are removed.
- Reference answers average 266.64 words; mean edit distance after human correction 42.17 words.
- Difficulty stats: avg 4.72 reasoning hops, 3.19 files, 8.71 functions per question; 90.9% multi-hop; 77.6% cross-file.

Released fields
- `Benchmark/<repo>.jsonl` rows contain only `question` and `answer`. Category labels and structural metadata from the paper are not in the release. However, the answers embed file paths (e.g. `src/flask/json/provider.py`), function names, and line references ("lines 144-148", "line 320"), so gold citations can be regex-extracted and validated against the pinned commit. HF mirror: `swe-qa/SWE-QA-Benchmark` (Apache-2.0; card shows 1,440 rows, i.e. duplicated splits).

Evaluation
- Judge: Claude Sonnet 4.5; five dimensions (Correctness, Completeness, Relevance, Clarity, Coherence) each 1-20, summed to 100; 5 votes per dimension; anonymized and shuffled. Human check: 3 non-author engineers, same rubric; "high agreement" reported. `score/llm-as-a-judge.py` in the repo.

## 3. Other repo-level QA datasets with pinned commits and file/line evidence (post mid-2025)

| Dataset | Size | Repos / lang | Commit pinned | File/line evidence | License | Notes / URL |
|---|---|---|---|---|---|---|
| **SWE-QA-Pro** (TIGER-Lab, ACL 2026, 2603.16124) | Bench 260 test Qs; SFT 1,000 trajectories; RL 464 Qs with Claude Code gold answers | 26 long-tail Python repos from SWE-rebench with executable envs (training questions span 1,484 repos) | Yes: `commit_id` column | Yes: answers require explicit file path + line references (e.g. `src/qibo/models/circuit.py: lines 1287-1332`) | MIT | HF `TIGER-Lab/SWE-QA-Pro-Bench` (columns repo, commit_id, cluster, qa_type {What/Where/Why/How + subclass}, question, answer) and `TIGER-Lab/SWE-QA-Pro-SFT-Trajectories` (hermes-format tool calls; tools `semantic_search`, `view_codebase`, `execute_readonly_command`). Pipeline: 1.69M issues from 3,468 repos, embed + k-means into 48 clusters, Claude Code proposes QA from 20 sampled issues per cluster, human edit, difficulty filter (drop questions solvable without tools). Judge: 5 dims x 1-10, 3 votes. Qwen3-8B SFT+GRPO (ms-swift + verl-tool) reaches 35.39 vs GPT-4o. https://github.com/TIGER-AI-Lab/SWE-QA-Pro |
| **DeepCodeBench** (Qodo) | 1,114 QA: 912 train / 232 test | 8 repos (card is transformers-centric) | Yes: commit in `metadata` | Yes: answers name files/functions; `facts` list (1-35 atomic facts) for fact-recall grading | Apache-2.0 | HF `Qodo/deep_code_bench`; generated from PR context (changed code + containing methods/classes/files + PR title/description) by LLM; difficulty easy/moderate/hard. Fact-recall grading is close to programmatic. https://www.qodo.ai/blog/deepcodebench-real-world-codebase-understanding-by-qa-benchmarking/ |
| **CodeScout localization data** (OpenHands, 2603.17829) | `OpenHands/SWE-rebench-code-search` 17,591 rows; also `SWE-smith-py-code-search` (39K instances / 128 repos per paper), `SWE-Gym-code-search`; eval sets `SWE-bench_{Verified,Lite,Pro}-locagent`; rollouts `CodeScout_Training_Rollouts`, `CodeScout_Eval_Rollouts` | Python | Yes: `base_commit` | Yes: `file_changes[]` with `edited_entities`, `edited_modules`, `added_entities`, `added_modules` derived from the gold patch | MIT (code); dataset license not stated on card | Fully programmatic ground truth (file/module/function F1 reward). Best source for verifiable "where is X / which function implements Y" tasks. Models CodeScout-1.7B/4B/14B (Qwen3), bash-only tools. https://github.com/OpenHands/codescout , https://huggingface.co/collections/OpenHands/codescout |
| **Code-QA-Bench** (2605.29277) | 528 code-derivable + 100 doc-dependent | 10 SWE-bench Python repos: django, pylint, sympy, scikit-learn, astropy, matplotlib, sphinx, pytest, xarray, seaborn | not stated | Yes: gold answers carry >= 3 code-evidence items with files and functions | CC BY 4.0 | Answer-first generation agent with `read_file`, `list_directory`, `search_code`; doc-dependent claims stripped in a Keep/Remove/Rewrite audit; judge 0-5 on accuracy/completeness/specificity. "Code and data are open-source" but no URL in the paper; not found on GitHub/HF. |
| **StackRepoQA** (FSE 2026, 2603.26567) | 1,318 Stack Overflow Q/A | 134 Java repos | No | No | CC BY 4.0 (Zenodo) | Developer questions matched to repos by keyword; judge 1-10. Style only. |
| **DeepRepoQA** (2608.24221) | no new dataset; adds 30 Java Qs (Strata, Fineract, Shiro) | | | | | Zenodo 10.5281/zenodo.21063159, https://github.com/peng-weihan/DeepRepoQA ; MCTS agent over tree-sitter + voyage-code-3. |
| **Deep Agentic Search study** (2608.01507) | none | | | | | Finding worth noting: a 3-tool ReAct agent (`get_repo_structure`, `search_rag`, `read_file`) scored 65.2% pass on SWE-QA vs 46.2% for a deep multi-agent harness, at less than half the cost. Supports our small-tool-set design. |

## 4. Recommendation

- Drop CoReQA. Use SWE-QA-Bench (720, pinned commits, answers with extractable file/line refs) and SWE-QA-Pro-Bench (260, `commit_id`, explicit file/line answers, What/Where/Why/How labels) as held-out evals.
- Use CodeScout's `SWE-rebench-code-search` / `SWE-Gym-code-search` (base_commit + gold files/modules/entities) as the backbone of verifiable "where" tasks, and DeepCodeBench's 912-row train split (commit + facts) for "what/how" tasks with fact-recall grading.
- Borrow SWE-QA-Pro's SFT trajectories (1,000, read-only tools, hermes tool-call format) as an optional warm-start before RL, after converting tool names to ours.
- Generate the remaining tasks answer-first (Code-QA-Bench / SWE-QA-Pro style), seeded from SWE-rebench issues, with the citation verifier as the primary reward.

---

# Part 2: SWE-QA-Pro recipe, DeepCodeBench, CodeScout (verified 2026-09-18)

Verification method: parquet files downloaded from HF and GitHub and inspected locally with pyarrow (`scratchpad/hf/`); source files fetched raw from `github.com/TIGER-AI-Lab/SWE-QA-Pro`; paper `arxiv.org/html/2603.16124v1`; HF dataset API for licenses.

## 5. SWE-QA-Pro (TIGER-AI-Lab, ACL 2026)

### 5.1 Tool set (exact, from `eval/sweqapro/tools/*.py` and the `tools` column of the SFT trajectories)

```python
def view_codebase(path: str, view_range: Optional[List[int]] = None, concise: bool = False, python_only: bool = False) -> str
# file: "{lineno:6d} {line}" numbered lines; view_range [start, end] 1-based, end=-1 for EOF;
# concise=True (py only) elides function bodies >= 3 lines as "... elided lines X-Y ...";
# directory: recursive listing depth 2, hidden paths excluded; MAX_RESPONSE_LEN = 6000 chars then truncation notice.

def semantic_search(term: str, path: str = ".", *, python_only: bool = False, max_files: int = 100, include_lines: bool = False) -> Dict
# plain substring match (`term in line`), NOT embeddings despite the name; returns
# {success, scope: "file"|"directory", term, root, files: [{path, count, matches?}], truncated, error};
# directory mode gives per-file match counts only; include_lines=True adds line numbers + text.

def execute_readonly_command(command: str) -> str
# ALLOWED first tokens: ls tree find basename dirname realpath pwd cat head tail less more grep egrep fgrep rg ag wc sort uniq cut awk sed file stat du df
# BLOCKED: git ipython jupyter nohup python python3 pip pip3 npm node yarn make cmake docker sudo su chmod chown rm rmdir mv cp mkdir touch echo printf tee dd vi vim nano emacs wget curl scp rsync kill killall pkill systemctl service export unset alias unalias cd
# rejects ">", ">>", "tee", "xargs", "exec"; training server TIMEOUT = 10 s; returns STDOUT/STDERR text.
```

JSON schemas in the trajectories (`tools` column) match these: `view_codebase{path (req), view_range[int,int], concise, python_only}`, `semantic_search{term (req), path, python_only, max_files, include_lines}`, `execute_readonly_command{command (req)}`. There is no explicit `finish` tool; the episode ends when the assistant emits a `<finish>...</finish>` block (the RL server ends the env on `is_last_step`). The system prompt (`eval/prompts/agent_system_prompt.txt`, identical in the SFT rows and the RL `prompt` column) mandates Planning -> Investigation -> Synthesis -> Finalization, "A. Reasoning / B. Final Answer", exactly one `<finish>` block, evidence cited as `path: line start-end`, no code blocks. Trajectory tool usage over 1,000 rows: `view_codebase` 14,422 calls, `execute_readonly_command` 4,952, `semantic_search` 3,589; mean 15.0 assistant turns (min 5, max 25) and 23.0 tool calls (6-57) per trajectory.

### 5.2 Trajectory format (`TIGER-Lab/SWE-QA-Pro-SFT-Trajectories`, 1,000 rows, 61 MB, MIT)

Columns: `tools` (list of OpenAI-style function specs; note `parameters.properties` is a JSON *string*, not an object) and `messages` (list of `{role, content}`), roles in order `system, user, assistant, tool_call, tool_response, ..., assistant`. `tool_call` content is a JSON string `{"name": "...", "arguments": {...}}`; `tool_response` content is a JSON string such as `{"command_output": "STDOUT:\n..."}`. This is ms-swift's "hermes" agent template. The user turn is `Repository Path: repos_tmp/worker_<id>/<repo-name>\nQuestion: ...` followed by a 5-step workflow instruction. Row 0: python-neo, "Why does the AnalogSignal class's times property computation become a performance bottleneck..."; final answer cites `neo/core/analogsignal.py, lines 154-156` etc.

Repo coverage: **999 distinct repo names across 1,000 rows** (one question per repo; only `python-sdk` appears twice). The rows carry the repo *folder name only*, no owner and **no commit id**, so the trajectories cannot be re-run against an exact snapshot without the authors' `repos_tmp` checkouts. Overlap with the 15 SWE-QA-Bench repo names: `matplotlib` and `streamlink` (2 rows). No sign of xarray/sphinx/sqlfluff in SFT.

### 5.3 RL data (`train/RL/verl-tool/dataset/{train,test}.parquet` in the GitHub repo)

- `train.parquet`: 464 rows, columns `session_id, data_source ("SongchengCai/swe-qa-pro"), prompt (system+user messages), use_tool, ability ("swe-qa"), reward_model{ground_truth, question, style}, extra_info{answer, commit_id, index, question, repo_name, repo_path, split}`. **464 distinct repos, every row has a 40-char `commit_id`.** Overlap with SWE-QA-Bench: `astropy/astropy`, `conan-io/conan` (1 row each). Ground-truth answers are Claude-written prose with `path: line a-b` citations.
- `test.parquet`: 48 rows (validation), adds `cluster{id,name}`, `qa_type{class_name, sub_class_name}`, `refined_ground_truth`, `annotated_answer`; 23 repos; overlap `pydata/xarray`, `sphinx-doc/sphinx`, `sqlfluff/sqlfluff`.
- `TIGER-Lab/SWE-QA-Pro-Bench` (260 rows, MIT): columns `repo, commit_id, cluster, qa_type, question, answer`; 26 repos x 10 questions: PennyLaneAI/pennylane, bridgecrewio/checkov, cekit/cekit, docker/docker-py, dwavesystems/dwave-cloud-client, ethereum/web3.py, fitbenchmarking/fitbenchmarking, frictionlessdata/frictionless-py, geopandas/geopandas, getsentry/responses, hgrecco/pint, hylang/hy, microsoft/pybryt, mkdocs/mkdocs, mwaskom/seaborn, numba/numba, omni-us/jsonargparse, pydata/xarray, python-pillow/Pillow, qiboteam/qibo, sanic-org/sanic, sphinx-doc/sphinx, sqlfluff/sqlfluff, stfc/PSyclone, tox-dev/tox, yt-dlp/yt-dlp. **Overlap with SWE-QA-Bench: xarray, sphinx, sqlfluff (30 of 260 questions)**; commits differ from SWE-QA-Bench's pins, so treat those 30 as near-duplicates when using both as evals. qa_type L1: Where 77, How 67, Why 65, What 51; 12 L2 sub-classes; 95.8% of answers contain a line reference.

### 5.4 RL reward (exact, from `verl_tool/workers/reward_manager/swe_qa_pro.py`, class `SWEQAProRewardManager`)

- Judge: OpenAI client, default `model_name="gpt-4o-2024-11-20"` (overridable by env `MODEL`, key/base_url from `.env`). The paper says the RL judge is "distinct from the evaluation judge" (eval used DeepSeek/OpenAI backends via `eval/sweqapro/scoring/judge.py`).
- Judge prompt: rate candidate vs reference on five 1-10 integers, output ONLY a JSON object with keys `correctness, completeness, relevance, clarity, reasoning` (eval prompt `eval/prompts/judge_prompt.txt` is the short version: "Correctness: accuracy of core points and details; Completeness: coverage of key points from reference; Relevance: focus on question topic; Clarity: fluency and precision; Reasoning: logic and argumentation quality").
- Reward: `weights = {"correctness": 0.3, "completeness": 0.2, "relevance": 0.2, "clarity": 0.1, "reasoning": 0.2}`; `reward = sum(scores[k]*weights[k]) / 10.0` (range 0.1-1.0).
- Format: `_extract_answer` requires a `<finish>...</finish>` block; if missing, `scores = None`.
- Failure fallback: samples whose judge call fails or lack `<finish>` get the **group-average per-dimension scores**; if the whole group failed, all dimensions = 1 (reward 0.1).
- **No** citation/file-path verification, no turn/efficiency penalty, no invalid-tool-call penalty, no length penalty. Citations are only enforced by the system prompt and the judge's "correctness of details".

### 5.5 Hyperparameters (from `train/SFT/scripts/train_sweqapro_8B.sh` and `train/RL/verl-tool/scripts/train_sweqapro_8B.sh`)

SFT (ms-swift): `Qwen/Qwen3-8B`, `train_type full` (no LoRA), lr 5e-6, 3 epochs in the script (paper Table 5 says 4), `max_length 32768`, per-device batch 1, grad-accum 2, weight decay 0.05, bf16, flash_attn, gradient checkpointing, DeepSpeed ZeRO-3, 8 GPUs, `agent_template hermes`, `loss_scale hermes` (assistant-only loss, tool responses masked), dataset `SFT/dataset/train.jsonl`. Teacher: paper says Claude Sonnet 4.5; repo README says "Claude Code". No filtering threshold is documented.

RL (verl-tool, GRPO): `Qwen/Qwen3-8B` path but the README says to point `model_name` at the SFT checkpoint; full fine-tuning, FSDP, 8 GPUs x 1 node, `reward_manager sweqapro`, tool server type `swe_qa_pro` (random port 30000-31000, clones `https://github.com/{repo_name}.git` per trajectory and checks out `commit_id`), actor lr 1e-6, `use_kl_loss True`, `kl_loss_coef 0.02`, `kl_loss_type low_var_kl`, entropy coef 0, `train_batch_size 8`, `ppo_mini_batch_size 8`, micro batch 1/GPU, `rollout.n 8`, temperature 1.0, `max_turns 25`, `action_stop_tokens </tool_call>`, `max_prompt_length 2048`, `max_response_length 8192`, `max_obs_length 28000`, `mask_observations True`, gpu_memory_utilization 0.8, `total_epochs 1` (464 questions / 8 = 58 optimizer steps), save/test every 10. Tool calls are parsed with regex `<tool_call>\s*(\{.*?\})\s*</tool_call>` and validated against the three names.

### 5.6 Results (paper Table 2, overall score = sum of five judge dims, max 50; per-dimension values omitted here because the HTML extraction mislabeled columns)

Qwen3-8B direct 26.61; Qwen3-8B + agent 30.03; SWE-QA-Pro-8B SFT direct 24.42; SFT + agent 34.34; **SFT + RL + agent 35.39**; GPT-4o + agent 33.08; Claude Sonnet 4.5 + agent 40.67. Ablation (Table 5): SFT on 1,464 vs 1,000 trajectories gives "consistent but modest improvements", whereas "SFT-1000 + RL-464 achieves substantially higher scores in both Correctness and Completeness than SFT-only variants". **No RL-only (GRPO from base) run is reported**, and no tool ablation.

Licenses: repo MIT; `SWE-QA-Pro-Bench` MIT; `SWE-QA-Pro-SFT-Trajectories` MIT (both confirmed via HF API `cardData.license`).

## 6. DeepCodeBench (`Qodo/deep_code_bench`, Apache-2.0)

- Splits: `train` 912 rows (572 KB parquet), `test` 232 rows (160 KB); 1,144 total.
- Columns: `question: str`, `answer: str`, `facts: list[str]`, `metadata: struct{commit: str, difficulty: str, found_stats: struct{path: int64}, includes_code: bool, includes_location_hints: bool, is_core_question: bool, n_context_files: int64, n_context_nodes: int64, n_files_pr: int64, pr: int64, repo: str, scope: str, type: str}`, `id: str` (UUID).
- Repo/commit storage: `metadata.repo` is a clone URL (`https://github.com/huggingface/transformers.git`), `metadata.commit` a 40-char SHA, `metadata.pr` the source PR number. **Exactly one commit per repo.**
- Repos (rows): fastai/fastai 209, getzep/graphiti 188, huggingface/diffusers 181, microsoft/qlib 160, huggingface/transformers 152, dmlc/xgboost 131, microsoft/LightGBM 89, keras-team/keras 34. **Overlap with SWE-QA-Bench: none.** Note xgboost/LightGBM/keras contain substantial C++ alongside Python.
- Distributions: difficulty moderate 938 / hard 159 / easy 47; scope deep 598 / broad 546; `includes_location_hints` True 542; `is_core_question` True 462; `includes_code` True 13; `type` always `open_question`. `facts` per row mean 6.58 (min 1, max 35).
- `facts` format: atomic declarative sentences naming files/functions, e.g. "The function pad_collate_fn.inner is defined in src/transformers/pipelines/base.py.", "If tokenizer is None and feature_extractor is not None, pad_collate_fn.inner uses the feature extractor's padding value (f_padding_value)."
- Grading procedure (Qodo blog; the README only lists fields): "fact recall" after the TREC-2003 QA track. Facts are pre-extracted from the gold answer; for each fact "a simple LLM call" checks whether the fact is present in the predicted answer; the score is the fraction recalled. Reported: Qodo Aware ~76-80%, Codex ~74%, Claude Code 64%, Gemini CLI 45%. No threshold or prompt is published; write a one-line "Is this fact stated or clearly implied by the answer? yes/no" judge per fact.
- Generation (blog): questions/answers generated by an LLM from PR context (changed code, enclosing methods/classes/files, PR title/description).

## 7. CodeScout code-search sets (OpenHands)

Common schema (all three, verified from parquet):
```
instance_id: string
file_changes: list<struct<file: string,
                          changes: struct<added_entities: list<string>, added_modules: list<string>,
                                          edited_entities: list<string>, edited_modules: list<string>>>>
repo: string
base_commit: string   (null for SWE-smith)
problem_statement: string
patch: string
```
Entity strings are `"<file path>:<Class>.<method>"` or `"<file path>:<function>"`; module strings are `"<file path>:<Class>"`. Example: `{"file": "sepal_ui/sepalwidgets/inputs.py", "changes": {"added_entities": ["sepal_ui/sepalwidgets/inputs.py:DatePicker.check_date", ...], "added_modules": null, "edited_entities": ["sepal_ui/sepalwidgets/inputs.py:DatePicker.__init__"], "edited_modules": ["sepal_ui/sepalwidgets/inputs.py:DatePicker"]}}`. Lists are null when empty. No per-row license field in any of the three; HF `cardData.license` is unset for all three (the codescout code repo is MIT; upstream sources are SWE-rebench CC BY 4.0, SWE-Gym MIT, SWE-smith MIT).

| Dataset | Rows | Distinct repos | Language | base_commit | Files/row (mean, max) | Rows with >=1 entity | Entities/row | Overlap with SWE-QA-Bench 15 |
|---|---|---|---|---|---|---|---|---|
| `OpenHands/SWE-rebench-code-search` | 17,591 (85 MB) | 3,103 | 100% `.py` files in `file_changes` | present (40-char) | 2.24, 15 | 16,981 | 4.81 | 9: astropy, conan, matplotlib, xarray, scikit-learn, sphinx, sqlfluff, streamlink, sympy |
| `OpenHands/SWE-smith-py-code-search` | 39,287 (44 MB) | 131 (`swesmith/<owner>__<repo>.<sha7>` mirrors; commit is embedded in the repo name, `base_commit` column is null) | 100% `.py` | null | 1.11, 31 | 38,683 | 1.56 | 2: conan, sqlfluff |
| `OpenHands/SWE-Gym-code-search` | 2,317 (6 MB) | 11 | 100% `.py` | present | 1.91, 66 | 2,273 | 3.83 | 1: conan |

Notes: SWE-smith problem statements are synthetic bug descriptions over mutated code (mirror repos under the `swesmith` org must be cloned, not the upstream). SWE-rebench rows are real issues; problem statements up to 52K chars. For our "where" tasks, the `edited_entities` list at `base_commit` is a directly checkable gold set (file F1, function F1), and the 9 overlapping repos should be held out or de-duplicated against SWE-QA-Bench evaluation commits.
