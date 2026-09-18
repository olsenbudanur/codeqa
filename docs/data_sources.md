# Data sources

Verified by fetching rows (details and stats in `docs/research/source_shapes.md`, `docs/research/coreqa_and_sweqa_research.md`). Chosen to keep the repo count small.

| Source | Role | Rows | Repos | Ground truth | License | How it becomes a task |
|---|---|---|---|---|---|---|
| DeepCodeBench train (Qodo) | train, natural | 912 | 8, one commit each | answer names file + function; `facts` list, mean 6.6 atomic statements | Apache-2.0 | as-is; facts = rubric; line range from index |
| DeepCodeBench test | held-out, in-repo | 232 | same 8 | same | Apache-2.0 | as-is |
| CodeScout SWE-rebench-code-search (OpenHands) | train, programmatic locate | 17,591 total; use top ~15 repos by row count, one commit each | 3,103 total | edited files + entities `path:Class.method`, 4.8 per row | unverified on HF | Haiku rewrites issue → where/which question; grade by file/entity match |
| Structural generator | train, programmatic | ~40–60 per repo | same repos as above | exact location / literal | ours | tree-sitter + Haiku paraphrase |
| Teacher generator | train, judged | ~400–600 | same repos | reference + rubric + spans | ours | Claude answer-first, blind verify |
| SWE-QA-Bench | held-out, unseen repos, external comparison | 720 | 15, pinned commits | answer with path + lines (76% both) | Apache-2.0 | eval only; citations by regex; their judge script |
| SWE-QA-Pro-Bench (TIGER-Lab) | optional final eval | 260 | 26 × 10 | answer with path + lines | MIT | eval only; take 5–10 repos if used |

Skipped, with reasons: SWE-QA-Pro RL train (464 rows over 464 repos, too many repos for local); SWE-QA-Pro SFT trajectories (no commits, different tools); CoReQA (never released); SWE-smith (synthetic mirrors, no base commit).

Exclusions: the 9 CodeScout repos that overlap SWE-QA-Bench (astropy, conan, matplotlib, xarray, scikit-learn, sphinx, sqlfluff, streamlink, sympy) are never used for training.

Repo budget: 8 DeepCodeBench + ~15 CodeScout = ~23 training snapshots; 15 SWE-QA-Bench eval snapshots. Structural and teacher tasks run over the 23 training repos only.

Precedents to cite: RepoSearch-R1 (RL on Qwen3-8B, 5 tools, judge-only reward), SWE-QA-Pro (SFT then GRPO on Qwen3-8B, judge-only reward, RL added +1 point, 15 turns / 23 tool calls per trajectory), DeepRepoQA (inference-time MCTS on frontier models).

## Also available (verified, not in the default plan)
- **Code-QA-Bench** (Lens-Frontier, MIT): 628 questions with rubric-style key files and pinned SHAs. Third eval set if wanted.
- **Long Code Arena bug localization** (JetBrains, Apache-2.0): 7,479 rows with base_sha and changed_files. Licensed fallback for CodeScout if its license cannot be confirmed.
- **SWE-QA-Pro RL train** (MIT): 464 rows over 464 repos with commits. Skipped only because of the few-repos rule.

Evidence: `docs/research/source_shapes.md`, `docs/research/dataset_census.md`, `docs/research/coreqa_and_sweqa_research.md`.
