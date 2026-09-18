# Agent Design (simple version)

## What it is

A small model that answers a question about a repo by looking things up in a pre-built index, reading a few line ranges, and replying with an answer where every claim has a citation to lines it actually read. One trajectory per question. No search tree, no helper agents.

## Before any question: index the repo once

Every repo gets indexed at snapshot time. The same index is used in training and in the product.

- **Structural layer** (code, free): every class and function with its file, line range, signature, and parent, from tree-sitter. Plus the file tree and an import graph.
- **Semantic layer** (cheap LLM, once): one paragraph per directory or module, one line per large file. This replaces embedding search, which DeepRepoQA found to be the least important part of their system.
- **Repo map** (rendered from the two layers): a token-capped text overview, about 3k tokens, that goes into the prompt.

## What the model sees

1. **Rules.** Cite every claim as `[path:L10-L20]`. Only cite lines you read. Answer as soon as the evidence is enough. You have N tool calls. Keep the answer short.
2. **The repo map.**
3. **The question.**

Thinking is on. The model may reason briefly before each action.

## The five tools

All read-only. All results are curated before the model sees them: ranked, deduplicated, boilerplate collapsed, capped. Every line comes back numbered so citations are copied, not invented.

| Tool | What it does | Notes |
|---|---|---|
| `overview(path)` | Summary of a directory or file, its children, its top symbols | The conceptual entry point. Cheap in tokens. |
| `find_symbol(name, kind?, file_pattern?)` | Definitions matching a name, with file and line range | The structural entry point. DeepRepoQA's best runs used this with a file pattern. |
| `grep(pattern, file_pattern?)` | Regex hits grouped by file, with line numbers | Literal fallback when names are unknown. |
| `read_file(path, start, end)` | Numbered lines in a range, max ~150, with total line count | Line ranges, never whole files. |
| `list_dir(path)` | Entries with type and line counts | Rarely needed once the map exists. |

Errors are forgiving: a bad path returns the nearest matching paths; an oversized range returns the first chunk and the total; when one call remains, the result says so and tells the model to answer.

## How a turn works

1. Model thinks briefly, then emits one or more tool calls. Several calls in one turn cost one round trip, so parallel calls are encouraged.
2. The environment runs them and appends the curated results.
3. Repeat.

## How it ends

The model replies **without a tool call**. That reply is the answer. The episode also ends on max turns, the tool-call budget, context overflow, or an unparseable call, each with a small negative reward.

## Rules the model must follow, and the grader checks

- **At least one citation**, in the exact format. No citation means the answer does not count. (DeepRepoQA's termination rule.)
- **Citations point to lines that were read in this episode.** A citation to an unopened file is fabrication and scores zero.
- **Verify before finishing.** The last read should cover the range being cited. DeepRepoQA found this reduces paraphrase drift; for us it falls out of the grounding rule.
- **No redundant reads.** Re-reading the same range costs a small penalty.

## Budgets, per task type, stored in the task record

| Type | Tool calls | Turns |
|---|---|---|
| locate, value | ~6 | 6 |
| enumerate, trace | ~10 | 8 |
| explain | ~12 | 10 |

DeepRepoQA's budget curve gains most between 5 and 10 steps and plateaus by 20. Our caps sit in that range, on a single trajectory instead of a tree.

## The behavior we want to emerge

- Read the map, pick a candidate. Call `overview` or `find_symbol` with a file pattern.
- Get a line number from the hit. Read only the range around it.
- Stop as soon as the claim is supported. Two to four calls for locate and value questions, four to eight for trace and explain.
- Every citation is to a range that was read.

## What we borrowed from DeepRepoQA, and what we replaced

**Borrowed (the environment):** index-backed actions; class/function lookup with a file pattern; evidence curation on tool outputs; the rule that an answer needs a supporting code span; line-range reads instead of whole files; dropping semantic search per their ablation.

**Replaced (the search):** their MCTS with perception, planning, execution, and evaluation agents on a 480B model becomes a single trajectory from a 4B model trained with RL. Their evaluation agent, which scores how useful each action was, is what the advantage signal teaches our policy to internalize.

**Comparison:** evaluate on SWE-QA-Bench with the same five-dimension judge (saved at `docs/research/swe_qa_llm_as_a_judge.py`). Keep those 15 repos out of the training corpus. Efficiency baseline: SWE-QA-Pro's trained Qwen3-8B averages 15 turns and 23 tool calls per question; RepoSearch-R1 caps at 10 turns, one tool per turn. Our target is under 8 tool calls at equal or better correctness.

**Precedents in one line each.** RepoSearch-R1: RL on Qwen3-8B with five tools like ours, judge-only reward, training-time tree search. SWE-QA-Pro: SFT then GRPO on Qwen3-8B, judge-only reward, RL added one point over SFT. DeepRepoQA: inference-time tree search on frontier models. None of them grade citations or efficiency; that is the gap this design fills.

## Same in training and product

The trainer, the teacher, and the product backend all run this exact environment. Only the model behind it changes: Tinker's sampler in training, Claude for the teacher, vLLM on Modal in the product. The product streams each turn as a research log row, renders citations as links to the file at the line, and marks each citation verified using the same checker the grader uses.