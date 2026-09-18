> Final component list and repo layout: see [components.md](components.md). Component names there supersede the stream letters and numbering below.

# Code Research Agent — System Design

Nine components, three contracts, one rule. A small model learns to explore a repo with five read-only tools and answer with verified citations. The same environment that trains it also serves it.

**The one rule.** Anything the model sees at inference must exist identically during training: the prompt, the repo map, the five tools, the caps, the answer format. One environment class, three drivers: the trainer, the teacher, the product.

## The whole system

```
OFFLINE (once per repo)
  GitHub --tarball--> Snapshot --files--> Index --symbols--> Task generators --records--> tasks.jsonl
                                                                 |                            |
                                                   teacher runs the same env with Claude      | batches
                                                                 v                            v
TRAINING (Tinker loop)
  Grader <--history-- [ Repo Env: 5 tools, prompt, map ] <--runs 8 episodes/task-- Trainer <--tokens/grads--> Tinker
     |                                                                                |
     +---------------------- reward + metrics per episode ----------------------------+
                                        ^                                             |
                                        | same env, same tools                        | export, merge
PRODUCT (per question)                  |                                             v
  Browser <--SSE--> FastAPI backend  ---+   ---chat completions per turn--->   vLLM on Modal
                         |
                         +--verify citations--> Grader
```

One environment, three drivers. Only the model behind it changes: Claude for the teacher, Tinker's sampler in training, vLLM on Modal in the product.

## One episode

```
Prompt (rules + repo map + question) -> Model turn (think, then act)
    -> tool calls -> Env runs tools, appends numbered results -> Model turn ... (loop)
    -> reply with no tool call = the answer -> Grader (reward) or render (product)
```

Episode also ends on max turns, tool-call budget, context overflow, or an unparseable call (small negative reward). Loss is computed only on tokens the model wrote; the cookbook masks prompt and tool results.

## The three contracts

Write these first. Every component reads or writes one of them.

- **Task record.** One JSONL line per task: what the prompt needs plus what the grader needs. Written by generators, read by trainer and eval.
- **Episode history.** The message list the cookbook hands to the reward function, plus tool-call counts and tokens read. One JSON per episode on disk. Read by grader and viewer.
- **Model endpoint profile.** Model id, renderer name, context cap, tool-call parser. Names a Tinker model in training, an OpenAI-compatible URL in the product. Swapping 4B for 9B is a new profile.

```json
{
  "task_id": "flask-0042",
  "repo_id": "pallets/flask@a1b2c3d",
  "question": "Where is the session cookie signed, and what key is used?",
  "task_type": "trace",
  "grading": {
    "expected_paths": ["src/flask/sessions.py"],
    "expected_literal": null,
    "reference_answer": "...",
    "rubric": ["names SecureCookieSessionInterface", "cites the signing call"],
    "required_citations": [{"path": "src/flask/sessions.py", "start": 310, "end": 340}]
  },
  "budget": {"max_tool_calls": 10, "max_turns": 8, "max_answer_tokens": 400},
  "split": "train"
}
```

## Components

Tag says who builds it: an agent against a spec, or you, because the spec is the hard part.

### 1. Snapshot — agent builds
- **Does:** fetch GitHub tarball at a pinned commit; strip vendored, generated, binary, oversize files (linguist vendor list); normalize line endings; write a manifest.
- **In:** the fixed repo list from the data sources below, about 25 training repos and 15 eval repos, each at the commit the source pins. Few repos on purpose: everything runs locally, and every repo costs a snapshot, an index, and summaries.
- **Out:** `data/repos/<id>/` + `manifest.json`.
- **Size:** ~150 lines.

### 2. Index — agent builds
- **Does:** structural layer with tree-sitter (symbols with line ranges, file tree, import graph); semantic layer with Haiku (paragraph per directory, line per large file); token-capped repo map for the prompt.
- **In:** a snapshot folder.
- **Out:** `data/index/<id>/symbols.json`, `summaries.json`, `map.txt` (~3k tokens).
- **Depends:** Snapshot; Anthropic key (~50–100 calls per repo). Pin tree-sitter < 0.26.

### 3. Repo Env — you design, agent fills in
- **Does:** stateful class per repo exposing `overview(path)`, `list_dir`, `read_file(path, start, end)`, `grep(pattern, glob)`, `find_symbol(name)` as cookbook `@tool` methods. Builds initial messages from rules + map + question. Numbered lines. Forgiving errors: nearest paths on miss, first chunk on oversize range, warning when one call remains.
- **In:** task record + model profile.
- **Out:** an env the cookbook can run; same tool specs for product and teacher drivers.
- **Depends:** Snapshot, Index, tinker-cookbook tool_use, ripgrep.
- **You decide:** tool names/output formats, system rules, citation pattern, caps per task type.

### 4. Task sources — you design, agent fills in
- **Does:** four sources feed one filter. Import: DeepCodeBench train rows as-is, facts list becomes the rubric, cited function resolved to a line range through the index. Derive: CodeScout rows on the chosen repos, issue text rewritten by Haiku into a where/which question, gold = edited files and entities, graded programmatically. Generate structural: symbol → paraphrased question, answer = location or literal. Generate teacher: Claude runs Repo Env answer-first, writes question/reference/rubric/citations, a blind second run confirms. Then dedupe, base-model pass-rate filter 10–90%.
- **Out:** `data/tasks/train.jsonl`, `heldout.jsonl`. Target 1500–2500 kept from ~3000 raw, ~55% programmatically graded.
- **Held-out:** in-repo = DeepCodeBench test (232, same 8 repos); unseen-repo = SWE-QA-Bench (720, 15 repos never trained on). See Data sources.
- **You decide:** the CodeScout rewrite prompt, teacher prompt, ground truth per type, filter thresholds.

### 5. Grader — you design, agent fills in
- **Does:** format gate; citation checker (path exists, lines exist, file was opened this episode); correctness (exact match or rubric judge with reference); efficiency multiplier from tool calls + context tokens, only when correctness passes. Returns reward + metrics dict.
- **Out:** reward in [0,1], metrics, one JSON trace per episode.
- **Depends:** task contract only; build first against fixtures. Judge ~250 calls per training step.
- **You decide:** gating order, efficiency formula, adversarial tests (padded, confident wrong, restated question, fabricated citation).

### 6. Trainer — agent drafts, you review
- **Does:** dataset builder loading task records, group-size copies of Repo Env per task; config with model profile, LoRA rank 32, cookbook LR, group 8, 32–64 tasks/step, eval every 10 steps; calls the cookbook training entrypoint.
- **Out:** checkpoints in Tinker; `logs/<run>/metrics.jsonl`, rollout summaries, cookbook HTML rollout viewer. WandB optional.
- **You review:** mask covers only model tokens; rewards vary within a group; loss finite. Smoke test 10 tasks × 3 steps first.

### 7. Export and serve — agent builds
- **Does:** download LoRA, merge with cookbook weights helper, upload to Modal volume, vLLM OpenAI-compatible server with thinking on and Qwen tool-call/reasoning parsers.
- **Out:** URL + model name for the endpoint profile. One deployment per model.
- **Gotcha:** serve merged weights; vLLM LoRA on Qwen3.5 is unreliable. Rehearse on day one with the smoke-test checkpoint.

### 8. Product — you design UX, agent builds
- **Does:** backend picks repo, runs one Repo Env episode with chosen endpoint, streams tool calls + thinking as SSE, verifies citations with Grader's checker, reports tool calls/tokens/time. Frontend: text or dictation, live research log, answer with citations opening file at line, sources panel, model switcher.
- **Out:** FastAPI + one static page. React only if time.
- **Depends:** works on day one against Claude or base Qwen.

### 9. Eval and traces — agent builds
- **Does:** run held-out against any endpoint; report correctness, citation validity, tool calls and tokens per correct answer, answer length, by task type; plot from metrics JSONL; compare base / trained / Claude.
- **Signal:** reward up while citation validity flat = hacking.

## Data sources

Verified by fetching rows (details and stats in `research/source_shapes.md`, `research/coreqa_and_sweqa_research.md`). Chosen to keep the repo count small.

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

## On disk

```
data/
  repos/<repo_id>/            extracted snapshot + manifest.json
  index/<repo_id>/            symbols.json  summaries.json  map.txt
  tasks/train.jsonl           task records
  tasks/heldout.jsonl         split by repo
logs/<run>/                   metrics.jsonl, rollout summaries, cookbook HTML viewer
traces/<run>/<episode>.json   full history + reward components
models/<name>/                merged weights, uploaded to the Modal volume
```

## Build order

1. **Contracts and Repo Env spec.** One hour. Unblocks everything.
2. **Fan out.** Snapshot, Index, Grader against fixtures, Export/serve with base Qwen, Product shell — in parallel.
3. **Repo Env with Claude as the model.** Also the teacher path; first integration test.
4. **Trainer smoke test.** 10 tasks, 3 steps. Then rehearse export + serve with that checkpoint.
5. **Import, derive, generate, filter.** DeepCodeBench import and CodeScout derivation first (hours), structural next, teacher in background. Pass-rate filter.
6. **Overnight run one.** Correctness only.
7. **Day two.** Read traces, add efficiency term, run two. Finish product. Eval, plots, talk.