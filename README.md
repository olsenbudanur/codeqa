# codeqa — an RL-trained, citation-grounded code Q&A agent

A small model (Qwen3.5-4B, trained with RL on [Tinker](https://thinkingmachines.ai/tinker/)) that answers questions about a code repository by looking things up in a pre-built index, reading a few line ranges with read-only tools, and replying with an answer where **every claim carries a citation to lines it actually read**. One trajectory per question, few tool calls, no code execution, no sandbox.

The same environment serves three drivers: the trainer (Tinker), the teacher (Claude), and the product (a FastAPI backend plus a React UI that streams the agent's research log and marks each citation verified or not).

```
Q: Where is the Flask application class defined, and what does it inherit from?

  find_symbol("Flask", kind="class")        -> src/flask/app.py:L81-L1054  class Flask(App)
  read_file("src/flask/app.py", 81, 100)

A: `Flask` is defined in src/flask/app.py and inherits from `App` [src/flask/app.py:L81-L81] ...
   Sources: [src/flask/app.py:L81-L100] class definition and docstring
```

## Why this exists

Prior work on repo Q&A agents (RepoSearch-R1, SWE-QA-Pro, DeepRepoQA) rewards only answer quality as judged by an LLM. None of them check that the citations are real, that the cited lines were actually read, or how many tool calls the answer cost. This project trains for all three:

- **Grounding.** A citation to a file or range the agent never opened scores zero. Fabrication is a gate failure, not a partial-credit deduction.
- **Correctness.** Verifiable task types (locate, value, enumerate) are scored programmatically with zero API calls. Judged types (trace, explain) use a cheap LLM judge over atomic rubric items.
- **Efficiency.** Run two multiplies the reward by a term that falls off once tool calls or prompt tokens pass half the budget.

Target: under 8 tool calls per question at equal or better correctness than the precedents (SWE-QA-Pro's trained 8B model averages 23 tool calls per question).

## How an episode works

**Before any question**, the repo is indexed once (`codeqa/agent/indexing`): tree-sitter symbols with file and line ranges, one-paragraph LLM summaries per directory, and a token-capped repo map (about 3k tokens) rendered from both. No embeddings. The same index is used in training and in the product.

**What the model sees.** The rules (cite every claim as `[path:L10-L20]`, cite only lines you read, answer as soon as the evidence is enough, you have N tool calls, keep it under M tokens), the repo map, and the question. Thinking is on.

**The five tools**, all read-only, all outputs curated (ranked, deduplicated, capped) and line-numbered so citations are copied rather than invented:

| Tool | What it does |
|---|---|
| `overview(path)` | Summary of a directory or file, its children, its top symbols |
| `find_symbol(name, kind?, file_pattern?)` | Definitions matching a name, with file and line range |
| `grep(pattern, file_pattern?)` | Regex hits grouped by file, with line numbers |
| `read_file(path, start, end)` | Numbered lines in a range, max ~150 |
| `list_dir(path)` | Entries with type and line counts |

**How it ends.** The model replies without a tool call. That reply is the answer. Episodes also end on the tool-call budget, max turns, or context overflow, each with a small negative reward.

Full design and the precedents it borrows from: `docs/agent_design.md`. The code: `codeqa/agent/README.md`.

## The reward

```
gates:   answer exists and fits the cap; citations parse; cited files and ranges exist;
         every cited line was read this episode                    -> any failure = 0
correct: locate | value | enumerate  -> literal / symbol-set / path-set match, no API calls
         trace | explain             -> Haiku judge, fraction of rubric items satisfied  -> [0, 1]
eff:     1.0 in run one; [0.5, 1] from tool calls + prompt tokens vs budget in run two
reward = correct * eff
         judge API failure -> NaN -> group mean (zero advantage)
         no final answer (budget or max turns) -> -0.1
```

The grader is a pure library with three consumers: the trainer for reward, the product for verified citation badges, and evals for held-out scoring. Its threat model covers padding, confident wrong answers, fabricated citations, citing unopened files, zero tool calls, judge injection, answers hidden in thinking, and redundant reads, each with an adversarial fixture trace in `tests/fixtures/traces/`. See `codeqa/grader/README.md`.

## Core parts

| Component | Folder | Consumes | Produces |
|---|---|---|---|
| Shared | `codeqa/shared/` | nothing | `contracts.py` (every schema), `paths.py`, `profiles.py`, JSONL helpers |
| Clients | `codeqa/clients/` | keys | thin wrappers for Tinker, Anthropic, OpenAI-compatible (vLLM), GitHub, Hugging Face; one `ModelClient` protocol |
| Agent | `codeqa/agent/` | repo list, clients | index under `data/index`, `RepoEnv`, `run_episode`. **The core.** Identical in training, teacher, product |
| Datagen | `codeqa/datagen/` | agent, grader, public datasets | `data/tasks/{raw,train,eval}` |
| Grader | `codeqa/grader/` | task + trace | `GradeResult`, `check_citations` |
| Trainer | `codeqa/trainer/` | tasks, agent, grader, Tinker | checkpoints, `data/logs/<run>/` |
| Evals | `codeqa/evals/` | profiles, task files, agent, grader | `data/evals/<profile>/<set>/`, plots, the SWE-QA judge score |
| Serving | `codeqa/serving/`, `apps/inference/` | a Tinker checkpoint | merged weights on a Modal volume, a vLLM endpoint (day two) |
| Product | `apps/api/`, `apps/web/` | agent, grader, profiles | the demo: FastAPI SSE backend, Vite + React + shadcn SPA |

Import rules: everyone may import `shared` and `clients`; `agent` and `grader` import nothing else; `datagen`, `trainer`, `evals`, and `apps/api` import the core and never each other. No component reads another's files except through `codeqa/shared/paths.py`. Details and rationale: `docs/components.md`.

Endpoint profiles live in `profiles.yaml`: `claude`, `haiku`, `qwen4b-base`, and one `qwen4b-<run>-step<N>` per trained checkpoint (a `tinker://` path, no keys).

## Data

Natural repo Q&A with pinned commits is scarce (a census found about 2.7k rows publicly), so training data is mostly generated over a small, fixed set of repos.

| Source | Role | Ground truth |
|---|---|---|
| DeepCodeBench train (Qodo) | train, natural questions | facts list as rubric, symbol spans as evidence |
| CodeScout (OpenHands SWE-rebench-code-search) | train, programmatic locate and trace | edited files and entities from real issues, question rewritten by Haiku |
| Structural generator | train, programmatic | exact locations and literals from the tree-sitter index, over a docstring-stripped snapshot so grepping the paraphrase does not work |
| Teacher generator | train, judged | Claude answers first with the real tools, then a blind Haiku run must satisfy the rubric |
| DeepCodeBench test | held-out, same repos | as above |
| SWE-QA-Bench | held-out, 15 repos never trained on | annotator answers with paths and lines; also scored with the benchmark's own five-dimension judge |

Every raw task is sampled a few times with the base model and filtered to a pass-rate window so GRPO groups have variance. Current training file: 1,489 tasks over 23 repos, 62 % programmatic. Counts and how each source becomes a task: `docs/data_sources.md`, `codeqa/datagen/README.md`, `data/tasks/README.md`.

## Training

The trainer is a thin dataset builder and config on top of `tinker-cookbook`'s RL loop (LoRA rank 32, importance-sampling loss, `remove_constant_reward_groups`). One task becomes a group of `RepoEnv` instances; rewards are computed per group so judge failures can fall back to the group mean and the efficiency term can see token counts.

```
uv run python -u -m codeqa.trainer.run --tasks data/tasks/train/all.jsonl --profile qwen4b-base \
    --run-name run1 --lr 1e-4 --group-size 8 --groups-per-batch 16 --steps 50 \
    --eval-tasks data/tasks/eval/fast.jsonl --eval-every 10 --save-every 10
uv run python -m codeqa.evals.monitor --run run1        # per-step table, collapse checks, plots
```

Decisions taken along the way, with the evidence behind each (learning rate, judge choice, answer caps, the no-answer penalty, why grep hits count as read lines): `docs/decisions.md`.

## Evaluation

```
uv run python -u -m codeqa.evals.run --profile <profile> --tasks data/tasks/eval/fast.jsonl
uv run python -m codeqa.evals.report --set fast --markdown
uv run python -m codeqa.evals.sweqa_judge --profile <profile> --set sweqa      # external SWE-QA number, Sonnet judge
```

Baselines on the 120-task fast set (60 DeepCodeBench test + 60 SWE-QA), measured 2026-09-18 before the answer caps were raised; the trained checkpoints go in the same table:

| Profile | Reward | Correct rate | Citation valid | Tool calls | SWE-QA judge /100 |
|---|---|---|---|---|---|
| Claude Sonnet 5 | 0.20 | 0.22 | 0.24 | 5.2 | 73.1 |
| Qwen3.5-4B base | 0.00 | 0.00 | 0.02 | 7.7 | 40.6 |

The base model reads the right files but answers without the citation format or runs out of turns. The first smoke run (10 tasks, 3 steps) moved format compliance from 47 % to 87 % and tool calls from 6.0 to 2.6, which is the shape of signal the full run is after.

## Product

`apps/api` runs one episode per question through the same environment and streams C9 events over SSE; `apps/web` renders them as a research log, then the answer with citation chips (green: the range was read; amber: it was not) that open a file viewer at the cited lines. A repo can be added by URL and is indexed on the fly (a 38-file repo is ready in about 6 seconds).

```
uv run uvicorn apps.api.server:app --port 8000
cd apps/web && VITE_API_URL=http://localhost:8000 pnpm dev      # http://localhost:5173
cd apps/web && pnpm dev                                          # no backend: replays a recorded episode
```

Routes: `/` home, `/app` workbench, `/compare` one question on two profiles side by side. Demo script: `docs/demo.md`.

## Getting started

```
uv sync --group dev
cp .env.example .env            # TINKER_API_KEY, ANTHROPIC_API_KEY, ANTHROPIC_WORKSPACE_ID
uv run pytest -q -m "not live"  # offline tests on the fixtures, no keys needed
uv run python -m scripts.smoke_data      # HF rows -> tasks, snapshot flask, index (no keys)
uv run python -m scripts.smoke_clients   # every external service, fails fast
uv run python -m codeqa.agent.indexing.cli all --repos data/repo_list.txt     # snapshot -> index -> summaries -> map
uv run python -u -m scripts.smoke_driver claude                                 # one episode end to end
```

Requirements: Python 3.11 or 3.12 through `uv`, Node 20 with `pnpm` for the web app. Modal auth comes from `~/.modal.toml`. `data/` is gitignored and holds every snapshot, index, task file, trace, log, and eval; all of it is JSON or JSONL and regenerable from the CLIs above.

Smoke scripts in `scripts/` prove one seam each: `smoke_chat` (Tinker renderer round trip), `smoke_episode` (one cookbook episode), `smoke_driver` (one episode through any client), `smoke_grade` (the grader on fixtures), `smoke_train` (a three-step RL run that costs money).

## Layout

```
codeqa/          the Python package: shared, clients, agent, datagen, grader, trainer, evals, serving
apps/            api (FastAPI SSE), web (Vite + React), inference (Modal vLLM), trainer (Modal runner)
scripts/         smoke tests; deploy/ (secret scan, EC2 bootstrap and update)
tests/fixtures/  a mini repo, its index, five task records, adversarial traces
data/            gitignored: repos/ index/ tasks/ traces/ logs/ evals/ models/
docs/            reading order in docs/README.md; docs/agents/ is the onboarding page and lane briefs
```

## Deploy

The demo runs on one small EC2 instance with no GPU: inference is remote (Tinker sampling for trained checkpoints, Anthropic for the teacher). Caddy serves the built web app over HTTPS with basic auth and proxies `/api/*` to the FastAPI process under systemd. The repo holds no secrets and no data; `.env` is copied to the box and `data/` is synced with `rsync`.

```
scripts/deploy/secret_scan.sh          # before every push: forbidden paths, secret-shaped strings, live key values
```

The bootstrap and update scripts, the systemd unit, the Caddyfile, and the ops notes (update, rollback, logs, cost, stop) land here as lane E completes; the plan is in `docs/agents/lanes/E_deploy.md`.

## Docs

Start at `docs/README.md` for the reading order. `docs/contracts.md` names the ten seams between components (task record, trace, grade result, tool API, profiles, SSE events), all defined once in `codeqa/shared/contracts.py`. `docs/gap_specs.md` covers the checkpoint manifest, threat model, judge failure policy, in-loop eval, shaping variants, and on-demand indexing. `docs/research/` holds the evidence (dataset census, tool survey, paper notes).
