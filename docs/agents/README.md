# Agent onboarding

Read this first. It is everything an agent needs to work on one component of this project without reading the rest of the repo. Append what you learn to `LOG.md` in this folder.

## What the project is

RL-train a small model (Qwen3.5-4B, via Tinker) to be an efficient code Q&A agent, and serve it in a product. The agent answers a question about a repository by looking things up in a pre-built index, reading a few line ranges with read-only tools, and replying with an answer where every claim carries a citation `[path:L10-L20]` to lines it actually read. One trajectory per question, few tool calls, no code execution.

Deadline: two days from 2026-09-18. Deliverables: code, model checkpoints, a 15-minute talk.

**The one rule.** Anything the model sees at inference must exist identically during training: the prompt, the repo map, the five tools, the caps, the answer format. One environment class, three drivers: the trainer (Tinker), the teacher (Claude), the product (vLLM on Modal).

## Where things are

```
codeqa/            the Python package (import as codeqa.<component>.<module>)
  shared/          contracts.py = every schema. paths.py = the data/ layout. Import these; never redefine them.
  clients/         thin wrappers: tinker, anthropic, openai_compat (vLLM), github, hf
  agent/           THE CORE. prompts.py, tools.py, curation.py, env.py, driver.py, tests/, indexing/ (snapshot, index, summaries, repomap, strip_docstrings, cli = the pre-index step the tools read)
  datagen/         task sources (import DeepCodeBench, derive CodeScout, import SWE-QA, structural, teacher), filter, split
  grader/          task + trace -> reward. gates, citations, verifiers, judge, efficiency, grade
  trainer/         dataset builder + config -> tinker-cookbook RL loop
  serving/         Tinker checkpoint -> merged weights -> Modal volume
  evals/           profile x task file -> tables, plots
apps/              hosted things: inference (Modal vLLM), api (FastAPI SSE), web (Vite + React + shadcn SPA), trainer (Modal runner)
scripts/           smoke tests: smoke_data, smoke_clients, smoke_chat (A1), smoke_episode (A4), smoke_driver (A5), smoke_train (A6), smoke_grade (C1)
tests/fixtures/    mini_repo/, index/, tasks.jsonl, traces/<adversarial>.json
data/              gitignored. repos/ index/ tasks/{raw,train,eval} traces/ logs/ evals/ models/
docs/              README.md is the reading order. components.md, agent_design.md, contracts.md, data_sources.md, gap_specs.md, decisions.md, modal_runtime.md
docs/research/     evidence, not specs. docs/archive/ = superseded, do not read.
```

Full layout and import rules: `docs/components.md`. Your component's contract: `docs/contracts.md` (C1–C10).

## Lanes (one agent each, at most four at a time)

| Lane | File | Owner | Folders |
|---|---|---|---|
| A · Agent and runtime | `lanes/A_agent_runtime.md` | lead, sync — **done A1–A7**; API frozen (LOG 16:45) | `codeqa/agent/**`, `codeqa/clients/tinker.py`, `profiles.yaml` |
| B · Data | `lanes/B_data.md` | agent | `codeqa/datagen/**`, `data/tasks/**` |
| C · Training | `lanes/C_training.md` | agent | `codeqa/grader/**`, `codeqa/trainer/**`, `codeqa/evals/**` |
| D · Product | `lanes/D_product.md` | agent | `apps/web/**`, `apps/api/**` (day two: `codeqa/serving/**`, `apps/inference/**`) |
| E · Deploy | `lanes/E_deploy.md` | agent, starts when D2 is live | `scripts/deploy/**`, `.gitignore`, `README.md` deploy section; GitHub repo + one EC2 |

Launch brief for an agent: read this page, then your lane file, work the checklist top to bottom, append to the lane's progress log after each item, append cross-lane learnings to `LOG.md`, and stop with a contract change request if a schema must change.

## Rules every agent follows

1. **Contracts are law.** Schemas live in `codeqa/shared/contracts.py`. If your work needs a schema change, stop and say so in `LOG.md` under "Contract change requests"; do not fork the schema locally.
2. **Import rules.** Everyone may import `codeqa.shared` and `codeqa.clients`. `agent` and `grader` import nothing else in the package. `datagen`, `trainer`, `evals`, `apps/api` import the core and never each other. No component reads another's files except through `codeqa/shared/paths.py`.
3. **Build against fixtures first.** `tests/fixtures/` has a mini repo, its index, five task records, and adversarial traces. Your component must run on those before it touches `data/`.
4. **Every component folder ships** its code, a `cli.py` (or `__main__`) that runs standalone, a `README.md` stating what it consumes and produces by contract name, and its own `tests/`. Done means the tests pass and the CLI runs on fixtures.
5. **No Docker, no sandboxes, no vector DB, no LangChain.** Tools are read-only functions over extracted folders. The only containers are Modal images for vLLM.
6. **Fail fast.** Any call to an external service gets a hard timeout under 60s and prints progress unbuffered. Never pipe a long command through `tail`. Anything slower runs in the background. macOS has no `timeout` binary; use `asyncio.wait_for`.
7. **All data is JSON or JSONL** under `data/`, written through `codeqa/shared/jsonl.py`. No databases.
8. **Do not commit** `data/`, `.env`, or model weights.
9. **Write to `LOG.md`** when you learn something another agent would need: a gotcha, an API quirk, a measured number, a decision you had to make.

## Running things

```
uv sync --group dev                       # once
uv run python -m scripts.smoke_data       # HF rows -> tasks, snapshot flask, index, cross-check (no keys needed)
uv run python -m scripts.smoke_clients    # Tinker, Anthropic, Modal, GitHub, HF (keys from .env)
uv run python -u -m scripts.smoke_chat    # TinkerChatClient round trip (offline checks + one live 3-turn chat)
uv run python -u -m scripts.smoke_episode # one cookbook episode via RepoEnv on a real SWE-QA task
uv run python -u -m scripts.smoke_driver qwen4b-base   # run_episode through a chat client (also: claude)
uv run python -u -m scripts.smoke_train smoke1         # 10 tasks x 4 x 3 steps, checkpoint, sample from it (~3 min, costs money)
uv run python -m codeqa.agent.indexing.cli all --repos data/repo_list.txt [--fast] [--nodoc]   # snapshot -> index -> summaries -> map
uv run pytest codeqa/agent                # tools + indexing tests, offline
uv run pytest                             # all tests
```

Keys live in `.env` (see `.env.example`): `TINKER_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`. Modal auth comes from `~/.modal.toml` (never put empty `MODAL_TOKEN_*` lines in `.env`; they override the profile). GitHub uses `gh auth token`. HF is public for our datasets.

Modal resources that exist: secret `codeqa` (Tinker + Anthropic keys), volumes `codeqa-data` and `codeqa-models`.

## What is already proven (2026-09-18, updated 16:45)

- Contracts validate against real rows from DeepCodeBench, CodeScout, and SWE-QA-Bench; lane B imported all of both eval sets and snapshotted 23 repos.
- Indexing: snapshot + tree-sitter symbols + Haiku summaries + ≤3k-token map on flask in 13 s (`cli all`); nodoc variant keeps line numbers.
- `TinkerChatClient` renders/parses with the cookbook renderer; a hand-written tool call round-trips exactly (`smoke_chat`).
- The five tools pass 13 offline tests on the flask snapshot; `RepoEnv` runs a real SWE-QA task through the cookbook env (`smoke_episode`).
- `run_episode` works with Tinker base and Claude on the same question (`smoke_driver`): base Qwen answers correctly but without bracketed citations; Claude cites and verifies.
- First RL run (`smoke_train smoke1`, 10 graphiti tasks × 4 × 3 steps, real grader): reward 0.10 → 0.37, format pass 47 % → 87 %, tool calls 6.0 → 2.6; the step-3 checkpoint (`profiles.yaml: qwen4b-smoke1-step3`) cites correctly on an unseen flask question. Plumbing proof, not a result (same tasks each step).
- Grader (lane C): 56 offline tests + live judge test; the trainer smoke used it as the reward with zero judge calls (all tasks verifiable).
- Anthropic: Haiku 4.5 and Sonnet 5 respond; tool calls parse. Modal: secret + volumes exist (unused until day two).

## Gotchas already found (do not rediscover)

- The cookbook tool env does **not** inject tool declarations. `TinkerChatClient.chat` and `RepoEnv.make_cookbook_env` prepend the renderer prefix themselves (`clients.tinker.with_tool_prefix`); `RepoEnv.initial_messages()` is deliberately client-agnostic. Never build the prefix a third time.
- The renderer keeps thinking in history (`strip_thinking_from_history=False`, set once in `clients.tinker.renderer()`); the RL env re-renders the whole history every step, so stripping would break the sequence-extension property. Do not construct renderers elsewhere.
- The Qwen3.5 renderer strips leading whitespace on a message's first line, so `read_file` numbers lines as `L41 | code` (C3) and tool output must not rely on leading padding.
- ripgrep is not installed on the lead's Mac (the shell `rg` is a Claude Code shim); `RepoTools.grep` falls back to a pure-Python scan with identical output.
- Inside `reward_fn(history, env)` there are no token counts (`trace.stats.prompt_tokens` = 0 in training). Token counts live in `compute_group_rewards(trajectory_group, env_group)` via `trajectory.transitions[i].ob.length`; the run-two efficiency term belongs there.
- Tool argument validation errors come back as tool messages (cookbook `error_tool_result`), never exceptions; Claude recovered from one in `smoke_driver`. Keep it that way.
- Every repo the env runs on needs `data/index/<repo_id>/map.txt`; `RepoEnv` raises otherwise. Build with `cli map` (or `summarize` then `map`).
- `@tool` resolves type hints at decoration time. Under `from __future__ import annotations`, every name in a tool signature (`ToolResult`, `Annotated`) must be importable at module level.
- Cookbook `RewardFn` is `async (history) -> (reward, metrics)`; ours via `RepoEnv.make_cookbook_env` is `async (history, env)` so the grader can read `env.files_read()`. `EnvGroupBuilder.compute_group_rewards` is the group-level hook (lane C's `trainer/group_rewards.py` fills judge errors to the group mean there). `train.Config.remove_constant_reward_groups` exists; the smoke ran with it off.
- anthropic SDK ≥ 1.7 rejects `temperature`. Keys not scoped to a workspace need the `anthropic-workspace-id` header (the client reads `ANTHROPIC_WORKSPACE_ID`).
- Tinker SDK retries 402/5xx for minutes by default; `clients/tinker.py` sets `max_retries=1`.
- Empty `MODAL_TOKEN_*` env vars override `~/.modal.toml`.
- tree-sitter must stay `< 0.26` (pinned). `tree-sitter-language-pack` is the grammar source.
- vLLM LoRA serving on Qwen3.5 is unreliable; serve merged weights.
- `datasets-server.huggingface.co` returns 502s sometimes; `clients/hf.py` retries.

## Reward, in one block (so you grade consistently)

```
gates:   format valid; citations parse; cited files exist; every cited range was read this episode  -> any failure = 0
correct: exact/F1 match for locate|value|enumerate|codescout; rubric fraction via judge for judged types -> [0, 1]
eff:     1.0 in run one; [0.5, 1] from tool calls + prompt tokens vs budget in run two
reward = correct * eff        (judge API failure -> NaN -> group mean; format failure -> 0;
                               no final answer (stop=budget|max_turns) -> -0.1, decided 2026-09-18 evening)
```

Threat model and adversarial fixtures: `docs/gap_specs.md` §3.

## Data sources, in one table

| Source | Role | Rows | Repos |
|---|---|---|---|
| DeepCodeBench train (`Qodo/deep_code_bench`) | train, natural, facts = rubric | 912 | 8 |
| DeepCodeBench test | held-out in-repo | 232 | same 8 |
| CodeScout (`OpenHands/SWE-rebench-code-search`) | train, programmatic locate | top ~15 repos of 17,591 rows | ~15 |
| Structural generator | train, programmatic | 40–60 per repo | training repos |
| Teacher generator (Claude) | train, judged | 400–600 | training repos |
| SWE-QA-Bench (`swe-qa/SWE-QA-Benchmark`) | held-out unseen repos, external comparison | 720 | 15 |

Exclude the nine CodeScout repos that overlap SWE-QA-Bench. Details: `docs/data_sources.md`.

## Precedents to be aware of

RepoSearch-R1 (RL on Qwen3-8B, five tools like ours, judge-only reward), SWE-QA-Pro (SFT then GRPO on Qwen3-8B, judge-only reward, RL added one point, 15 turns / 23 calls per question), DeepRepoQA (inference-time tree search on frontier models). None grade citations or efficiency. That is the gap this project fills.
