# Lane A — Agent and runtime

**Owner:** lead (sync). **Status:** done (A1–A7); maintenance only. **Folders:** `codeqa/agent/**`, `codeqa/clients/tinker.py`, `scripts/smoke_*.py`.

## Mission
Build the environment every other lane runs against, and prove the whole chain once end to end: real tasks → cookbook env → three Tinker training steps → saved checkpoint → sample from it → cited answer.

## Consumes / produces
- Consumes: C1 manifest, C2 index (produces them too), C5 tasks from `data/tasks/raw/` (smoke files exist).
- Produces: C3 tools and driver, C4 prompts, C6 traces, the `RepoEnv` API every lane imports.

## Checklist
- [x] **A1 Tinker chat client.** `TinkerChatClient(profile).chat(messages, tools)` in `codeqa/clients/tinker.py`: render with the cookbook renderer (`create_conversation_prefix_with_tools` for specs, `build_generation_prompt`), `sample_async`, `parse_response` back to a `Message` with `tool_calls`. One sampling client per model path, cached. Profile kind `tinker` with `model` = base name or `tinker://` checkpoint path.
  Done when: a scripted three-turn conversation with a tool call round-trips and the parsed `ToolCall` matches what the model emitted.
- [x] **A2 Indexing.** `summaries.py` (Haiku, one paragraph per directory, one line per file over 300 lines, cached to `summaries.json`), `repomap.py` (tree + top symbols + summaries, ≤ 3k tokens), `strip_docstrings.py` (`<repo_id>__nodoc` snapshot + its own index), `cli.py` (`snapshot`, `index`, `summarize`, `map`, `all --repos <file>`).
  Done when: flask has `summaries.json` and `map.txt`; the nodoc variant indexes; `all` runs on a list of 3 repos without intervention.
- [x] **A3 Tools.** `tools.py`: `overview`, `find_symbol(name, kind, file_pattern)`, `grep(pattern, file_pattern)` via `rg --json`, `read_file(path, start, end)`, `list_dir`; all `@tool` on a stateful class; `curation.py` applies rank, dedupe, collapse, caps; errors are forgiving (nearest paths, first chunk, budget warning). Output formats exactly as C3.
  Done when: unit tests per tool for happy path, bad path, oversize, cap pass on the flask snapshot.
- [x] **A4 RepoEnv.** `env.py`: `RepoEnv(task, profile)` builds `initial_messages` = renderer tool prefix + system rules + user prompt (map + question); `tools()`, `files_read()`, call counting and the budget warning; `make_cookbook_env(reward_fn)` wrapping `build_agent_tool_env` with the task's budget.
  Done when: `scripts/smoke_episode.py` is rewritten to use `RepoEnv` on a task from `data/tasks/eval/smoke_sweqa_flask.jsonl` and completes.
- [x] **A5 Driver.** `driver.py`: `run_episode(env, client, on_event)` → `Trace` (C6) with stats and `stop_reason`; emits C9 events; works with Anthropic and Tinker clients.
  Done when: one episode via Claude and one via Tinker base both write valid trace JSON to `data/traces/dev/`.
- [x] **A6 First end-to-end training.** Stub grader (format + grounding only, no judge). Ten real tasks, group size 4, three steps through the cookbook loop (`tinker_cookbook.rl.train`), `metrics.jsonl` and the HTML rollout page present, `save_weights_for_sampler`, then `TinkerChatClient` on the saved path answers a flask question through `run_episode`.
  Done when: `data/models/manifest.json` has the checkpoint record and `data/traces/dev/` has a trace answered by it.
- [x] **A7 Handoff.** Freeze the `RepoEnv` / `run_episode` signatures; post in `docs/agents/LOG.md`; other lanes switch from stubs.

## Waits on / provides
- Waits on: nothing.
- Provides: A4 + A5 unblock B5, B7, C2, C3, D2. A6 gives C2 a working trainer skeleton to configure.

## Commands
```
uv run python -u -m scripts.smoke_episode
uv run python -m codeqa.agent.indexing.cli all --repos data/repo_list.txt
uv run pytest codeqa/agent
```

## Gotchas for this lane
- Prepend the renderer tool prefix or the model invents tool syntax. Five schemas ≈ 600–900 prompt tokens.
- `@tool` resolves hints at decoration time; keep `ToolResult`/`Annotated` module-level.
- Keep tool outputs small: the env re-renders the whole conversation every turn.
- Do not use `from tinker_cookbook.recipes.search_tool` imports; they pull chromadb. Copy patterns, not modules.

## Progress log (append-only)
Format: `- [YYYY-MM-DD HH:MM] A<n> done — one line with paths/numbers`
- [2026-09-18 12:55] pre-A: two-tool prototype episode ran end to end on flask (`scripts/smoke_episode.py`, `data/smoke_episode3.log`); untrained model used tools correctly, missed citation format.
- [2026-09-18 14:25] A1 done — `TinkerChatClient` in `codeqa/clients/tinker.py` (+ `to_cookbook`/`from_cookbook`/`with_tool_prefix`); `scripts/smoke_chat.py` passes: offline round trip exact, live 3 turns / 4 tool calls / 7s on flask (`data/smoke_chat.log`). Renderer keeps thinking in history (see LOG). Model cited as [`path:L81-L81`] with backticks → format gate would fail; expected RL signal.
- [2026-09-18 14:50] A2 done — `codeqa/agent/indexing/{summaries,repomap,strip_docstrings,cli}.py` + `snapshot.load_manifest`; `cli all --repos data/repo_list.txt` ran on flask + spectree + scim2-filter-parser (3/3, 20s total, `data/index_all.log`); flask: 61 Haiku summaries in 10s, map 153 lines / <3k tokens, nodoc variant 292 docstrings blanked with line numbers preserved; `codeqa/agent/tests/test_indexing.py` 3 pass.
- [2026-09-18 15:15] A3 done — `codeqa/agent/tools.py` (`RepoTools`, five `@tool` methods, call/error counting, `files_read` spans, budget note) + `codeqa/agent/curation.py` (caps, ranking, nearest paths, glob/substring patterns); `codeqa/agent/tests/test_tools.py` 13 pass incl. validation-error path. ripgrep is NOT installed on this Mac (shell `rg` is a Claude Code shim) → pure-Python grep fallback, same output; `brew install ripgrep` makes it faster, optional.
- [2026-09-18 15:15] A4 done — `codeqa/agent/env.py` `RepoEnv(task, profile)`: `initial_messages()`, `tools()/specs()`, `files_read()`, `make_cookbook_env(reward_fn(history, env))`, `trace_from_history()`, `from_question()`; `scripts/smoke_episode.py` rewritten on `RepoEnv` + real SWE-QA task: 3 turns / 2 reads / 7s, 4,177 prompt tokens at start, correct answer, no bracketed citation → stub reward 0 (`data/smoke_episode_env.log`, trace in `data/traces/dev/`).
- [2026-09-18 15:40] A5 done — `codeqa/agent/driver.py` `run_episode(env, client, on_event) -> Trace` + `save_trace`; `profiles.yaml` + `codeqa/shared/profiles.py` (`claude`, `haiku`, `qwen4b-base`); `scripts/smoke_driver.py <profile>` passes for `qwen4b-base` (6 turns / 5 calls / 12s / 46.8k prompt tokens / 0 bracketed citations) and `claude` (6 turns / 5 calls / 19s / 49.8k prompt tokens / 3 verified citations); traces in `data/traces/dev/sweqa-flask-001__{qwen4b-base,claude}.json`; logs `data/smoke_driver_{qwen,claude}.log`.
- [2026-09-18 16:40] A6 done — `scripts/smoke_train.py smoke1`: 10 graphiti DeepCodeBench tasks × group 4 × 3 steps through `tinker_cookbook.rl.train.main` with the real grader (C1) as reward, 132s training (~35s/step for 40 episodes), `data/logs/smoke1/{metrics.jsonl,checkpoints.jsonl,iteration_00000N/train.html}`; reward 0.10→0.32→0.37, format_ok 47.5%→87.5%, citations_grounded 12.5%→45%, tool calls 6.0→2.6, mixed-reward groups 40%→80%. Checkpoint `tinker://…/sampler_weights/final` → profile `qwen4b-smoke1-step3` in `profiles.yaml` → `run_episode` on flask answered with bracketed citations (`data/traces/dev/sweqa-flask-001__qwen4b-smoke1-step3.json`); `data/models/manifest.json` has the record. Log `data/smoke_train.log`.
- [2026-09-18 16:45] A7 done — signatures frozen in LOG (16:45 entry); lanes switch from stubs.
- [2026-09-18 23:20] A8 (post-freeze, additive) — agent variants `noindex|nomap|bash|bash_nomap` via `RepoEnv(variant=)` or `CODEQA_AGENT_VARIANT`; `bash` tool with restricted shell + seen-lines grounding (`codeqa/agent/{variants,shell}.py`); 20 tests; default behaviour byte-identical (146 offline tests pass).
- [2026-09-19 00:10] A9 (additive) — Modal executor for the bash variant (`codeqa/agent/modal_shell.py`, `CODEQA_BASH_EXECUTOR=modal`): 4-sandbox pool, 0.7 s/command, 32 concurrent ok; snapshots on volume `codeqa-data`. Base-model variant comparison on `fast` (`scripts/compare_evals.py`): five tools 56 % answered / 51 % found gold file / 50k tokens; bash 44 % / 44 % / 40k; no-index 47 % / 40 % / 27k → index helps the untrained model finish and find; bash is cheaper.
- [2026-09-19 01:20] A10 (additive) — detached Modal runner `apps/trainer/modal_runner.py` + `scripts/modal_sync.sh`; first eval job ran on Modal end to end (LOG 01:20); training smoke on Modal launched.

## Open questions for the lead
- ~~Caps: 150 lines per read, 30 grep hits / 10 files, 20 symbol hits, 60 overview lines.~~ Implemented as the defaults of `codeqa.agent.curation.Caps` (plus 50 list entries, 160 chars per line); change there, tests cover the cap paths.
