# Agent log (append-only)

Append an entry when you learn something another agent would need, measure a number, hit an API quirk, or make a decision your task forced. Newest at the bottom. Keep entries short. Do not edit earlier entries; add a new one that corrects them.

Entry format:

```
## YYYY-MM-DD HH:MM · <component> · <agent or person>
- What: one line
- Detail: a few lines, exact names, numbers, paths
- Affects: which components / contracts
```

## Contract change requests (open)

_None. Add one here if your task needs a schema change, then stop and wait for the lead._

---

## 2026-09-18 13:00 · setup · lead
- What: sync setup complete; all clients proven; one cookbook episode ran end to end
- Detail: see `docs/agents/README.md` "What is already proven" and "Gotchas". Smoke logs in `data/smoke_*.log`.
- Affects: everyone

## 2026-09-18 13:05 · agent · lead
- What: tool declarations must be prepended by the env
- Detail: `renderer.create_conversation_prefix_with_tools([t.to_spec() for t in tools], system_prompt)` before the user message. Two tools' schemas took the prompt from 263 to 740 tokens.
- Affects: agent/env.py, trainer, evals, apps/api

## 2026-09-18 13:05 · grader · lead
- What: step-0 model gets the answer right but the citation format wrong
- Detail: untrained Qwen3.5-4B wrote `src/flask/app.py:L81` in backticks, not `[src/flask/app.py:L81-L85]`. Pass-rate filter must report format_ok and correctness separately.
- Affects: grader, datagen/filter

## 2026-09-18 13:05 · datagen · lead
- What: SWE-QA citation extraction resolves ~65% of paths
- Detail: 46 citations extracted from 10 flask rows; 30 paths existed in the snapshot. Misses are bare filenames without a directory. Resolve bare basenames against the manifest when unique.
- Affects: datagen/sources/import_sweqa.py, grader/citations.py

## 2026-09-18 14:10 · trainer · lane C agent
- What: Tinker re-verified live today; both sampling and training paths work. `data/smoke_tinker.log` and the Tinker section of `data/smoke_clients.log` are stale (402 billing block from before the account was funded); ignore them.
- Detail: `scripts.smoke_episode` ran end to end (740 prompt tokens with 2 tool specs, 2 turns, 9s, 2 tool calls; step-0 model still fails only citation format). `create_lora_training_client_async(base_model="Qwen/Qwen3.5-4B", rank=32)` took 1.2s; `save_weights_and_get_sampling_client_async` took 13.5s (its `name=` kwarg is deprecated and ignored: checkpoints are ephemeral unless saved via the training loop). Versions: tinker 0.29.1, tinker-cookbook 0.5.7.
- Affects: trainer, evals, serving

## 2026-09-18 14:10 · trainer · lane C agent
- What: cookbook RL loop facts the trainer and evaluator must build against (tinker-cookbook 0.5.7)
- Detail: `train.Config` is a `chz` class; required fields `learning_rate`, `dataset_builder: RLDatasetBuilder`, `model_name`, `recipe_name`, `max_tokens`, `log_path`; has `eval_every`, `save_every`, `evaluator_builders: list[SamplingClientEvaluatorBuilder]`, `lora_rank=32`, `remove_constant_reward_groups`, `num_groups_to_log`, `rollout_json_export`, `wandb_project/wandb_name`, `renderer_name`, `max_steps`, `temperature`. No `groups_per_batch` on Config: batch size is the dataset's `get_batch` slice (see `recipes/search_tool/search_env.py`, our closest analogue: `make_envs` returns `group_size` copies of `build_agent_tool_env`). Evaluator interface is `tinker_cookbook.eval.evaluators.SamplingClientEvaluator.__call__(sampling_client) -> dict[str, float]`; metrics land as `eval/<label>/<key>`. Train env metrics land as `env/all/<key>` and `env/<tag>/<key>` from `logging_tags()`; the built-in reward key is `env/all/reward/total`, plus `turns_per_episode`, `ac_tokens_per_turn`, `ob_tokens_per_turn`; anything the `reward_fn` returns in its metrics dict is mean-aggregated under the same prefix, so our `format_ok`, `citations_grounded`, `correctness`, `efficiency`, `tool_calls`, `prompt_tokens` keys come from the grader for free.
- Detail (NaN policy, gap_specs §4): total reward = sum of per-step rewards + `compute_group_rewards` return, so a NaN step reward poisons the sum. Plan: the env reward_fn returns 0.0 with metric `judge_error=1` on judge failure; `compute_group_rewards(trajectory_group, env_group)` then adds the mean of the non-error siblings' rewards to those trajectories, giving exactly the group mean and advantage 0. Groups where every sample errored get 0 everywhere and are dropped by `remove_constant_reward_groups=True`. `group_reward_std` and `unique_tool_sequences_per_group` also go in that hook's metrics dict.
- Detail (termination): `Config.termination=None` resolves to no reward clamp for Qwen models (only Inkling models get grade-then-clamp), so budget/max_turns stops keep our own grader's verdict. `build_agent_tool_env(max_tool_calls=...)` is checked pre-dispatch; env metrics `max_turns` and `max_tool_calls` are emitted on those stops.
- Affects: trainer, evals, agent (make_cookbook_env signature)

## 2026-09-18 14:20 · contracts · lead
- What: three additive fields, all defaulted, no consumer changes required
- Detail: `Message.usage: dict[str,int]` (assistant only; `prompt_tokens`, `completion_tokens`, filled by the Tinker client), `Message.parse_error: str | None` (assistant only; malformed tool call or truncated generation, raw text kept; the driver maps it to `stop_reason=parse_error|overflow`), `EndpointProfile.base_model: str | None` (required when `model` is a `tinker://` checkpoint path; owns tokenizer + renderer). `make_client` now accepts `kind: tinker`.
- Affects: shared/contracts.py, clients/base.py, apps/api profiles, grader (ignore both new Message fields)

## 2026-09-18 14:20 · agent · lead
- What: renderer keeps thinking in history (`strip_thinking_from_history=False`), set once in `clients/tinker.renderer()`
- Detail: the cookbook RL env re-renders the whole history every step; with the default strip the model would lose its own turn-N thinking at turn N+1 and the trajectory loses the sequence-extension property. Product client and training env both call `tk.renderer()`, so they match by construction. Do not construct renderers elsewhere.
- Affects: agent/env.py, trainer, evals, apps/api

## 2026-09-18 14:20 · agent · lead
- What: Qwen3.5 renderer strips string message content, so a tool result's first line loses leading whitespace
- Detail: `"   81 | x"` renders as `"81 | x"` on line one only. Tool outputs must not rely on leading padding; A3 will number lines as `L81 | code` (no leading spaces, same token as the citation syntax).
- Affects: agent/tools.py, agent/curation.py, grader/citations.py (parse both forms defensively)

## 2026-09-18 13:25 · apps/web · product agent
- What: D1 web shell done on the mock stream; Vite SPA instead of a Next.js fork
- Detail: `apps/web` (Vite + React 19 + Tailwind v4 + shadcn/Radix). `pnpm dev` replays `apps/web/mock/events.json`; `VITE_API_URL` switches to the real API. Routes `/` (home, hero replays the recorded episode) and `/app` (workbench). shadcn-admin is Vite + TanStack Router, not Next.js, and the app needs no SSR, so it is a plain SPA over fetch + ReadableStream SSE. `pnpm test`, `pnpm typecheck`, `pnpm lint`, `pnpm build` all pass.
- Affects: apps/api (endpoint shapes the UI expects, see `apps/web/README.md`), D3 profiles

## 2026-09-18 13:25 · apps/web · product agent
- What: API surface the UI expects from apps/api, beyond the lane doc
- Detail: `GET /profiles` → `[{name, kind, model, label?, note?}]`; `GET /repos` rows carry `stage` (`snapshot|index|summaries|ready|error`) plus `files`, `lines`, `symbols`; `GET /repos/{job_id}/status` → `{repo_id, stage, progress: 0..1, seconds, message?}`; `GET /file?repo_id&path` returns the whole file as text (the viewer highlights the range). `citations` event items need `verified`; the UI falls back to the local grounding rule (cited range ⊆ a read_file range) until that event arrives.
- Affects: apps/api, C9 (no schema change; `verified` is already in the contract example)

## 2026-09-18 13:25 · shared · product agent
- What: `CITATION_RE` is mirrored in TypeScript; a test pins parity
- Detail: `apps/web/src/lib/contracts.ts` holds the regex verbatim; `tests/test_web_contracts.py` compares it to `codeqa.shared.contracts.CITATION_RE`. Added `pythonpath = ["."]` to `[tool.pytest.ini_options]` so `tests/` can import `codeqa` (package = false in uv).
- Affects: shared/contracts.py (any regex change must be copied to the TS file or the test fails)

## 2026-09-18 13:25 · apps/web · product agent
- What: react-markdown strips unknown URL schemes
- Detail: citation chips are rendered by rewriting `[path:L10-L20]` to a `cite:` link and overriding `a`. react-markdown's default `urlTransform` drops `cite:` hrefs silently; pass `urlTransform={(u) => u}`.
- Affects: apps/web

## 2026-09-18 14:50 · agent/indexing · lead
- What: A2 done; indexing CLI available for every repo on disk
- Detail: `uv run python -m codeqa.agent.indexing.cli all --repos <file>` (lines `owner/repo@sha`) runs snapshot → index → summarize (Haiku, ~1 call per directory + 1 per file ≥300 lines, cached/resumable in `summaries.json`) → `map.txt` (≤3k Qwen tokens, detail reduced until it fits). Per-repo subcommands: `index`, `summarize`, `map`, `nodoc <repo_id>`. `--fast` skips summaries; `--nodoc` also builds `<repo_id>__nodoc` (docstrings blanked, line numbers preserved, own index + map). Flask: 61 summaries in 10s. **Every repo the env runs on needs `map.txt`** (the prompt embeds it) — lane B: run `map` (or `summarize` then `map`) on the 23 repos you snapshotted; the 8 DeepCodeBench + 15 SWE-QA repos currently have `symbols.json` only. Large repos (django 5.4k files) hit the 400-directory summary cap; that is intended.
- Affects: datagen (B3/B4 need `nodoc`), evals, apps/api (`GET /repos` stage = ready only when `map.txt` exists)

## 2026-09-18 14:50 · contracts · lead
- What: C3 `read_file` line format is `L41 | code` (was `  41 | code`)
- Detail: the renderer strips leading whitespace on a tool result's first line (see 14:20). `L41` is also the exact token used in citations. `docs/contracts.md` and `codeqa/agent/README.md` updated. apps/web mock still shows the old format from `smoke_episode3`; harmless.
- Affects: grader/citations.py (if it parses tool output), apps/web file viewer (parse `^L(\d+) \| ` if it ever reads tool output)

## 2026-09-18 15:10 · datagen · lane B
- What: B1 + B2 done; 23 repos snapshotted and indexed (8 DeepCodeBench + 15 SWE-QA), 433 MB under data/repos, 70 s total
- Detail: `uv run python -m codeqa.datagen.cli import --source all`. DeepCodeBench 912/232 records, paths 98% resolved, required_citations on 92% of tasks (symbol named in the answer -> index span). SWE-QA 720 records, citations 92% resolved (baseline 65%); fixes were unique-basename, unique-suffix, stripped checkout prefix (`workspace/conan/...`), and picking among ambiguous basenames by which file defines a symbol the answer names. Remaining misses: ambiguous basenames with no symbol hint (astropy `core.py`), line numbers past EOF (annotator was on a different commit). reflex and streamlink answers contain no line numbers at all, so those two repos are judge-only for citations.
- Affects: grader (C3 evals), evals

## 2026-09-18 15:10 · datagen · lane B
- What: `required_citations` should be scored as evidence overlap, not a hard gate
- Detail: DeepCodeBench spans are whole symbols (a class can be 150+ lines, e.g. `OpenAIGenericClient` L37-L179); SWE-QA spans are the annotator's lines. A correct answer can legitimately cite other lines of the same symbol. Recommend: credit if any cited range intersects any required span; fabrication is already caught by the files_read grounding gate.
- Affects: grader/citations.py, contracts.md wording

## 2026-09-18 15:10 · datagen · lane B
- What: `symbols.json` on disk is `{"repo_id", "symbols": [...]}`, contracts.md C2 shows a bare list
- Detail: `codeqa.agent.indexing.index.load_symbols` reads the dict form. Anyone reading the file directly should use that helper. Also `tests/fixtures/` is empty (README says mini_repo/index/tasks/traces exist); datagen tests use in-memory manifests instead.
- Affects: contracts.md, agent/indexing, everyone who builds against fixtures

## 2026-09-18 15:15 · datagen · lane B
- What: CodeScout decision: use it, N=15, attribute upstream
- Detail: `OpenHands/SWE-rebench-code-search` has no license tag on HF; it is derived from `nebius/SWE-rebench` (CC-BY-4.0). Content is public GitHub issues + entity locations. Using it for internal training with attribution to SWE-rebench; the Apache-2.0 fallback (`JetBrains-Research/lca-bug-localization`) lacks entity gold. Lead can veto; swapping is a one-module change in `sources/`.
- Affects: datagen, talk (data slide attribution)

## 2026-09-18 15:15 · agent · lead
- What: A3 + A4 done — `RepoTools` and `RepoEnv` are importable; signatures are draft until A7 (expect only additions)
- Detail: `RepoEnv(task, profile)` → `initial_messages() -> list[Message]` (system rules + map + question, client-agnostic), `specs()` (five ToolSpecs for `tools=`), `tools()` (cookbook FunctionTools), `files_read() -> list[Span]`, `tool_calls_made`, `tool_errors`, `make_cookbook_env(reward_fn, *, max_generation_tokens=None, max_trajectory_tokens=None)` where `reward_fn(history, env) -> (float, dict)` (note the extra `env` arg vs the cookbook), `trace_from_history(history, seconds) -> Trace` (C6), `RepoEnv.from_question(repo_id, question, profile)` for ad-hoc product questions. Tool output formats: `find_symbol` → `path:Lstart-Lend  kind  signature  [in Parent]`; `grep` → `path:Lnn: text`; `read_file` → header `path:Ls-Le`, lines `Lnn | code`, footer `(total N lines)` or `(range cut to 150 lines; continue from L151; total N lines)`; errors start with `ERROR not_found|is_directory|is_file|bad_range|bad_pattern`; budget lines `[1 tool call remaining. Answer on your next turn.]` / `[No tool calls remaining. Answer now.]`. Prompt at start on flask ≈ 4.2k tokens (map 2.9k + 5 specs + rules). Tests: `uv run pytest codeqa/agent` (13 + 3).
- Affects: trainer (C2 uses `make_cookbook_env`; grader reads `env.files_read()`), datagen B5/B6, apps/api D2

## 2026-09-18 15:15 · env · lead
- What: ripgrep is not installed on this machine; tools fall back to pure-Python grep automatically
- Detail: the shell `rg` is a Claude Code shim, `shutil.which("rg")` is None. `RepoTools.grep` uses `rg --json` when a real binary exists (`RG_BIN` in `agent/tools.py`), else scans manifest files with `re` (same output format, ~1s on django-sized repos). `brew install ripgrep` is optional.
- Affects: everyone running episodes locally; Modal image (day two) should apt-get ripgrep

## 2026-09-18 15:20 · grader · lane C agent
- What: C1 grader done. `codeqa/grader/` (repo, gates, citations, verifiers, judge, efficiency, grade, cli, README), 56 offline tests in <1s, plus a live Haiku test on 5 hand-graded answers (passes; Haiku is strict on vague items, which is what we want).
- Detail: `grade(task, trace, variant, judge_client, repo) -> GradeResult` (async; `grade_sync` too), `metrics(result, trace, task) -> dict` gives the trainer/evals keys: `reward format_ok citations_parse citations_exist citations_grounded identifier_grounded correctness efficiency judge_error tool_calls tool_errors prompt_tokens completion_tokens answer_tokens redundant_reads turns gate_<name> stop_<reason>`. Judged types = trace|explain with rubric or reference; everything else verifies with zero API calls. `KeywordJudge` is an offline stand-in for tests only. Fixtures: `tests/fixtures/mini_repo/` (20 files), `tests/fixtures/index/symbols.json`, `tests/fixtures/tasks.jsonl` (5, one per type), `tests/fixtures/traces/` (4 good + 11 adversarial); regenerate with `PYTHONPATH=. uv run python tests/fixtures/make_fixtures.py`. Fixture repo_id is `mini__repo__0000001`.
- Detail: end to end proven: `scripts/smoke_grade.py` runs a real Tinker episode on flask with a real SWE-QA task and calls the real grader from inside the cookbook `reward_fn`; result lands in `data/traces/smoke_grade/<task_id>.json`. Step-0 model: 3 turns, 2 calls, 8.2k prompt tokens, fails the citation gate (backticks, no brackets), as before.
- Gotcha: `data/index/<repo_id>/symbols.json` is `{"repo_id", "symbols": [...]}` (index.py), not the bare list shown in docs/contracts.md C2. Grader accepts both; fixture uses the wrapped shape. Docs should be updated to the wrapped shape.
- Gotcha: pytest does not load `.env`; live tests call `load_dotenv()` themselves. `pytest -m live` marker added in pyproject.
- Decision: answer length uses a tokenizer-free proxy `max(words, chars/4)` so the grader carries no model dependency. Efficiency shaping: first half of budget free, then linear 1.0 → 0.5 at 100% of budget; prompt-token budget = max_tool_calls × 3000. Redundant reads (a span already fully covered) count as extra calls under `multiplicative`.
- Affects: trainer, evals, datagen/filter (use `grade` + `metrics`), apps/api (use `check_citations` for verified badges)

## 2026-09-18 15:40 · agent · lead
- What: A5 done — `run_episode` works with Tinker and Anthropic clients; `profiles.yaml` exists (D3 names agreed by fiat: `claude`, `haiku`, `qwen4b-base`, trained = `qwen4b-<run>-step<N>`)
- Detail: `codeqa.agent.driver.run_episode(env, client, on_event=None, *, temperature=1.0, max_tokens=None) -> Trace`; `on_event` gets `SSEEvent(type, payload)` per C9 (sync or async callback); `save_trace(trace, run) -> path`. `codeqa.shared.profiles.get_profile(name)` / `load_profiles()` / `add_profile(profile)` read/write `profiles.yaml`. The driver emits a light `citations` event (`exists`, `verified` = range inside a read range); the grader's `check_citations` stays the authority for grading. Numbers on one SWE-QA flask task: base Qwen 6 turns / 5 calls / 46.8k cumulative prompt tokens / 0 bracketed citations; Claude Sonnet 5: 6 turns / 5 calls / 49.8k / 3 verified citations. Cumulative prompt tokens per episode ≈ 10× the initial prompt: that is the run-two efficiency headroom.
- Detail: Claude once emitted a malformed tool arg (`start: "150, \"end\": 330}"`); the cookbook validation error came back as a normal tool message and it recovered next turn. Keep validation errors as tool messages, never exceptions.
- Affects: apps/api D2 (import `run_episode`, `RepoEnv.from_question`, `get_profile`), datagen B5, evals C3

## 2026-09-18 15:40 · contracts · lead
- What: docs/contracts.md C2 now shows the wrapped `symbols.json` shape; C8 now mirrors `profiles.yaml` incl. `base_model` for checkpoints
- Affects: nobody's code; doc parity only

## 2026-09-18 14:05 · apps/web · product agent
- What: home page redesigned; four design skills installed at repo level
- Detail: `npx skills add mengto/skills@{product-proof-saas,light-mode-paper-technical,beautiful-shadows} 101-skills/superpowers@landing-page-design` → `.agents/skills/` + `.claude/skills/` symlinks. Direction and motion notes in `apps/web/README.md`. `mengto/skills` has ~90 web-design direction skills worth reusing for the talk slides or eval plots.
- Affects: apps/web; anyone styling docs/slides

## 2026-09-18 14:05 · apps/web · product agent
- What: streaming UI must not resize its container as rows arrive
- Detail: the hero stage grew row by row and shifted the whole layout. Fixed by a fixed-height scroll container that scrolls to the newest row (`StageBody` in `pages/home.tsx`). Apply the same rule to any embedded live ledger.
- Affects: apps/web

## 2026-09-18 16:45 · agent · lead
- What: A6 done — the whole chain works once: tasks → RepoEnv → cookbook loop → checkpoint → sample from it → cited answer. A7: signatures below are FROZEN; additions only from here.
- Detail (numbers, `scripts/smoke_train.py smoke1`, 10 graphiti tasks × group 4 × 3 steps, lr 4.9e-4 from `hyperparam_utils.get_lr`, LoRA 32, real grader, no judge calls since all tasks were verifiable): reward 0.10 → 0.32 → 0.37; format_ok 0.475 → 0.875; citations_grounded 0.125 → 0.45; tool calls/episode 6.0 → 2.6; turns 5.05 → 3.55; groups with mixed reward 0.4 → 0.8. 35 s per step for 40 episodes. Same 10 tasks every step (wrap-around), so treat the curve as a plumbing proof, not a result. The step-3 checkpoint answered an unseen flask question with `[path:Lx-Ly]` citations (`data/traces/dev/sweqa-flask-001__qwen4b-smoke1-step3.json`). Files: `data/logs/smoke1/`, `data/models/manifest.json`, `profiles.yaml` entry `qwen4b-smoke1-step3`.
- Frozen API:
  - `codeqa.agent.env.RepoEnv(task: Task, profile: EndpointProfile, caps=DEFAULT_CAPS)`; `.initial_messages() -> list[Message]`; `.specs() -> list[ToolSpec]`; `.tools() -> list[FunctionTool]`; `.files_read() -> list[Span]`; `.tool_calls_made`, `.tool_errors`; `.budget: Budget`; `.make_cookbook_env(reward_fn, *, max_generation_tokens=None, max_trajectory_tokens=None)` with `async reward_fn(history: list[cookbook Message], env: RepoEnv) -> (float, dict[str, float])`; `.trace_from_history(history, seconds=0.0) -> Trace`; `RepoEnv.from_question(repo_id, question, profile, task_type="explain", budget=None)`.
  - `codeqa.agent.driver.run_episode(env, client, on_event=None, *, temperature=1.0, max_tokens=None) -> Trace`; `save_trace(trace, run="dev") -> Path`. Events per C9 as `SSEEvent(type, payload)`.
  - `codeqa.clients.base.make_client(profile) -> ModelClient` for kinds `anthropic | openai | tinker`; `codeqa.clients.tinker.TinkerChatClient(profile).chat(messages, tools, max_tokens, temperature) -> Message` (fills `usage`, `parse_error`); `tk.renderer(base_model)` is the only renderer constructor.
  - `codeqa.shared.profiles.get_profile(name) / load_profiles() / add_profile(profile)`; checkpoint profiles are named `qwen4b-<run>-step<N>` with `model=tinker://…/sampler_weights/<N|final>` and `base_model=Qwen/Qwen3.5-4B`.
  - Trainer skeleton to lift into `codeqa/trainer` (C2): `TaskGroupBuilder`, `TaskDataset`, `SmokeDatasetBuilder` and the `train.Config(...)` call in `scripts/smoke_train.py`. `compute_group_rewards` there already applies `trainer/group_rewards.fill_judge_errors` + `group_metrics`.
- Gotchas for C2/C3:
  - `env/all/prompt_tokens` is 0 during training: the cookbook history carries no token counts, so `trace_from_history` cannot fill `stats.prompt_tokens`. Token counts ARE available in `compute_group_rewards(trajectory_group, env_group)` via `trajectory.transitions[i].ob.length` / `.ac`. For run two, apply the efficiency multiplier in that hook (return `correctness × eff − correctness` as the group delta) or use the cookbook's own `env/all/ob_tokens_per_turn`, `total_ob_tokens`, `ac_tokens_per_turn` for reporting. `tool_calls` and `turns` are exact.
  - `save_every=1` wrote checkpoints `000001..000003` plus `final`; each sampler save costs ~10 s. Use `save_every=10` for real runs.
  - Smoke ran with `remove_constant_reward_groups=False` so a training step always happens; real runs use `True` (60 % of step-0 groups were all-zero).
  - `Config.max_tokens` is per-turn generation (1024 was enough: ~200 tokens/turn average, thinking included). `max_trajectory_tokens` = `profile.max_context` (32 768).
  - `num_groups_to_log=2` prints whole rollouts into `logs.log` (tool outputs included); `iteration_N/train.html` is the readable page.
  - The evaluator for `eval_every` is not written; `run_episode` + `grade` over `data/tasks/eval/fast.jsonl` is the intended body (C3).
- Affects: C2 (lift the builder), C3, B5/B6 (use `run_episode`, `qwen4b-base`), D2 (`RepoEnv.from_question`, `run_episode`, `get_profile`; `GET /profiles` can list `profiles.yaml`; `qwen4b-smoke1-step3` is a usable trained profile for the model switcher today)

## 2026-09-18 17:05 · datagen · lane B
- What: B3 done (1,017 CodeScout tasks, 15 repos) and B4 half done (480 structural tasks, 8 repos); 15 more training repos exist under data/repos
- Detail: CodeScout repos = top 15 by rows with entity gold after excluding all 15 SWE-QA-Bench repos and the 8 DeepCodeBench repos: sqlglot, dvc, Pillow, dask, geopandas, ignite, pydicom, nilearn, PyBaMM, networkx, pre-commit, textual, pennylane, pyupgrade, borg (`data/repo_list_codescout.txt`, one commit each). sqlglot alone is 254 tasks (25%); B6 should cap per repo (~120) when building train/all.jsonl. dvc/pyupgrade/borg lose half their rows because gold entities moved between the row's commit and the chosen one. Haiku rewrites: 1,240 calls, ~$1, 3 min at concurrency 8; cache `data/cache/rewrites/codescout.jsonl`. Structural: 60/repo balanced across locate/value/enumerate/trace, locate questions paraphrased from docstrings by Haiku with the name withheld (buffer 3x, ~20% rejected for leaking), tasks point at `<repo_id>__nodoc`. Class evidence spans are clipped to 40 lines everywhere (`resolve.symbol_span`).
- Gotchas: datasets-server rate-limits at ~50 pages/min (429 with Retry-After); `clients/hf.py` now honours it and `datagen/cache.py` resumes a partial pull. tree-sitter (this pack version) emits module-level `assignment` nodes directly under `module`, not wrapped in `expression_statement`; handle both. `[map]` runs for large repos print harmless `Event loop is closed` tracebacks from httpx at shutdown.
- Affects: grader (expected_symbols for CodeScout are `path:Class.method` after dropping the bare class when a method is also gold), trainer (B6 per-repo cap), evals

## 2026-09-18 16:40 · trainer · lane C agent
- What: C2 trainer runs end to end on real Tinker. `codeqa/trainer/` (dataset_builder, group_rewards, heldout_evaluator, config, run). Smoke: 10 SWE-QA flask tasks × group 4 × 3 steps with the real grader and Haiku judge, in-loop eval on 2 tasks every step; `data/logs/smoke3/metrics.jsonl` has `env/all/{reward,format_ok,citations_grounded,correctness,efficiency,tool_calls,prompt_tokens,...}`, `env/all/group_reward_std`, `env/all/unique_tool_sequences_per_group`, and `eval/fast/env/all/*` at steps 0, 1, 2; `checkpoints.jsonl` has tinker:// sampler paths. ~2 min per step at this size.
- Detail (how it is wired): the env reward_fn only captures the history; `CodeQAGroupBuilder.compute_group_rewards` builds C6 traces (`RepoEnv.trace_from_history` + token counts from the trajectory), grades the group concurrently (judge semaphore 16), applies NaN→group-mean, and attaches every grader metric per trajectory. Eval metric prefix is `eval/fast/env/all/<key>` (the cookbook prepends the evaluator name to its own `env/all/` keys). `cookbook.metric_util.dict_mean` averages a key over only the rows that have it, so every metric must be emitted on every row (fixed for `stop_<reason>`).
- Detail (CLI): `uv run python -u -m codeqa.trainer.run --tasks <C5 jsonl> --profile qwen4b-base --run-name <run> [--steps N --epochs E --group-size 8 --groups-per-batch 32 --lr ... --variant ... --eval-tasks <jsonl> --eval-every 10 --eval-max-tasks N --judge-model ... --if-exists delete|resume]`. `--epochs` is needed when the task file is smaller than steps × groups_per_batch (a 10-task file is one batch).
- **Finding, needs a lead decision (reward + lr):** with the cookbook's recommended LoRA lr (`get_lr` → 4.9e-4) the policy collapsed after ONE step: step 0 reward 0.15 (2/40 rollouts > 0, only 2 groups with variance after `remove_constant_reward_groups`), steps 1–2 reward 0.0 with std 0: every rollout burned the full 12-call budget and never answered (stop=budget, format gate). Two causes stack: (1) on these explain tasks the untrained model already hits the budget in 27/40 rollouts, so the batch is nearly all zeros and the update comes from 8 samples; (2) "never answer" and "answer with bad citations" both score 0, so nothing pushes toward answering. Control run at lr 1e-4 in progress (`smoke_lr1e4`). Proposal: (a) lr 1e-4 or lower for run one; (b) a small negative reward (−0.1) when the episode ends with no final answer (stop=budget|max_turns), keeping format failure at 0 so answering badly still beats not answering; (c) start run one on locate/value tasks (6-call budgets, verifiable) where step-0 variance exists, not on SWE-QA explain. (b) is a reward change → decisions.md.
- Affects: trainer, lead (run-one kickoff), datagen (pass-rate filter should report stop=budget rate per task type)

## 2026-09-18 16:40 · evals · lane C agent
- What: C3 evals done. `codeqa/evals/` (run, report, plots, sweqa_judge). `uv run python -u -m codeqa.evals.run --profile qwen4b-base --tasks data/tasks/eval/smoke_sweqa_flask.jsonl` runs lane A's `run_episode` with any profile kind, grades, writes `data/evals/<profile>/<set>/{per_task.jsonl,results.json,traces/}`, and fills `CheckpointRecord.evals[<set>]` in `data/models/manifest.json` for the record whose `profile` matches. `evals.plots --run <run>` renders `headline.png` (reward, correctness, citation validity, tool calls per correct) and `all_curves.png` from metrics.jsonl; `--set <set> --profiles ...` renders eval bars. `evals.report --set <set>` prints headline-by-profile and source×type tables (`--markdown`).
- Detail: SWE-QA five-dimension judge (`evals.sweqa_judge`, Sonnet 5) works: base Qwen3.5-4B on 3 flask tasks = 66/100 (correctness 12.3, completeness 11.3, relevance 15, clarity 15, reasoning 12.3). Never a training signal.
- Gotcha: Sonnet 5 thinks adaptively by default and spent the whole `max_tokens` on thinking (empty reply). `clients/anthropic.py` now sends `thinking={"type": "disabled"}` when `profile.thinking` is False (verified accepted by Sonnet 5 and Haiku 4.5). Judge profiles set `thinking=False`.
- Gotcha: `summary.answer_tokens` now uses the Qwen tokenizer (proxy fallback); the untrained model writes 430–490-token answers against a 450 cap on explain tasks, so ~2/3 of its answers fail the length gate at temperature 0.2. Consider raising explain's `max_answer_tokens` or accept it as the pressure it is (lead call; affects `DEFAULT_BUDGETS`).
- Affects: talk (plots), serving (manifest), lead

## 2026-09-18 16:45 · handoff · lane C agent (C5)
- Metric keys (trainer, `data/logs/<run>/metrics.jsonl`, one row per step): `env/all/<k>` and `env/<source>/<k>`, `env/<task_type>/<k>` for k in `reward format_ok citations_parse citations_exist citations_grounded identifier_grounded correctness efficiency judge_error tool_calls tool_errors prompt_tokens completion_tokens answer_tokens redundant_reads turns gate_{format,citations,grounding,budget,judge_error} stop_{answer,max_turns,budget,overflow,parse_error,error} group_reward_std group_reward_mean unique_tool_sequences_per_group judge_error_rate group_all_judge_errors`; cookbook extras `env/all/reward/total turns_per_episode ac_tokens_per_turn ob_tokens_per_turn`, `optim/lr optim/entropy optim/kl_sample_train_v1|v2`, `progress/batch`. Held-out: the same under `eval/fast/env/all/`.
- Eval summary keys (`data/evals/<profile>/<set>/results.json` → `summary`, also written into the manifest): `n reward correctness format_ok citations_parse citations_exist citations_grounded identifier_grounded efficiency judge_error tool_calls tool_errors prompt_tokens completion_tokens answer_tokens turns seconds correct_rate citation_valid tool_calls_per_correct prompt_tokens_per_correct stop_<reason> gate_<name>` (+ `sweqa_total sweqa_correctness` after `evals.sweqa_judge`).
- Manifest: `data/models/manifest.json` is a JSON list of `CheckpointRecord`; `evals.run` matches on `record.profile == --profile` and sets `record.evals[<set>] = summary` (finite numbers only). Serving should write the record (name, run, step, tinker_path, profile `qwen4b-<run>-step<N>`) and `add_profile` the tinker profile (`kind: tinker, model: tinker://..., base_model: Qwen/Qwen3.5-4B, renderer: qwen3_5`) so `evals.run --profile qwen4b-<run>-step<N>` works with no other glue.

## 2026-09-18 17:30 · trainer · lane C agent (corrects 16:40)
- What: control run at lr 1e-4 (`data/logs/smoke_lr1e4`, same 10 SWE-QA explain tasks × 4 × 3 steps, `remove_constant_reward_groups=True`) does NOT collapse: reward 0.18 → 0.25 → 0.28, group_reward_std ~0.3 throughout, unique tool sequences 3.7 → 4.0, stop=budget 8 % → 0 %, tool calls 5.0 → 6.7, answer tokens 323 → 118. Format failures rose (gate_format 0.25 → 0.5) because episodes now end at max_turns (0 → 42 %) or parse errors (6–8 %) instead of at the budget: the model is exploring more turns, not refusing to answer.
- Correction to 16:40: "27/40 rollouts hit the budget at step 0" was wrong (a metrics bug averaged `stop_*` over rows that had the key; fixed). Step-0 stop=budget was 8 %. The collapse at lr 4.9e-4 came from an update built on the only 2 groups (8 rollouts) that survived `remove_constant_reward_groups` on judged explain tasks, which the lead's graphiti smoke (verifiable tasks, 40 % mixed groups at step 0, `remove_constant_reward_groups=False`) never hit.
- Recommendation for run one: lr 1e-4 (not `get_lr`'s 4.9e-4) and a task mix where step-0 groups have variance (locate/value/CodeScout first, judged types after the pass-rate filter). The "no final answer → −0.1" shaping stays a proposal for the lead; not implemented. `codeqa/trainer` fills `prompt_tokens` from the trajectory in `compute_group_rewards`, so the 16:45 gotcha about `env/all/prompt_tokens = 0` does not apply there; `scripts/smoke_train.py` can switch to `codeqa.trainer.run` when convenient (same builder shape, plus epochs, evaluator, variant, judge wiring).
- Affects: lead (run-one config), trainer

## 2026-09-18 18:25 · datagen · lane B
- What: B4 done (1,328 structural tasks, 23 repos, all `__nodoc`); B5 teacher running; B6 code in place
- Detail: raw totals now: deepcodebench 912, codescout 1,017, structural 1,328, teacher (running, ~30/repo x 23 seeds, 90%+ kept so far). All 23 training repos + 15 eval repos have `symbols.json`, `summaries.json`, `map.txt`; the 23 training repos also have `__nodoc` variants (`data/repo_list_{dcb,codescout,sweqa}.txt`). Teacher loop = Sonnet with the real `RepoTools` (14-call cap) from a seed file -> JSON task; blind Haiku through `RepoEnv` + `run_episode`; `grader.judge` on the rubric; keep at >= 0.5. Smoke: 3/4 kept, $0.11 per kept task, ~30 s per attempt. B6 `filter` samples the student through `RepoEnv` and records format_ok / grounded / ungated correctness / correctness-given-format per task in `data/tasks/reports/passrate.jsonl`; `split` applies the window and a per-repo cap of 120 and writes `train/all.jsonl`, `eval/fast.jsonl`, `data/tasks/README.md`.
- Affects: C2 (train/all.jsonl arrives after the pass-rate run), C3 (`eval/fast.jsonl`), D (repos with maps = all 38)

## 2026-09-18 18:40 · decisions · lead
- What: the five run-one decisions are in `docs/decisions.md` (evening entry): lr 1e-4; no-answer penalty −0.1 adopted (lane C implements in the group hook); Haiku in-loop, Sonnet only for the final SWE-QA number; run-two formula accepted as implemented; B6 at two samples per task. Run one starts on the verifiable raw tasks and does not wait for B6. The lead launches training runs; lanes do not.
- Affects: C (implement the penalty; hand the lead the run command), B (B6 scale), D (nothing)

## 2026-09-18 15:10 · apps/web · product agent
- What: streaming episode display settled on a chat transcript; ledger stays for the workbench
- Detail: tried vertical append (hero resized), fixed-height scroll (blank at start), in-place row swap (rows flipped, confusing), horizontal strip (rejected). What worked: `components/chat/transcript.tsx`, a fixed-height chat viewport pinned to the bottom, activity rows collapse to a summary when the answer arrives. User preference: chat-shaped, readable top-down, hero must fit one viewport. Consider the same component for the workbench's main pane.
- Affects: apps/web, D2 API consumers of C9 (no contract change)

## 2026-09-18 18:50 · evals · lead
- What: `data/tasks/eval/fast.jsonl` exists (60 DeepCodeBench test + 60 SWE-QA, seed 0, round-robin over repos) so run one's in-loop eval uses the real held-out set from step 0
- Detail: lane B's `split` may overwrite it with the same recipe; keep it 120 tasks and seeded so curves stay comparable across runs. `run1_verifiable.jsonl` (2,521 verifiable raw tasks, 46 snapshots) is the run-one training file.
- Affects: C (`--eval-tasks data/tasks/eval/fast.jsonl`), B (split recipe)

## 2026-09-18 18:50 · datagen · lane B
- What: **Anthropic credit balance exhausted** around 18:35; every Anthropic call now returns 400 `Your credit balance is too low`
- Detail: hit during the B5 teacher run after 81 attempts (68 tasks kept, `raw/teacher.jsonl`); the remaining 561 seeds failed instantly and were purged from `data/tasks/reports/teacher_attempts.jsonl` so `uv run python -m codeqa.datagen.cli teach` resumes them once credits are back (~$60 more at $0.14/kept). Blocked until then: teacher, the judge (grader returns NaN after 3 retries, so judged tasks in training get the group mean), summaries for any new repo. Not blocked: Tinker sampling. B6 pass-rate measurement is running on the verifiable sources only (structural + codescout, 2 samples, two processes) into `data/tasks/reports/passrate.jsonl`; deepcodebench + teacher need the judge and will be measured after a top-up.
- Measured meanwhile (`qwen4b-base`, 32 DeepCodeBench tasks x 2): answered 45%, format_ok 30%, cited-and-grounded 9%, found the gold lines 48%, content-correct 15%, strict reward 3%, 8.0 tool calls. Typical failure: greps with invented identifiers (`KoCache`) or finds the right file and keeps reading until max_turns. Tinker sampling throughput ~0.3 episodes/s regardless of concurrency 8-32 (likely client-side rendering CPU), hence two processes.
- Affects: everyone using Anthropic (B5, C judge, D api with `claude` profile); lead: top up credits

## 2026-09-18 19:30 · trainer · lane C agent
- What: no-answer penalty landed (decisions #2). `trainer/group_rewards.no_answer_penalty(stop_reason, gate_failed)` → −0.1 when `stop ∈ {budget, max_turns}` and the format gate fired (i.e. no final answer); applied in `CodeQAGroupBuilder.compute_group_rewards` before the NaN→group-mean fill. Tests: `trainer/tests/test_group_rewards.py::test_no_answer_penalty_only_for_stalled_episodes` and `trainer/tests/test_penalty_fixture.py` (stalled fixture → −0.1, padded answer → 0, good → 1, advantage order stalled < bad < good). New metrics `env/all/reward_shaped` (trained total) and `env/all/no_answer_penalty` (rate); `env/all/reward` stays the grader's reward. A run launched before this entry does not have it; the lead decides on a restart.
- Detail (answers to the lead's two questions): (1) step time is **sampling-bound**: in `smoke3`/`smoke_lr1e4` `time/policy_sample` summed 530–1090 s per step across ~300 samples (mean 2–3.5 s, max 8–13 s) while `time/compute_group_rewards` (grader + Haiku) totalled 0–11 s per step (<5 %); wall time per step 48–54 s for 40 episodes + 11 s train/save. Groups roll out concurrently, so 128 episodes/step should cost roughly the slowest group (~40 s) plus train if Tinker sampling keeps up, else up to ~3×; judge stays negligible (128 calls behind a semaphore of 16 ≈ 10–15 s worst case). (2) `remove_constant_reward_groups` never returns an empty list: when every group is uniform it logs "All rewards are uniform. There will be no gradient" and returns the first group, so the step runs with zero advantages (a wasted step, no crash). The separate "all groups failed or filtered, skipping batch" path covers rollout errors. The collapse was not an empty batch; it was an update from the 2 surviving groups at lr 4.9e-4.
- Detail (run two): `--load-checkpoint` wants the **state path** (`checkpoints.jsonl` → `state_path`, `tinker://…/weights/<step>`); the cookbook loads it with `create_training_client_from_state_async` (weights only, fresh optimizer). Command in `codeqa/trainer/README.md` ("Run two"): same task file, `--variant multiplicative --load-checkpoint <state_path> --lr 1e-4`.
- Detail (monitor): `uv run python -m codeqa.evals.monitor --run run1` → per-step train table, `eval/fast` table, collapse checks (std<0.05, unique sequences ≤1.5, stall rate >50 % or +15 pts over step 0, format gate spike, reward falling while calls rise, judge errors, 4-step dead run) and `plots/headline.png`, `plots/all_curves.png`. Verified: flags smoke3 (3 warnings), passes smoke_lr1e4.
- Detail (Sonnet judge cost): measured 3,207 input + 44 output tokens per task (the SWE-QA prompt is ~2.9k of it) → at $2/$10 per MTok ≈ $0.0068 per task, ≈ $0.68 per 100 tasks, ≈ $4.9 for all 720 per profile. Cheap enough to run the full 720 for each profile in the table.
- Affects: lead (restart decision, run-two launch), talk

## 2026-09-18 19:40 · apps/api · product agent
- What: D2 done — `apps/api` streams real episodes to the web UI with `claude` and `qwen4b-smoke1-step3`
- Detail: `uv run uvicorn apps.api.server:app --port 8000`; endpoints and numbers in `apps/api/README.md`. `/ask` drops the driver's `citations` + `done` and emits the grader's `check_citations` result then `done`, so the UI badges are authoritative. One `make_client` per profile, cached; Tinker warm-up 3.9 s. Traces land in `data/traces/product/`. `POST /repos` fast-indexes (ready after snapshot+index+map, summaries fill in and the map is rebuilt): itsdangerous ready in 6.3 s. Added `fastapi`, `uvicorn` to pyproject; `apps/api/tests` (7 tests, no network) added to pytest testpaths.
- Gotcha: a sync `def` FastAPI endpoint runs in a threadpool, so `asyncio.create_task` there raises "no running event loop". Anything that spawns background work must be `async def`.
- Gotcha: Claude sometimes writes `[path:L114-L120, L40-L45]`; `CITATION_RE` only catches the first range. The grader is the authority, so the UI shows it as one citation. Worth a note for B (teacher) and the grader's format gate.
- Affects: apps/web (set `VITE_API_URL`), D4, lead (demo)

## 2026-09-18 20:05 · apps/web · product agent
- What: D4 done — `docs/demo.md` and a `/compare` page (same question, two profiles, side by side)
- Detail: measured base vs `qwen4b-smoke1-step3` on the flask locate question through the UI: base 3 calls / 21.4k prompt tokens / 14.1 s / no bracketed citations; step-3 1 call / 8.5k / 12.1 s / one bracketed citation, unverified (it cited L81 without reading it and guessed "around line 18-20" for App). Good talk material for "format learned first, grounding next"; swap in the run-one profile before the talk.
- Affects: lead (talk), D5

## 2026-09-18 19:40 · datagen · lane B
- What: interim step-0 pass rates of `qwen4b-base` by source (2 samples/task, real RepoEnv + run_episode + grader), useful for choosing what run one trains on
- Detail (running means, first 100-300 tasks per source): structural: answered 62%, format_ok 46%, found gold lines 65%, content-correct 55%, strict reward 29%. codescout: answered 40%, format_ok 10%, found 53%, content-correct 20%, strict 1%. deepcodebench: answered 46%, format_ok 23%, found 54%, content-correct 33%, strict 13%. Reading: structural is the only source where the base model already earns reward often enough for GRPO groups to have variance at step 0; codescout is found-but-not-answered (the model reads the right file and runs out of turns, or answers without the `[path:Lx-Ly]` format). Recommendation for run one: mix heavy on structural + deepcodebench for the first steps, codescout ramps once format is learned. Two samples quantize the score to {0, 0.5, 1}; a refine pass adds two more samples to the tasks at 0 or 1 (`cli filter --refine`). Tinker sampling saturates at ~1-2 episodes/s across all local processes.
- Credits: teacher resumed at 10 seeds/repo with a $24 cap (`--max-cost`); 80% of attempts kept; expected ~240 teacher tasks total.
- Affects: C2 (dataset mix for run one), C3 (baseline numbers to plot against)

## 2026-09-18 20:20 · trainer · lead
- What: run-one training file is `data/tasks/train/run1.jsonl` (1,841 tasks): all structural (1,328), DeepCodeBench verifiable enumerate/locate/value (176), CodeScout trace (337). CodeScout locate is held back for run two (step-0 strict reward ≈ 1 %, format_ok 6 % per lane B's pass rates); judged types wait for B6.
- Detail: `run1_verifiable.jsonl` (2,521) is the superset. The trainer shuffles with `--seed`, so file order is irrelevant. When run one launches, the three `cli filter` processes will be paused so Tinker throughput goes to training; `filter` resumes from `reports/passrate.jsonl` (504 rows measured so far).
- Affects: C (run-one command uses run1.jsonl), B (filter pause/resume)

## 2026-09-18 20:10 · account · lane C agent — ACTION NEEDED
- What: the Anthropic account returned `400 invalid_request_error: Your credit balance is too low to access the Anthropic API` during the base-model fast baseline (one Haiku judge call → NaN, task `dcb-3c01df1e`), and the first Claude fast baseline failed all 120 episodes in 15 s (rows show `stop=error`), which is consistent with the same error. Later probes succeeded, so the balance is hovering near zero or auto-reloading. Run one's Haiku judge and every Claude eval depend on it; a dry balance mid-run turns judged rewards into NaN (group mean, no signal) and Claude baselines into zeros.
- Ask: top up the Anthropic balance (or raise the auto-reload) before run one and the SWE-QA baselines. Expected spend: Haiku judge ≈ $0.003/call (~1.3k in + 100 out), 128 calls/step × 50 steps ≈ $20 for run one; Sonnet SWE-QA judge ≈ $0.007/task; Claude teacher episodes ≈ 60k input tokens each ≈ $0.15/episode at Sonnet rates.
- Affects: everyone using `clients/anthropic.py`

## 2026-09-18 20:20 · clients/anthropic · lane C agent
- What: Claude episodes that exhaust the tool budget died with `400 … unexpected tool_use_id found in tool_result blocks: call_7_budget` (1 of 86 fast-set episodes, `sweqa-reflex-030`). `driver.run_episode` appends the budget notice as a `role=tool` message with `call_id=call_<turn>_budget`, and `_to_anthropic` turned it into a `tool_result` that answers no `tool_use`.
- Fix: `clients/anthropic.py::_to_anthropic` now emits a tool message whose `call_id` matches no prior tool call as a plain user text block (real tool results unchanged). Lane A may prefer to make the budget notice a user message in the driver; either way the client is now tolerant.
- Affects: apps/api (Claude profile), datagen teacher, evals

## 2026-09-18 21:00 · evals · lane C agent — baselines (left columns of the talk table)
- What: `data/tasks/eval/fast.jsonl` (120: 60 DeepCodeBench test + 60 SWE-QA), temperature 0.2, Haiku judge, `--set fast`. Results in `data/evals/{qwen4b-base,claude}/fast/`; `uv run python -m codeqa.evals.report --set fast --markdown` prints the table; `data/evals/plots/eval_fast.png` is the bar figure.

| profile | reward | correct_rate | format_ok | citation_valid | tool_calls | tool_calls/correct | stop=answer | n |
|---|---|---|---|---|---|---|---|---|
| claude (Sonnet 5) | 0.199 | 0.217 | 0.317 | 0.242 | 5.15 | 23.8 | 0.94 | 120 |
| qwen4b-base | 0.000 | 0.000 | 0.258 | 0.017 | 7.66 | – | 0.56 | 120 |

  By source × type, claude: deepcodebench/explain reward 0.38 (n=49), deepcodebench/enumerate 0.36 (11), sweqa/locate 0.08 (14), **sweqa/explain 0.00 (46)**. qwen4b-base: 0.00 everywhere; 44 % of its episodes end without an answer (max_turns 29 %, budget 15 %), 74 % fail the format gate, 2/120 pass all gates.
- Base on the SWE-QA 100 sample (`data/tasks/eval/sweqa_100.jsonl`, seed 0, round-robin over the 15 repos): reward 0.008, format_ok 0.05, 95 % format gate. Sonnet five-dimension judge running on it; Claude's 100 episodes running (concurrency 3 to stay under rate limits; ~15 s each).
- **Finding for the lead (answer cap):** Claude answers 42/46 SWE-QA explain tasks and every one of its 53 over-cap SWE-QA answers carries bracket citations, but they are 500–1,060 tokens against `max_answer_tokens=450` (explain) / 250 (locate). Claude answer length: SWE-QA p50 671, p80 886, p90 968; DeepCodeBench p50 415, p80 624. Pass rate at cap 450 / 600 / 800 / 1000: SWE-QA 16 / 42 / 71 / 95 %; DeepCodeBench 64 / 78 / 98 / 100 %. The cap is in the prompt ("keep it under N tokens"), so it must be fixed before run one and identical at inference. Options: (a) raise explain to ~800 and locate to ~400 in `DEFAULT_BUDGETS` (teacher passes 71 % / 98 %; costs ~350 more sampled tokens per answer); (b) keep 450 and accept that the in-loop `eval/fast` SWE-QA half stays near 0 under our reward, using the Sonnet SWE-QA judge (no length gate) as the external number. My recommendation: (a), because the teacher data for judged types otherwise fails our own gate and B5's teacher traces would be mostly unusable.
- Grader change: a task of a verifiable type with no gold but a rubric/reference now goes to the judge instead of scoring "nothing to verify" (3 such tasks in fast: two DeepCodeBench enumerate rows with facts but no paths, one SWE-QA locate with neither → still 0). Lane B: `dcb-f4ac8a62`, `dcb-d58dd0e8` need `expected_paths`, `sweqa-streamlink-040` has no gold at all.
- Affects: lead (cap decision before run one), datagen (B5 teacher, task gold), talk

## 2026-09-18 20:40 · apps/web · product agent
- What: workbench rail is now a repo switcher dropdown (shadcn-admin team-switcher shape) over a saved-conversations list; idle state centered
- Detail: `components/repo/repo-switcher.tsx` (dropdown + "Add a repository" dialog), `components/repo/conversations.tsx`, `lib/history.ts` (localStorage, per browser, 100 max; reopening a conversation restores the full episode incl. verdicts). If history should be shared or server-side, `GET /conversations` can be built from `data/traces/product/` traces; the UI store is one module to swap.
- Affects: apps/web, D5 (nothing)

## 2026-09-18 21:00 · apps/web, apps/api · product agent
- What: per-repo sample questions from the index; composer hides once a question is asked; default model is `qwen4b-base`; favicon
- Detail: `GET /repos/{repo_id}/suggestions` → three questions (locate: class with most methods; trace: largest in-package function; explain: second class), generic fallback when the index is missing. Mock keeps the flask set. Workbench shows Stop while running and "Ask another" after; no follow-ups (one trajectory per question, by design).
- Affects: apps/web, apps/api

## 2026-09-18 21:20 · contracts · lead
- What: answer caps raised (decisions.md #6): locate 400, value 300, enumerate 500, trace 600, explain 800 (`DEFAULT_BUDGETS` in `shared/contracts.py`). Task files carry `budget: null`, so nothing on disk needs rebuilding; the prompt picks the new cap up at runtime.
- Detail: lane C — please re-run `evals.run` for `qwen4b-base` and `claude` on `fast` under the new caps before the table goes in the talk. Lane B — pass-rate rows for judged types were measured under the old caps; verifiable rows are unaffected (base answers are short).
- Affects: C (baselines), B (B6 judged-type rows), D (nothing; product reads budgets from the task type)

## 2026-09-18 21:20 · agent/driver · lead
- What: the driver's "tool calls dropped, answer now" notice is now a `role=user` message (was a `role=tool` message with a synthetic `call_id`, which Anthropic rejects as an orphan `tool_result`). Lane C's tolerance in `_to_anthropic` (20:20) can stay as belt and braces.
- Affects: apps/api, datagen teacher, evals (Claude profile)

## 2026-09-18 21:25 · grader/fixtures · lead
- What: `tests/fixtures/make_fixtures.py` "verbatim" case now pastes the two mini-repo files three times (~2k tokens) so it still exceeds the explain cap after the caps were raised to 800; fixtures regenerated. No grader logic touched.
- Affects: C (fixture regenerated under your folder; regenerate again if you change the generator)

## 2026-09-18 21:40 · evals · lane C agent — SWE-QA external number (Sonnet 5 five-dimension judge, 100-task seeded sample)
| profile | SWE-QA total /100 | correctness /20 | completeness | relevance | clarity | reasoning | our reward | our format_ok |
|---|---|---|---|---|---|---|---|---|
| claude (Sonnet 5) | 73.1 | 14.5 | 13.1 | 16.0 | 14.8 | 14.6 | 0.02 | 0.04 |
| qwen4b-base | 40.6 | 8.0 | 7.1 | 9.0 | 8.8 | 7.8 | 0.01 | 0.05 |
- Detail: `data/evals/<profile>/sweqa_100/{results.json,sweqa_judge.json}`; `uv run python -m codeqa.evals.report --set sweqa_100`. Judge cost measured at ≈ $0.007/task; 100 tasks ≈ $0.70 per profile, so the full 720 (~$5 per profile) is fine to run for the final table. Under our reward both profiles are ≈ 0 on SWE-QA because 95 % of their answers exceed the 450-token cap (LOG 21:00); the external judge has no length gate, which is why it separates the two models cleanly (73 vs 41) while our reward does not. This is the strongest argument for deciding the cap before run one.
- Affects: lead, talk (table columns 1–2 filled)

## 2026-09-18 21:45 · agent/tools + grader semantics · lead
- What: `grep` hits are now recorded in `RepoEnv.files_read()` as one-line spans (decisions.md #7), so citing a line seen in grep output passes the grounding gate. `find_symbol` still records nothing. Test in `codeqa/agent/tests/test_tools.py`.
- Detail: evidence in decisions #7; 7/120 of Claude's held-out episodes failed grounding this way. `efficiency.redundant_reads` sees these spans too; a later `read_file` around a grep hit is not "fully covered" so it is not counted as redundant. C: nothing to change in the grader. D: the UI's local grounding rule (cited range ⊆ read range) should include grep spans if it recomputes them client-side; the API's `check_citations` already uses `files_read`.
- Affects: grader (semantics only), apps/web badges, evals (re-run baselines after this + the caps)

## 2026-09-18 22:10 · grader · lane C agent — "is the judge or the data wrong?" audit (answer: no; the cap and one grounding rule are)
- Claude's fast-set correct rate of 0.22 decomposes as: 82/120 format gate (76 over the answer cap, 6 never answered), 7 grounding, 2 no citations, 29 reach correctness. Among those 29, mean correctness is 0.82 and 19 are 1.0.
- Judge reliability on Claude's 20 judged answers: Haiku re-run agrees with itself to mean |Δ| = 0.01; Haiku vs Sonnet 5 mean |Δ| = 0.07 (means 0.93 vs 0.92); every disagreement is one rubric item. Judge is sound; keep Haiku in-loop.
- Data issues found (lane B): rubric typo in `dcb-4344b2a4` ("≥ 2024.1.1" should be 2024.12.1; Claude was right and lost the item); two DeepCodeBench rows labelled `enumerate` with a single gold path where the question is unanswerable without the source file (`dcb-d806e6bd` cited the C++ twin `src/io/dataset.cpp` instead of `lightgbm/basic.py`; `dcb-d9050518` found a different docstring in `qlib`); `dcb-d58dd0e8` has facts but no gold path. DeepCodeBench questions assume the file context, so `locate/enumerate` derived from them are noisy; `explain` with the facts rubric is fine.
- **Grounding rule (lane A + lead):** all 7 grounding failures are Claude citing the exact line of a `grep` hit it never opened with `read_file` (e.g. `ranking_utils.h:L39-L40` after grep showed `ranking_utils.h:L39: …`). The model did see that line's text, so the citation is evidence-backed; under the current rule it scores 0 and the policy would learn to spend an extra `read_file` before citing any grep hit. Recommendation: `RepoTools.grep` (and `find_symbol` for the signature line) append `Span(path, line, line)` to `files_read` for each hit it displays. One-line env change; grader needs nothing. Affects 7/120 Claude episodes here (reward 0 → ~0.8) and every trained policy.
- Net for the lead before run one: (1) raise the answer cap (LOG 21:00); (2) decide the grep-hit grounding rule; (3) lane B fixes the three rows. The judge and the grader are not the problem.
- Affects: lead, agent (files_read), datagen

## 2026-09-18 22:30 · evals · lane C agent
- What: judge agreement numbers (0.01 Haiku–Haiku, 0.07 Haiku–Sonnet, on 20 judged Claude answers) written into `docs/decisions.md` #3 for the talk. Grader/trainer/evals tests pass under the new caps and the regenerated verbatim fixture (69).
- Queued: both fast baselines (`qwen4b-base`, `claude`, `--set fast`) re-run automatically under the new caps and the grep grounding rule as soon as lane B's three-row fix lands in `data/tasks/eval/fast.jsonl`. The watcher triggers when `dcb-4344b2a4`'s rubric no longer contains "2024.1.1" AND `dcb-d58dd0e8` has `expected_paths` or `expected_symbols` (or those rows are removed). Lane B: if the fix goes to `deepcodebench_test.jsonl` only, please regenerate `fast.jsonl` with the same seed so the watcher sees it. Results will overwrite `data/evals/<profile>/fast/`; the pre-fix numbers are in LOG 21:00.
- Affects: lane B (trigger), lead (talk table)

## 2026-09-18 22:40 · datagen · lane B
- What: the three DeepCodeBench eval rows from the 22:10 audit are fixed, at import time so re-imports keep them (`import_deepcodebench.FIXES`)
- Detail: `dcb-4344b2a4` rubric + reference now read "≥ 2024.12.1" (the guard is `_DASK_2024_12_1`). `dcb-d806e6bd` and `dcb-d9050518` are `explain` (facts rubric, judged) instead of `enumerate` with one gold path. `dcb-d58dd0e8` is `explain` and now carries `expected_paths` = its two citation files (`fastai/text/data.py`, `fastai/vision/data.py`). Re-imported `raw/deepcodebench.jsonl` + `eval/deepcodebench_test.jsonl` (test types now 180 explain / 48 enumerate / 4 locate) and replaced the same four rows in `eval/fast.jsonl` in place, sampling untouched. Test: `test_deepcodebench_review_fixes_apply_at_import`.
- Decision #7 (grep hits count as read): the pass-rate rows in `reports/passrate.jsonl` were measured under the old rule, so `grounded`/`reward` are slightly pessimistic for samples that cited a grep hit line; `found`, `answered`, `correct_lenient` (the difficulty inputs) are unaffected. No re-run.
- Affects: C3 (fast set rows changed), evals

## 2026-09-18 22:30 · apps · lead
- What: lane D gets D6 "Workshop" (training runs, checkpoints, data pages + read-only API) instead of a separate lane; spec in `lanes/D_product.md`. Inputs are all on disk today; metric key names in LOG 16:45. Nothing in D6 launches or modifies anything.
- Affects: D (build), C (if you rename a metric key, post it), B (`data/tasks/README.md` counts are what the data page is checked against)

## 2026-09-18 22:45 · apps · lead
- What: D6 spec tightened. The run page opens on a reward-growth chart (per-step points, 5-step rolling mean, ±1 s.e. band from `group_reward_std`, plateau strip with last-10-vs-previous-10 gain, overlay of two runs). Training data is shown as task cards (question headline, repo/type chips, gold as code chips opening the file viewer, rubric checklist, step-0 pass-rate bars, filters incl. pass-rate bucket, example-episode link). Details in `lanes/D_product.md` D6.
- Affects: D

## 2026-09-18 23:20 · datagen · lane B — handoff (B6, B7)
- What: `data/tasks/train/all.jsonl` = 1,489 tasks (C5) over 23 repos; eval files final; lane B complete
- Counts (source x type): codescout 444 (261 locate, 183 trace) · deepcodebench 445 (361 explain, 80 enumerate) · structural 401 (154 locate, 112 enumerate, 123 trace, 12 value) · teacher 199 (105 explain, 93 trace). Types overall: 466 explain, 418 locate, 399 trace, 192 enumerate, 14 value. 62% programmatic, 38% judged. 98% carry `required_citations`; 27% point at `__nodoc` repo ids (structural). Eval: `deepcodebench_test` 232, `sweqa` 720, `fast` 120.
- How it was filtered: every raw task sampled 2x with `qwen4b-base` through `RepoEnv` + `run_episode` + grader (rows in `reports/passrate.jsonl`); tasks at a two-sample 0 or 1 got 2 more samples (2,482 of 3,494 have n=4). Difficulty score = content-only correctness over answered samples when >= 2 answered, else the found-gold-lines rate (the base model finds the right lines and runs out of turns far more often than it answers). Window [0.1, 0.9] keeps 1,515; hard reserve 691 (`raw/reserve_hard.jsonl`), easy reserve 1,288 (`raw/reserve_easy.jsonl`, includes 26 over the sqlglot cap). Per-source step-0 rates (all raw tasks): structural: answered 63%, format_ok 53%, found gold 76%, content-correct 56%, strict reward 0.10, tool calls 5.2 · codescout: answered 37%, format_ok 9%, found gold 48%, content-correct 18%, strict reward 0.00, tool calls 6.3 · deepcodebench: answered 53%, format_ok 36%, found gold 60%, content-correct 40%, strict reward 0.05, tool calls 8.1 · teacher: answered 87%, format_ok 68%, found gold 91%, content-correct 65%, strict reward 0.12, tool calls 4.8.
- Gaps / caveats for C: (1) only 14 value tasks survived: the base model reads literals correctly, so most structural value tasks are in the easy reserve; pull from there if value coverage matters. (2) teacher is 199 tasks, not 400–600 (credit outage + $50 cap). (3) `grounded`/`reward` columns were measured under the pre-decision-#7 grounding rule; difficulty inputs are unaffected. (4) `eval/fast.jsonl` is regenerated by `cli split` deterministically (seed 7, stratified by repo); it equals the earlier file when built from the same eval files. (5) Structural tasks require the `__nodoc` snapshot + index + map to exist wherever training runs (`data/repos/*__nodoc`, `data/index/*__nodoc`); the 23 exist locally.
- Reproduce: `cli import --source all` -> `cli derive --repos 15` -> indexing `cli all --repos data/repo_list_codescout.txt --nodoc` -> `cli generate` -> `cli teach --seeds-per-repo 10 --max-cost 24` -> `cli filter --samples 2` (shard with `--shard i/n`; Tinker scales with sessions, ~40 tasks/min per process) -> `cli filter --refine` -> `cli split`. Reports for every step in `data/tasks/reports/`.
- Spend: Anthropic ~$52 total for lane B today (teacher $29, rewrites/paraphrases/summaries ~$7, judge in filter ~$7, misc), ~$27 of it after the top-up.
- Affects: C2 (dataset ready), C3 (eval files final), lead (talk numbers)

## 2026-09-18 22:55 · apps · lead
- What: D6 adds a Traces page: every episode on disk (dev, product, eval traces, training rollouts) in one list, opened in the existing research-log view with a grade panel and citation table beside it, plus a side-by-side compare of two traces on the same task. Grades come from the adjacent per_task row or rollout summary; ungraded traces get `check_citations` only (no judge calls from the UI).
- Affects: D; C (per_task.jsonl and rollout summary shapes are read as-is; post if they change)

## 2026-09-18 23:40 · deploy · lane E — E1 scoped (not pushed yet)
- What: dry-run of the first commit without `git init` (simulated index over the work tree): 277 files, 3.9 MB, nothing over 500 KB, no `data/`, `.env`, `dist/`, `node_modules/`, or key files. Pattern scan and a scan for the literal values of every key on this machine (`.env`, `~/.modal.toml`, `gh auth token`, HF, AWS) found nothing. `uv run pytest -q -m "not live"` passes (119). `scripts/deploy/secret_scan.sh` does all three checks; run it before every push.
- Gotcha: the Tinker key is `tml-…` (73 chars), not `tinker_…` as the lane file says; `tinker_` also matches every `tinker_cookbook` import. Anthropic `sk-ant-…` (108), workspace `wrkspc_…` (31), GitHub `gho_…`. The scan script has the corrected prefixes.
- Gotcha: `.gitignore` had `data/` + `!data/.gitkeep`, which does not re-include the file (git never descends into an excluded directory). Changed to `data/*` + `!data/.gitkeep`.
- Note: `.agents/skills/**`, `.claude/skills/**` (symlinks into `.agents`), and `skills-lock.json` are the web lane's design-skill installs (~3 MB of demo JPGs); they would be committed as-is. Lead to decide whether to ignore them.
- Affects: lead (repo name/owner, skill folders), E2+

## 2026-09-19 15:00 · evals · lane C agent — fast baselines re-run under the new caps + grep grounding (lane B's final fast.jsonl detected 14:36)
| profile | reward | correct_rate | format_ok | citation_valid | tool_calls | tool_calls/correct | n | (before: reward / correct) |
|---|---|---|---|---|---|---|---|---|
| claude (Sonnet 5) | 0.452 | 0.483 | 0.542 | 0.500 | 5.15 | 10.7 | 120 | 0.199 / 0.217 |
| qwen4b-base | 0.036 | 0.042 | 0.475 | 0.058 | 7.58 | 182 | 120 | 0.000 / 0.000 |
- By group, claude: deepcodebench/explain 0.64 (n=52), deepcodebench/enumerate 0.62 (8), sweqa/explain 0.34 (46), sweqa/locate 0.05 (14). Base: deepcodebench/enumerate 0.25, everything else ≤ 0.03; 42 % of base episodes still end without an answer.
- These are the talk's left columns. `data/evals/<profile>/fast/`, `evals.report --set fast --markdown`, `data/evals/plots/eval_fast.png` (re-render with `evals.plots --set fast`).
- Affects: lead (talk table)

## 2026-09-19 15:10 · clients/anthropic · lane C agent
- What: 2/120 Claude fast episodes died with `400 … tool_use ids were found without tool_result blocks immediately after` (`sweqa-requests-042`, `sweqa-pytest-037`). When the driver drops the calls beyond the budget from a multi-call turn, the assistant message keeps all its `tool_use` blocks but only the executed ones get a `tool_result`; Anthropic requires one per id.
- Fix: `_to_anthropic` now synthesizes a `tool_result` ("[tool call dropped: no tool calls remaining. Answer now.]") for every unanswered `tool_use` id, placed first in the following user turn. Lane A may prefer the driver to emit real `role=tool` messages for dropped calls with their `call_id`s; either way the client is now tolerant. The Tinker path is unaffected (cookbook renders its own).
- Affects: apps/api (Claude profile), datagen teacher, evals
