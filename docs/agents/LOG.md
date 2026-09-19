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

- 2026-09-19 01:10 · apps/api → evals import: `apps/api/workshop.py` calls `codeqa.evals.monitor.checks` for the run Health panel. Request: move `checks` (pure, rows → warnings) to `codeqa/shared/run_health.py` and have both import it. Until then the import is guarded and documented (LOG 01:10).

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

## 2026-09-19 15:40 · trainer · lane C agent — training-file audit (`data/tasks/train/all.jsonl`, 1,489 tasks)
- **Structure: clean.** All 1,489 parse as C5; 46 snapshots (23 repos + nodoc twins) all have `map.txt`; every `expected_paths` file exists, every `expected_symbols` entry resolves in its index (0 misses), every `required_citations` span is inside its file; 0 eval task ids or questions in train; the 15 SWE-QA repos are absent from train (the 8 DeepCodeBench repos are in both by design: test split is held-out in-repo); 0 duplicate questions; 0 ungradeable tasks (577 judged with rubrics, mean 6.4 items, max 24; 912 verifiable). Budgets all `null` → new caps apply.
- **Gold caveat (not a blocker):** CodeScout gold is "entities touched by the fix PR": 264/444 tasks have 2–8 gold symbols over up to 3 files, some unrelated to the question (e.g. `cs-tobymao__sqlglot-3182` asks about `Expression.transform` but gold also lists `DuckDB`, `Snowflake`, `_date_delta_sql`). Symbol-set F1 gives partial credit, so it trains, but expect CodeScout multi-gold correctness to cap well below 1.
- **The pass-rate window is on a lenient metric, not on our reward.** `reports/passrate.jsonl` shows the [0.1, 0.9] band holds for `correct_lenient_all` (68 % of kept tasks inside it), while under our reward only 8 % of kept tasks ever scored > 0 in the base samples (mean reward 0.029; measured under the old caps and grounding rule).
- **Step-0 signal, re-measured under the current environment** (`data/tasks/eval/train_probe.jsonl`: 40 tasks stratified by source × type × 8 samples at temperature 1.0, `data/evals/qwen4b-base/train_probe/`): mean reward 0.016; gates: format 55 % (175/320; stalls + over-cap), citations 41 % (132: answered, no bracket citations), grounding 1 %, passed 3 % (11). Groups of 8 with reward variance: **5/40 = 12 %**; with the no-answer penalty the shaped reward has variance in **31/40 = 78 %** (answer vs stall). Per group: codescout locate/trace and structural trace/enumerate/value 0 % reward variance; deepcodebench explain/enumerate 33 %; structural locate 20 %.
- **What this means for run one** (16 groups × 8 per step): ≈ 2 groups/step carry a correctness signal, ≈ 12 carry only the "answer, don't stall" signal. Expect the first steps to raise `stop_answer` and lower stalls, then a plateau near reward 0 with `citations_parse` stuck ≈ 0.6 unless bracket citations emerge by chance (≈ 4 passing episodes per step). The monitor's std/uniq checks will flag that plateau.
- **Recommendations (lead decision, in order of cost):** (1) `--groups-per-batch 32` (2× sampling, ≈ 4 correctness groups/step); (2) a small grounded-citation credit, e.g. +0.05 when all gates pass and correctness is 0 (a reward change: keeps stall −0.1 < no citations 0 < grounded-but-wrong 0.05 < correct; bounded, and the RepoSearch-R1 recipe uses a 0.1 format term the same way); (3) if `citations_parse` has not moved by step 10, a short SFT warm start on lane B's 199 teacher traces (Claude answers with bracket citations) via the cookbook's supervised loop, then resume RL from that checkpoint. Not recommended: re-windowing the data on our reward (only 8 % of raw tasks qualify).
- Affects: lead (run-one config/reward), lane B (CodeScout gold shape, for the talk's data slide)

## 2026-09-19 00:30 · apps/web, apps/api · product agent
- What: D6 Workshop done — runs, checkpoints, data, traces pages over `data/`, read-only
- Detail: `apps/api/workshop.py` (router mounted in `server.py`; endpoints in `apps/api/README.md`; 5 s mtime cache; 4 tests on a synthetic data tree). Web: `/workshop/*` (see `apps/web/README.md`). Verified on real data: smoke3 vs smoke_lr1e4 overlay shows the lr collapse from LOG 16:40/17:30; data counts equal `data/tasks/README.md`; 944 traces listed; `sweqa-flask-001` opens for claude / qwen4b-base / qwen4b-smoke1-step3 and compares side by side; live polling confirmed by touching a metrics.jsonl. Charts use the dataviz reference palette (`--series-1..4` in `index.css`), one axis per chart, held-out evals as points on the same axis.
- Gaps: (1) `qwen4b-smoke1-step3` has no `data/evals/<profile>/fast/` yet, so the checkpoints row is dashes until `evals.run --profile qwen4b-smoke1-step3 --tasks data/tasks/eval/fast.jsonl` runs; the join is automatic. (2) Training rollout summaries carry no task_id/repo, so rollout traces show tags only and no citation check; if the trainer adds `task_id` to `trajectory_metrics` or the summary row, the traces page will pick it up (looked up in `_task_index`). (3) Rollout `assistant_content` on tool turns is "\n" (thinking not logged), so rollout rows have no thinking lines.
- Gotcha: `Trace.messages` for Claude include one malformed `read_file` arg (`start: "150, \"end\": 330}"`, LOG 15:40); the UI now labels it "malformed range" instead of "lines NaN–NaN".
- Affects: lead (talk figures: run overlay, same-task compare), C (metric keys read as-is; `task_id` in rollout summaries would unlock grounding checks there)

## 2026-09-19 16:30 · trainer · lane C agent — run-one improvements landed (per the lead: data stays ≥ 1k)
- What: (1) `groups_per_batch` default 32 (`RunSpec`, `--groups-per-batch`); run-one command in `codeqa/trainer/README.md`. (2) Grounded-citation credit: `trainer/group_rewards.grounded_credit` floors the shaped reward at +0.05 for answers that pass every gate but score 0; applied in `compute_group_rewards` next to the no-answer penalty; `--grounded-credit 0` disables; metric `env/all/grounded_credit` (rate); the held-out evaluator stays unshaped. Tests: `test_grounded_credit_ladder` + the existing penalty tests (6 pass). (3) SFT warm start: `codeqa/trainer/sft.py` (`build` = traces → gate-passing cookbook conversations rendered with RepoEnv's exact tool prefix; `train` = cookbook supervised loop). Dry-run on Claude's 120 fast traces (eval set; never to be trained on) to prove the pipeline: see the numbers in this entry's follow-up. decisions.md has the evening entry.
- Not done on purpose: no training or SFT launched; no teacher traces exist (lane B kept only `teacher_attempts.jsonl`), so SFT data needs `evals.run --profile claude --tasks data/tasks/train/teacher.jsonl --set sft_seed` (~$30) before `sft build`.
- Affects: lead (run-one launch command), trainer

## 2026-09-19 16:50 · trainer · lane C agent — SFT pipeline dry run (follow-up to 16:30)
- `sft build` over Claude's 120 fast traces (eval set, pipeline proof only): 96 traces no longer match a task id (lane B's final `fast.jsonl` renumbered them), 10 failed a gate, 7 below reward 0.5, 7 kept → rendered to cookbook datums of 5.5–9.6k tokens with ≈ 640–830 weighted assistant tokens each (tool-call XML rendered by the qwen3_5 renderer, `ALL_ASSISTANT_MESSAGES`). `build_sft_config` builds train/test datasets from the file. The cookbook's `FromConversationFileBuilder` cannot be used: it never rehydrates `tool_calls`, so `sft.py` carries its own loader.
- Caveat for whoever launches it: Claude traces carry no `<think>` blocks, so the warm start teaches "answer without thinking"; RL from that checkpoint samples with thinking on. Acceptable for a format warm start; if it hurts, `--epochs 1` and a low lr.
- Affects: lead (optional SFT), trainer

## 2026-09-19 01:10 · apps/web, apps/api · product agent
- What: optimizer metrics and health checks on the run page
- Detail: `GET /runs/{name}` now returns `optim/*`, `kl_ref/*`, `loss/*`, `time/train_step`, `time/total` rows and `warnings` = `codeqa.evals.monitor.checks(rows)`, so the dashboard and the CLI monitor show the same alarms. Run page: a Health panel (green line when nothing trips), an Optimizer panel charting the known keys (KL v1/v2, post-update KL when `--compute-post-kl`, entropy, lr) with the monitor's thresholds as reference lines (KL 0.05; entropy 40 % of step 0) and, adaptively, any other `optim/`, `kl_ref/` or `loss/` key that appears (e.g. `kl_ref/*` from `--kl-penalty`). Axes/tooltips use significant digits so 3e-4 reads as 3e-4. Gradient norms: not exposed by Tinker (confirmed by lane C); if a future SDK adds one under `optim/`, it charts with no code change.
- Contract change request (import rules): `apps/api` imports `codeqa.evals.monitor.checks` (one pure function over metric rows). Rule 2 says api and evals never import each other; the clean fix is moving `checks` to `codeqa/shared/run_health.py`. Lead's call; the import is guarded so a move is a one-line change here.
- Affects: C (metric keys read as-is; `checks` location), lead

## 2026-09-19 01:40 · apps/web · product agent
- What: `/workshop/live` — one-screen training monitor (default workshop page)
- Detail: auto-selects the run whose metrics.jsonl changed in the last 3 min (else the newest), polls every 10 s, reward chart with s.e. band + held-out points, Health (monitor checks), last-step numbers with deltas vs the previous step, optimizer strip (KL v1/v2, entropy, lr, any `optim/`/`kl_ref/`/`loss/` key), tab title `● <run> · step N · r 0.xxx`, toast on each new warning. Workbench header shows a green dot while a run is live. Try it on run one: `uv run uvicorn apps.api.server:app --port 8000` + `VITE_API_URL=http://localhost:8000 pnpm dev`.
- Affects: lead (run-one babysitting), C (nothing)

## 2026-09-19 17:30 · metrics · lane C agent — metric map for the Live page (lane D)
- Added two keys to `grader.metrics` (so they land as `env/all/*`, `env/<source>/*`, `env/<task_type>/*`, `eval/fast/env/all/*`): `correct` (1 if reward > 0) and `stalled` (1 if stop ∈ {budget, max_turns}). Everything below already exists in `data/logs/<run>/metrics.jsonl`; no other logging changes are needed.
- **Signal density (collapse early warning):** `env/all/by_group/frac_mixed` (share of groups with any gradient; ~0.12 at step 0 on this data), `frac_all_bad`, `frac_all_good`, `env/all/unique_tool_sequences_per_group` (→ 1 = rollouts collapsed), `env/all/group_reward_std`.
- **Gate funnel (stacked bar per step):** `env/all/gate_format`, `gate_citations`, `gate_grounding`, `gate_budget`, `gate_judge_error`; the remainder is the share that reached correctness.
- **Format emergence:** `env/all/citations_parse` (bracket citations present), `format_ok`, `citations_exist`, `citations_grounded`, `identifier_grounded`.
- **Stalls and shaping:** `env/all/stalled` (= `stop_budget` + `stop_max_turns`), `stop_answer`, `stop_parse_error`, `stop_overflow`; `env/all/no_answer_penalty` and `env/all/grounded_credit` (rates); `env/all/reward_shaped` (what trains) vs `env/all/reward` (what we report). Plot shaped and unshaped on one axis.
- **Per source / type:** `env/<codescout|deepcodebench|structural|teacher>/reward` and `env/<locate|value|enumerate|trace|explain>/reward` (any key above works under these prefixes).
- **Efficiency (run-two headline):** `env/all/tool_calls`, `prompt_tokens`, `answer_tokens`, `turns`, `tool_errors`, `redundant_reads`; derived `tool_calls / correct` and `prompt_tokens / correct` (use the new `correct` key, not mean reward).
- **Judge health:** `env/all/judge_error_rate`, `group_all_judge_errors`; `time/compute_group_rewards:total` vs `time/policy_sample:total` (sampling-bound check); `time/total` per step → steps/hour; `progress/done_frac` → ETA.
- **Optimizer:** `optim/lr`, `optim/entropy`, `optim/kl_sample_train_v1|v2`; only when launched with the flags: `optim/post_kl` (`--compute-post-kl`), `kl_ref/*` (`--kl-penalty`). Show when present. Gradient norms do not exist on Tinker.
- **Held-out (every `eval_every` steps):** `eval/fast/env/all/{reward,correct,correctness,format_ok,citations_parse,citations_grounded,tool_calls,stalled}`; the talk's four are reward, correctness, citations_grounded, tool_calls/correct.
- **Files, not metrics:** `data/logs/<run>/checkpoints.jsonl` (last row = latest resumable step, `state_path`/`sampler_path`); `data/logs/<run>/iteration_<N>/train_rollout_summaries.jsonl` (per-rollout rewards for a step; `eval_fast_rollout_summaries.jsonl` likewise); `data/evals/<profile>/<set>/results.json` for the baselines table.
- Affects: apps/web Live page (D), lead

## 2026-09-19 02:30 · apps/web · product agent
- What: run page shows lane C's full metric map (LOG 17:30), in their order; home page cut to one screen
- Detail: `components/workshop/panels.tsx`: signal density (by_group fractions stacked, unique tool sequences with the 1.5 collapse line, group std), gate funnel (stacked bars per step + "reached grading"), format emergence (citations_parse/format_ok/exist/grounded/identifier), reward shaped vs unshaped + stalled/no_answer_penalty/grounded_credit/answered rates, reward by source and by type, efficiency row incl. tool_calls/correct and prompt_tokens/correct from the new `correct` key, judge health + time split (sampling vs grading vs train step) + steps/hour + ETA from progress/done_frac, held-out block (talk's four numbers as tiles with change since the first eval, plus curves), optimizer. Every panel hides itself when its keys are absent, so smoke runs and run one share the page; `stalled` falls back to stop_budget+stop_max_turns for pre-17:30 runs. Live page adds signal density, gate funnel and the held-out tiles. API whitelists `time/policy_sample:total`, `time/compute_group_rewards:total`, `time/env_step:total`, `time/run_evaluations_parallel`.
- Home page: header, hero (headline, summary, chat replay, four one-line steps), footer. The before/after, how-it-works tiles, spec sheet, FAQ, sources block are gone (user: "a clean home page for a simple project").
- Affects: lead (Live/run pages for run one), C (nothing)

## 2026-09-19 03:10 · apps/api, apps/web · product agent
- What: password gate on the API and the web app
- Detail: API middleware requires `Authorization: Bearer <CODEQA_PASSWORD>` (default `Action!`) on every path except `/health` and `POST /auth/login`; CORS preflight passes. Web: every route except `/` shows a password screen; the password lives in localStorage and rides on every fetch (`lib/auth.ts`); any 401 clears it. Tests send the header; one test covers 401/200/login. Shared secret, not auth: no sessions, no rate limit. Anyone scripting the API (evals, demo curl) must add the header or set `CODEQA_PASSWORD` to match.
- Affects: everyone who calls apps/api (add the header), lead (demo)

## 2026-09-18 23:20 · agent · lead
- What: agent variants for ablations, off by default. `RepoEnv(task, profile, variant=...)` or env var `CODEQA_AGENT_VARIANT`; names `default` (unchanged five tools + map), `noindex` (grep/read_file/list_dir, no map), `nomap`, `bash` (one read-only shell tool, map kept), `bash_nomap`. Registry in `codeqa/agent/variants.py`; rules text per variant in `prompts.py` (`SYSTEM_RULES` untouched).
- Detail: the `bash` tool (`RepoTools.bash`, `codeqa/agent/shell.py`) runs one command in the snapshot root under restricted bash with an allowlisted PATH of read-only binaries, a pre-check (no `..`, absolute paths, `~`, redirection, substitution, `sed -i`, `find -delete/-exec`), and on macOS `sandbox-exec` (no writes, no network, no reads outside the snapshot). It records seen lines from numbered output (`grep -n`, `grep -rn`, `nl -ba … | sed -n`) into `files_read`, so the grounding gate and the grader are unchanged. Unnumbered output records nothing; the reward teaches the numbered forms. Each command = one tool call. 20 tests in `codeqa/agent/tests/test_bash.py`; full offline suite 146 pass.
- How to use without touching your CLI: `CODEQA_AGENT_VARIANT=bash uv run python -m codeqa.trainer.run …` / `codeqa.evals.run …`. Nothing else reads the variable. D: a `bash` tool call renders as a generic row (`Ran: <command>`) if you add one; not required today.
- Affects: nobody by default. C (optional `--agent-variant` flag later), D (optional generic row)

## 2026-09-18 23:50 · datagen · lane B
- What: `eval/fast.jsonl` is pinned to the 120 ids lane C already evaluated (`data/evals/claude/fast/per_task.jsonl`); my `cli split` had resampled it (only 24/120 overlap). `split` now keeps the existing ids and only refreshes their records; `--refresh-fast` resamples. Two more DeepCodeBench fixes: `dcb-625f7f86` and `dcb-d806e6bd` now end with "(in the Python package)": xgboost and LightGBM have a C++ core with the same logic, Sonnet answered from `ranking_utils.h` / `dataset.cpp` and scored 0/4 and 0/5 against Python-side rubrics. All fixes live in `import_deepcodebench.FIXES`.
- Also from lane C's Claude run on the fast set (the 14:50 file): 60/120 pass all gates; among them mean correctness 0.89 (DCB explain, 37), 1.0 (DCB enumerate, 5), 0.91 (SWE-QA explain, 17); only the two tasks above scored 0. Format still fails 55/120: 44 answers over the cap (median 903 tokens vs the 800 explain cap), 9 no final answer, 2 episode errors. Frontier-model validity check of the fast set with the current grader: running as `data/evals/claude/fast_validate/` (Sonnet 5, ~$12).
- Affects: C3 (fast set ids unchanged from their baseline; records refreshed), lead (answer cap still binds Claude at 800)

## 2026-09-19 04:20 · apps/api, apps/web · product agent
- What: every saved checkpoint is askable; per-event timing; format failures shown as errors; compare page explanations
- Detail: `GET /profiles` now appends a read-only profile per row of `data/logs/<run>/checkpoints.jsonl` (`qwen4b-<run>-step<N>`, kind tinker, `base_model` from config.json; profiles.yaml wins on a name clash) and `/ask` resolves either; the picker groups "Models" and "Training checkpoints". `/ask` stamps `t` on every event and adds `model_seconds` / `tool_seconds` to `stats` (measured between events in the API: model = tool result → next model event, tools = call → result); the ledger shows per-call seconds. `citations` carries `format_ok`/`format_reason` from `grader.gates.format_gate` + `citations_parse_gate`; the UI shows a red "fails the format check, scores 0" banner with the reason in plain words. Compare: framed composer + scan/bracket loading, rows for seconds in model / in tools, one-line explanation under each metric (prompt tokens = input tokens summed over turns; citations verified = share whose lines were read or grep-hit, not truth). Suggestion chips ask immediately. Measured base vs `qwen4b-smoke3-step3` on the flask locate question: 2 calls / 14.2k / 5.7 s vs 8 calls / 43.6k / 32.9 s, both 100% verified (grep hits now count). Tool time is ~0.01 s per call; the episode is model-bound.
- Contract note: `t`, `model_seconds`, `tool_seconds`, `format_ok`, `format_reason` are additive fields on C9 payloads (the schema is `dict[str, Any]`); no change to `contracts.py`. Listed in `apps/api/README.md`.
- Affects: lead (demo picker shows checkpoints), C (nothing)

## 2026-09-19 00:05 · datagen · lane B — frontier check of the fast set (Sonnet 5, current grader)
- What: `data/evals/claude/fast_validate/` — Sonnet 5 on the pinned 120-task fast set, temperature 0.2, ~$12
- Result: 61/120 clear every gate. Among those 61: 43 score 1.0, 15 score 0.5–0.9, 3 score 0. Mean correctness by slice: DCB explain 0.86 (n=40), DCB enumerate 1.0 (3), SWE-QA explain 0.91 (16), SWE-QA locate 0.58 (2). Reward over all 120 = 0.44. Gates: format 49 (38 over the answer cap, median ~900 tokens vs 800; 11 no final answer), grounding 7 (all multi-line `read_file` ranges the model never opened, none are grep hits, so decision #7 is in effect), citations 3.
- The 3 zeros are the same failure: an under-specified DeepCodeBench question answered correctly about a different plausible target in a large repo (`dcb-6602505a` Grounding-DINO instead of SAM; `dcb-c5c09820` BinaryCrossentropy instead of SparseCategoricalCrossentropy; `dcb-d9050518` a different docstring in qlib). DeepCodeBench questions were written with the source file as context. Fixed by appending a short context qualifier to those three questions (`import_deepcodebench.FIXES`, `question_suffix`), same treatment as the two C++/Python rows earlier. Re-imported; `cli split` refreshed the fast records in place (ids unchanged).
- Reading: task validity among reachable tasks is ~95% (58/61 non-zero) before the qualifiers and should be ~100% after; the judge is not the limiter. What limits a frontier model's reward is the answer cap (38/120) and answering without a final message (11/120), both grader/prompt policy.
- Affects: C3 (five fast-set questions now carry a qualifier; re-run baselines if exact comparability matters), lead (cap)

## 2026-09-19 05:10 · apps/api, apps/web · product agent
- What: compare page takes up to four models, and "Judge with Opus" grades them after researching the question itself
- Detail: `apps/api/judge.py` (`POST /judge`, SSE): Opus runs the same `RepoEnv` + `run_episode` harness (temperature 0.3, trace saved to `data/traces/judge/`), then one judge call per candidate with its own cited answer as the trusted reference and the candidate's citation check as evidence; JSON verdict 0–10 + correct + summary + issues + strengths. UI: referee panel shows Opus's research live, one verdict card per column, and an "Opus judge" row in the summary table. Measured on the flask locate question: Opus 7 calls / 16 s; untrained 7/10, `qwen4b-smoke_lr1e4-step3` 9/10, with the citation check quoted in the reasons. Also: the `/ask` clock now starts when the episode starts, so `model_seconds` cannot exceed `seconds`.
- Note: this is a product-side referee for demos; it is not a training signal and never touches the grader's judge (Haiku, rubric-based).
- Affects: lead (demo beat 6 can end on the referee), C (nothing)

## 2026-09-19 00:30 · datagen · lane B — Opus 5 on the 18 fast-set tasks Sonnet did not fully solve
- What: `data/evals/opus/fast_incorrect/` (profile `opus` added to profiles.yaml; task file `eval/fast_incorrect.jsonl`), ~$5
- Result: Opus clears the gates on 7/18 (3 perfect, mean 0.78, no zeros); 11/18 fail the 800-token cap (853–1,190 tokens; Opus median ~900, prompt 50k tokens, 6.7 calls per task). Re-judged with the cap ignored, those 11 score mean 0.81, 5 perfect, 0 zero. Two of the three former zeros (`dcb-6602505a`, `dcb-d9050518`) are 1.0 after the context qualifiers; `dcb-c5c09820` is 4/9 for both models: its rubric wants the Torch backend signature and the exact `super().__init__` line, which are true but more specific than the question asks.
- Net over 18: Opus content ≥ Sonnet on 12, equal on 3, lower on 3; combined (Sonnet, Opus content) no task is below 0.44, so none of the 18 is a broken task. The judge misses are mostly rubric items that name a specific file, exact string, or one of several call sites the answer left out, i.e. legitimate partial credit.
- For the lead: the answer cap is now the single largest cost for frontier models on this benchmark (Sonnet 38/120, Opus 11/18 among the hard ones). Either raise explain to ~1,200 or accept that the eval measures concision too, which is a defensible choice for an efficiency-focused agent, but then quote frontier numbers with that caveat.
- Affects: C3, lead (cap decision), talk

## 2026-09-19 18:20 · grader · lane C agent — independent reward red-team
- What: an independent reviewer agent red-teamed the reward (12 failure modes, 9 verified with crafted traces through the real grader; 15 improvements). Full report: `docs/research/reward_redteam_2026-09-19.md`.
- Headline: the judge is sound (live probe: hedging, injection, fake rubric echo, vague answers all 0; good answer 1.0). The gates produce three verified classes of false zeros: (1) grounding off-by-one around one-line grep spans, the very case decision #7 addressed; (2) symbol credit requires the `def` line in a cited range, so body or call-site citations score 0 on the 306 callee tasks; (3) content-free or shotgun answers score 1.0 on path-only / single-gold tasks. Run-two only: my prompt-token accounting sums the cumulative context (3-4× over), so every efficiency variant is degenerate until fixed.
- Lane C proposes B1-B7 (≈ 1 h, offline-tested against the adversarial fixtures) before run one, B8-B9 before run two; not B10/B12 without data. Awaiting the lead's go (reward change).
- Affects: lead, grader, trainer (run two)

## 2026-09-19 00:10 · agent/modal · lead
- What: the `bash` variant can run on Modal sandboxes: `CODEQA_BASH_EXECUTOR=modal` (default `local`), pool size `CODEQA_MODAL_SANDBOXES` (default 4). `codeqa/agent/modal_shell.py`; app `codeqa-sandbox`, image debian_slim + ripgrep, volume `codeqa-data` mounted at `/data` (snapshots at `/data/repos/<repo_id>`), `block_network=True`, sandbox lifetime 1 h, idle timeout 10 min.
- Detail (measured, flask): pool warm-up 12 s, then 0.7 s per command; 32 concurrent commands on 4 sandboxes in 4.5 s, 32/32 ok. Same pre-check and seen-lines extraction as local, so grounding is identical. One sandbox serves many concurrent execs; scale = pool size × concurrency, not one sandbox per episode. The pool is not torn down explicitly by the driver; Modal reaps it on idle.
- Volume: flask + graphiti uploaded; the full `data/repos` upload is running (`modal volume put codeqa-data data/repos /repos`). Re-run that command after new snapshots.
- Affects: nobody by default. C: `CODEQA_AGENT_VARIANT=bash CODEQA_BASH_EXECUTOR=modal` in front of `codeqa.trainer.run` gives a bash-agent run whose tool calls do not touch the laptop.

## 2026-09-19 06:20 · apps/web, apps/api · product agent
- What: repo page in the workshop: overview (verbatim map.txt, languages, top dirs, dropped counts), file browser → file viewer, and a tool console that runs the agent's five tools by hand
- Detail: `GET /repos/{id}/overview`, `GET /repos/{id}/tools` (specs + caps), `POST /repos/{id}/tool` (real `RepoTools`, per-repo instance cached on symbols.json mtime, budget off, `files_read` returned per call). Data page repo rows link to `/workshop/repos/<id>`. Handy for checking what a grep or read returns under the caps before blaming the model. Also: seconds removed from the compare table (backend speed is not comparable), `done` no longer erases an `error` in the episode reducer (no-answer episodes now show a red block in transcripts), judge button always visible, header buttons outlined, thinking per step unravels in both views.
- Affects: apps/web, apps/api (19 tests)

## 2026-09-19 00:55 · datagen · lane B — content-only scoring of the fast set + suspect list
- What: every Sonnet fast-set answer re-judged with the length gate ignored (`data/evals/claude/fast_validate/content_scores.json`)
- Result: 109/120 answered; content mean 0.83; 66 >= 0.9, 34 in 0.5-0.9, 9 < 0.5. By slice: DCB explain 0.85 (49), SWE-QA explain 0.85 (40), SWE-QA locate 0.74 (13), DCB enumerate 0.71 (7). Of the 9 below 0.5: 2 already fixed by qualifiers (Opus 1.0), 2 were correct answers zeroed by citation mechanics (`dcb-ba0f687e` no bracket citation; `dcb-eaf163e5` two ranges in one bracket `[path:L953-L954, L155-L157]`, which CITATION_RE rejects) -> relabelled `explain` (they carry facts rubrics); `dcb-61f91ebc` never named its pipeline -> qualifier "(in StableDiffusion3Pipeline)" + paths from citations; `sweqa-pylint-034` is a vague question whose own reference says the answer is builtin `object` (removal candidate); `dcb-c5c09820`, `sweqa-conan-029`, `sweqa-flask-046` look hard-but-fair. Opus 5 is running on these 7 + the 11 tasks Sonnet never answered (`data/evals/opus/fast_suspect/`).
- For lane A/C: models write multi-range citations `[path:L1-L2, L5-L7]`; the regex only accepts one range per bracket. Either accept comma-separated ranges in `CITATION_RE` or tell the model in the system prompt (one bracket per range). Cheap and it removes a spurious zero.
- Affects: C3 (three more fast rows changed), A (prompt or regex)

## 2026-09-19 01:20 · datagen · lane B — bad-test hunt on the fast set, verdict and removals
- Standard used: a task is bad when the best frontier answer (Sonnet 5, Opus 5), scored on content with the length gate ignored, is below 0.5 AND the miss traces to the task (wrong/stale gold, no determinate answer, missing context), not to the model. Partial credit on a fair rubric is not a bad test.
- Outcome on 120 tasks: 3 removed, 6 repaired, 111 untouched. Removed (import-time EXCLUDE): `dcb-b900be44` (gold says the `gpu_coord_descent` deprecation is in `python-package/xgboost/core.py`; at the pinned commit the string exists only in `src/gbm/gblinear.cc`, Opus was right), `dcb-d806e6bd` (gold's second error message "different number of rows" exists nowhere in the LightGBM snapshot; Python raises ValueError then warns), `sweqa-pylint-034` (no determinate answer; its own reference says builtin `object`). Repaired earlier: 5 context qualifiers (SAM, SparseCategoricalCrossentropy, CSI500 collector, StableDiffusion3Pipeline, Python package) and 2 enumerate->explain relabels for rows with a facts rubric. Replacements drawn deterministically from the same source: `dcb-2cd7c69a`, `dcb-6ca0c02e`, `sweqa-requests-011`. Eval sizes now 230 / 719 / 120. Still flagged, kept: `dcb-c5c09820` (rubric asks for Torch-backend details the question does not; both models 4/9), `sweqa-flask-004` and `sweqa-conan-029` (long multi-part SWE-QA references; frontier 0.33-0.5).
- Frontier content scores after fixes (gate ignored): Sonnet mean 0.83 over 109 answered (61% >= 0.9); Opus on the 29 hardest: 0.81 mean, 11 perfect. Under the current gates the same runs score 0.44 (Sonnet) because 38/120 answers exceed the 800-token cap and 11 end without a final message; Opus exceeds the cap on 11/18.
- Recommendation (grader policy, lead's call): replace the hard answer-length gate with a proportional term. Concretely, move length into the efficiency multiplier with the same shape as tool calls (free up to the cap, linear to 0.5 at 2x cap), keep a hard gate only at 3x cap or on truncation, and report content correctness and gated reward as separate columns. This keeps the concision pressure, stops zeroing correct answers, and makes frontier baselines interpretable. One-file change in `grader/efficiency.py` + `gates.py`; the eval set does not need to change.
- Cost of the checks: Sonnet fast set $12, Opus 18 + 18 tasks ~$10, re-judging ~$1. Lane B total after the top-up ~$46 of $50.
- Affects: C3 (fast ids changed by 3), lead (length policy), talk (frontier baseline numbers)

## 2026-09-19 00:40 · runtime · lead
- What: decision #8 — agent runtime jobs (train / evals / filter / teach) move to Modal as detached functions over the `codeqa-data` volume; Tinker unchanged; product on EC2 (lane E). Runner under construction at `apps/trainer/modal_runner.py`; the CLIs are unchanged and run inside the container. Data on the volume: `/data/repos` (uploading), `/data/index`, `/data/tasks`, `/data/profiles.yaml`. Outputs are committed to the volume every 30 s; pull with `modal volume get codeqa-data /logs data/logs` (script coming).
- Affects: C (run commands become `modal run …` wrappers; nothing in the trainer changes), B (same for filter/teach), D/E (the API reads synced data; D6 curves update as the sync runs)

## 2026-09-19 00:50 · deploy · lead (for lane E)
- What: two facts that touch the EC2 setup since the lane doc was written
- Detail: (1) Data source: the Modal volume `codeqa-data` now holds `/repos` (upload finishing), `/index`, `/tasks`, `/profiles.yaml`, and will hold `/logs`, `/evals`, `/traces`, `/models` as jobs run there. The box can pull with `modal volume get codeqa-data /<dir> /opt/codeqa/data/<dir>` (needs `modal token` on the box, read-only is fine) instead of rsync from the laptop; either works, pick one and write it in `update.sh`. (2) Env overrides exist: `CODEQA_DATA_DIR` (data root) and `CODEQA_PROFILES` (path to profiles.yaml); the API needs neither if the layout is `/opt/codeqa/data` + `/opt/codeqa/profiles.yaml`. Nothing on the box waits for the Modal runner; product episodes run on the box against Tinker sampling as before.
- Affects: E

## 2026-09-19 19:30 · grader · lane C agent — red-team fixes landed (decisions.md evening entry)
- What: seven grader rules changed + two run-two accounting fixes, each with a unit test (`grader/tests/test_redteam_fixes.py`, 79 lane tests pass) and re-verified with the reviewer's exploit scripts. Before → after on the verified cases: off-by-one after grep hit 0 → 1.0; five good citations + one off-by-one 0 → 1.0; body-not-def citation 0 → 1.0; callee task with call-site citation (caller named in question) 0 → 1.0; shotgun 12-name list 1.0 → 0.5; content-free path answer 1.0 → 0.5; literal "." 0 → 1.0; grep-after-read under multiplicative eff 0.5 → 1.0 on calls. Unchanged as intended: unread file, fabricated path, no-tools, wrong-case path, no-bracket answers all still 0; grounded-but-empty still floors at the 0.05 credit only.
- Rules now: grounding coverage ±1 line around any shown span (`citations.GROUNDING_TOLERANCE`), near misses logged as `env/all/grounded_by_tolerance`; `RepoFiles.symbols_in` uses body overlap; `predicted_symbols` also accepts a gold symbol named in the answer whose name appears on a grounded cited line; symbols named in the question are excluded from precision; single-gold any-of discounted by 2/|cited files| (paths) or 3/|named cited symbols|; path-only tasks ×0.5 unless the file is named outside the bracket; trivial literals (True/False/0/1/…; 8 of the 12 value tasks) require a grounded citation overlapping `required_citations` (all 12 have them); judge accepts "true"/"false" strings.
- Run two: `efficiency.TOKENS_PER_CALL` 3000 → 1500, budget = prefix + 1500 × max_tool_calls, usage measured on the FINAL context (`context_tokens(trace)` from per-turn `Message.usage`, which the trainer now fills from `transition.ob.length`/`ac.tokens` and the driver already fills). Replay of the 120 smoke_lr1e4 rollouts: median usage 1.88 → 0.52, share at the 0.5 floor 83 % → 0 %, 43 % in the free zone. Metrics `prefix_tokens`, `context_tokens` added. Grep spans no longer count as redundant reads.
- Fixtures: `make_fixtures.py` now stamps per-turn usage on assistant messages (1,000-token prefix + 600/turn) so fixture traces exercise the final-context path; regenerated.
- Affects: trainer (run one reward), evals (re-grade baselines: running), lead

## 2026-09-19 01:20 · runtime/modal · lead
- What: the detached Modal runner works: `apps/trainer/modal_runner.py` (app `codeqa-jobs`, image = debian_slim + ripgrep + `uv_sync` of our lockfile + local source mounted at run time, volume `codeqa-data` at `/data`, secret `codeqa`, 4 CPU / 8 GB, 24 h timeout). First job: `codeqa.evals.run` on 3 flask tasks, Tinker sampling from inside Modal, grader, outputs committed; `scripts/modal_sync.sh` pulled `data/evals/qwen4b-base/modal_smoke/` back to the laptop.
- How: `modal run --detach apps/trainer/modal_runner.py --module <codeqa.trainer.run|codeqa.evals.run|codeqa.datagen.cli> --args "<the same CLI args you use locally>" [--env KEY=VAL,…]`. `data/…` paths in args are rewritten to `/data/…`. Code changes need no rebuild (source is mounted); dependency changes rebuild the image (~1 min). Logs: `modal app logs codeqa-jobs`. Pull outputs: `scripts/modal_sync.sh`.
- Gotchas: the volume is not shared live — the runner commits every 30 s; run `modal_sync.sh` to see new rows locally. `profiles.yaml` on the volume is the one the runner appends checkpoints to (`CODEQA_PROFILES=/data/profiles.yaml`); the sync script drops it at `profiles.modal.yaml` for a manual merge. Jobs on the same volume from two containers both commit; last writer wins per file, so do not run two jobs that write the same run name.
- Affects: C (run commands), B (filter/teach on Modal instead of the laptop), D6 (curves come via the sync)

## 2026-09-19 20:10 · data hygiene · lane C agent — `scripts/modal_sync.sh` replaced local eval outputs
- What: `data/evals/qwen4b-base/{fast,sweqa_100,train_probe}` (the base-model baselines, LOG 15:00 / 21:40 / 15:40) disappeared at 17:08 when `modal_sync.sh` pulled `/evals` from the volume: `modal volume get … --force` replaces a local directory that also exists on the volume with the volume's copy, and the volume held only `qwen4b-base/modal_smoke`. `claude/` and `opus/` survived because the volume has no copy of them. The numbers are preserved in the LOG; the base `fast` baseline is being regenerated under the new grader (this entry's follow-up) and `sweqa_100` + its Sonnet judge are queued behind it.
- Ask (lead / D): make the sync non-destructive — pull into `data/modal/` and merge, or restrict `modal volume get` to run names produced on Modal — before anyone syncs while run one's local outputs exist. Do not run `modal_sync.sh` while a local `evals.run` is writing.
- Affects: everyone with local `data/` outputs

## 2026-09-19 01:45 · runtime/modal · lead
- What: training on Modal proven. `codeqa.trainer.run` ran one step (10 graphiti tasks × 4) inside the `codeqa-jobs` container in 100 s: rollouts with the five tools over `/data/repos`, real grader, Tinker LoRA step, sampler checkpoint `tinker://84cd9354-…/sampler_weights/final`, `data/logs/modal_smoke_train/{metrics.jsonl,checkpoints.jsonl,iteration_000000}` committed and pulled with `scripts/modal_sync.sh`. The no-answer penalty is visible (value tasks reward −0.05 at step 0).
- Gap for lane C (your folder): `codeqa.trainer.run` does not write the final checkpoint into `profiles.yaml` (`add_profile`) or `data/models/manifest.json` (`CheckpointRecord`) — `scripts/smoke_train.py` does both; please add it at the end of `run.py` so `evals.run --profile qwen4b-<run>-step<N>` works with no manual step, on the laptop and on Modal (`CODEQA_PROFILES` already points at the volume there). Also: a task file smaller than steps × groups_per_batch needs `--epochs`, as you documented; the runner passes args through unchanged.
- Affects: C (add_profile + manifest at end of run), everyone launching runs (use `modal run --detach apps/trainer/modal_runner.py …`)

## 2026-09-19 20:50 · evals · lane C agent — base-model baselines regenerated under the red-team grader (follow-up to 20:10)
| set | profile | reward | correct_rate | format_ok | citation_valid | tool_calls | SWE-QA judge /100 |
|---|---|---|---|---|---|---|---|
| fast (120) | qwen4b-base | 0.031 | 0.050 | 0.542 | 0.050 | 7.6 | – |
| fast (120) | claude (re-graded traces) | 0.479 | – | 0.547 | 0.521 | 5.1 | – |
| sweqa_100 | qwen4b-base | 0.033 | 0.040 | 0.400 | 0.040 | – | 37.2 (was 40.6 on the earlier sample of episodes) |
- Claude's re-grade moved 9/117 tasks: two zeros → 1.0 (the off-by-one grounding cases), the rest ±1 rubric item from judge re-runs; mean 0.463 → 0.479. Base fast 0.036 → 0.031 (noise; new episodes). `data/evals/qwen4b-base/{fast,sweqa_100}` are back on disk; `train_probe` is not regenerated (its numbers are in LOG 15:40). `evals.run` now recreates its output folders before every write so a concurrent `modal_sync.sh` no longer kills a run (it killed one at row 27 at 17:09).
- Affects: lead (talk table), D

## 2026-09-19 02:30 · runtime/modal · lead
- What: both agent variants verified on Modal end to end (LOG 01:45 for five tools; bash agent: eval job 3 tasks / 38 s with sandboxes spawned from the container, training step 43 s with checkpoint). Docs: `docs/modal_runtime.md` (what runs where, volume layout, commands, adding repos/tasks/profiles, gotchas). Scripts: `scripts/modal_push.sh` (inputs up, per-repo) and `scripts/modal_sync.sh` (outputs down).
- What: redirect hook added (inert until switched on): `codeqa.shared.runtime.maybe_redirect_to_modal(module, argv)` is the first line of `main()` in `codeqa/trainer/run.py`, `codeqa/evals/run.py`, `codeqa/datagen/cli.py` (one line each, B and C: please keep it). With `CODEQA_RUNTIME=modal` in `.env`, the same command you type launches detached on Modal and prints the job id; `CODEQA_RUNTIME=local` in front opts out. Inside Modal it is a no-op. `CODEQA_AGENT_VARIANT` / `CODEQA_BASH_EXECUTOR` are passed through.
- Affects: B, C (nothing to change; your commands stay the same), D (unchanged)

## 2026-09-19 02:50 · runtime/modal · lead — DEFAULT CHANGED
- What: `.env` now has `CODEQA_RUNTIME=modal`. From now on, `uv run python -m codeqa.trainer.run …`, `codeqa.evals.run …`, and `codeqa.datagen.cli …` launch the same command as a detached Modal job and return within seconds, printing the call id (`modal app logs codeqa-jobs` to watch; `scripts/modal_sync.sh` to pull results into `data/`). Verified: an unprefixed `evals.run` ran on Modal end to end.
- Opt out per command: `CODEQA_RUNTIME=local uv run python -m …` (smokes, quick checks). Block instead of returning: `CODEQA_RUNTIME_WAIT=1`. Tests and code inside Modal are never redirected.
- Caveats: (1) the per-repo snapshot upload to the volume is still running; a Modal job on a repo not yet uploaded fails at the map check with a clear error — check `data/modal_repo_upload.log` or `modal volume ls codeqa-data /repos`. (2) Already-running local processes (lane B's refine filters) stay local; restart them through the normal command if you want them on Modal. (3) Two jobs must not share a `--run-name` / `--set`. (4) New task files or profiles must be pushed first: `scripts/modal_push.sh`.
- Affects: B, C (your commands are unchanged; where they run changed), D (nothing), E (pull results with `scripts/modal_sync.sh` on the box)

## 2026-09-19 03:00 · trainer · lead — RUN ONE LAUNCHED
- What: run one started on the laptop (`CODEQA_RUNTIME=local`; the volume upload is not complete, run one's tasks span 46 repos). `data/tasks/train/run1.jsonl` (1,841 verifiable tasks), profile `qwen4b-base`, 50 steps, group 8, 16 groups per step, lr 1e-4, variant none, `remove_constant_reward_groups` on, eval every 10 on `data/tasks/eval/fast.jsonl`, checkpoints every 10. Log `data/logs/run1/`, console `data/run1.log`.
- Lane B: your three `cli filter --refine` processes were stopped at 03:00 to give Tinker throughput to training; `filter` resumes from `reports/passrate.jsonl`. Relaunch after run one finishes (they will go to Modal by default now).
- Affects: B (filters paused), C (monitor with `codeqa.evals.monitor --run run1`), D6 (curves in `data/logs/run1/metrics.jsonl`)

## 2026-09-19 21:40 · grader · lane C agent — answer length is now a soft term (decisions.md night entry); run-one snapshot
- What: the hard `max_answer_tokens` gate is gone. `gates.length_factor = min(1, cap / tokens)` (floor 0.1) multiplies the reward (folded into the `efficiency` component; metrics `length_factor`, `answer_over_cap`); the prompt guidance is unchanged. The threat the cap defended against gets its own gate: `gates.verbatim_share` > 50 % (answer lines pasted from tool outputs) → format failure. Fixtures now carry real numbered tool-output lines so `verbatim.json` trips it. 80 lane tests pass.
- Effect on the same Sonnet 5 fast traces: reward 0.48 → 0.73, reached grading 52 % → 88 %, correct rate 87 %, mean correctness among graded 0.89, over-cap answers keep ×0.82 on average. Remaining Sonnet losses: 31/117 partial rubric credit (~0.8 each), 7 `max_turns` with no answer (env budget), 4 grounding, 2 driver errors. Raising `max_turns` for explain/trace is the lever for "reached grading"; partial credit is the judge working.
- Runtime note: `.env` now defaults `CODEQA_RUNTIME=modal`, so my `evals.run` for the base baseline was redirected to Modal (job writing `evals/qwen4b-base/fast` on the volume; pull with `modal_sync.sh` when it finishes). Local commands need the `CODEQA_RUNTIME=local` prefix; the monitor and grader CLI are unaffected.
- Run one (launched by the lead 17:21 local, `data/logs/run1`, 16 groups × 8, run1.jsonl 1,841 verifiable tasks) runs the current grader (all markers present: grounded_by_tolerance, length_factor, grounded_credit, reward_shaped). Steps 0-2: reward 0.16 → 0.21 → 0.35, citations grounded 0.17 → 0.42, tool calls 5.8 → 4.7, stalls 35 % → 30 %, group std 0.26-0.31, unique sequences ~6/8, entropy 0.34, sampler-vs-trainer KL ~4e-4, ~45 s/step. Monitor checks: OK. Held-out step 0: reward 0.04.
- Affects: lead, D (Live page), talk

## 2026-09-19 03:40 · clients/anthropic + driver · lead
- What: prompt caching in `AnthropicClient` (two breakpoints: the system block = tools + rules + map, and the last message block). Measured on a flask question with Sonnet: turn 1 wrote 6,198 tokens to cache, turn 2 read 6,198 from cache and paid full price for 2 tokens. `Message.usage` now carries `cache_read_tokens`, `cache_write_tokens`, `uncached_prompt_tokens`; `prompt_tokens` stays "everything the model read" so it is comparable with the Tinker client and the grader's efficiency term is unchanged. Nothing else to do for teacher/judge/evals: every Anthropic call goes through this client.
- What: `run_episode(..., trim_tool_outputs_after=N)` (opt-in, default off): tool outputs older than the last N assistant turns are replaced by a one-line stub in the prompt the model sees; the trace keeps the full text and grounding still uses `files_read`. Product knob (D2 may expose it as a request option); not used in training.
- Why (context size): every turn resends the whole conversation, so an episode costs ~10× its unique tokens. Caching fixes the bill for Claude; the efficiency term (run two) and trimming address the size itself. Read-cap/map-size reductions are deferred: they change the prompt and would invalidate run one's policy.
- Affects: B (teacher cost drops ~3–5×), C (Claude baselines/audit cheaper; per-task rows gain cache fields), D (trim knob)

## 2026-09-19 08:10 · apps/web, apps/api · product agent
- What: trace page has an "Everything the model saw" tab: the full C6 message list (system prompt, user prompt with the map, every assistant turn with thinking and tool calls, every tool result untruncated, per-turn token usage); `GET /traces/{id}` now returns `messages`. Rollout traces reconstruct assistant/tool messages from the step logs (prompts are not in the summaries; `ob_len` shown as the prompt size per turn). Also: five typed example questions per repo (one per task type, `value` from a signature default, `enumerate` from a class's methods), task type passed through `/ask`, conversation cards in the rail, link-preview tags + og.png + icons, dev server derives the preview origin from the request host.
- Affects: apps/web, apps/api (19 tests)

## 2026-09-19 04:05 · trainer · lead — run one, step 10
- What: first held-out result of a trained checkpoint. `eval/fast` (120 tasks, 23 repos, mostly judged explain): reward 0.04 → **0.31**, correct 0.04 → 0.31, format ok 0.53 → 0.97, grounded citations 0.05 → 0.92, tool calls 7.6 → 4.4, max-turn stalls 0.28 → 0.00. Train reward per step is noisy (16 tasks/step) but format is at 0.99 and the remaining training-side failures are grounding (citing unread lines, ~26 %). Collapse checks OK: group std 0.23–0.31, unique tool sequences 5–7 of 8, entropy 0.35 → 0.23 (drifting down, watch), KL tiny. 40–70 s per step.
- Profile `qwen4b-run1-step10` is in `profiles.yaml` and `data/models/manifest.json` (added by hand; C: the trainer should do this). D: it is usable in the model switcher and `/compare` now — base vs step-10 on flask is the demo's beat. C: re-run the Claude `fast` baseline under the new caps so the table has a like-for-like strong-model column; at 0.31 the 4B step-10 checkpoint is above the old Sonnet number (0.20 under the 450 cap).
- Affects: D (demo, D6 checkpoints page), C (baselines, evals table), talk

## 2026-09-19 09:00 · grounding rule · product agent — find_symbol hits do not count as read (grep hits do)
- What: seen in the product on flask, "What is the default value of `use_cookies` in `Flask.test_client`?": `qwen4b-run1-step10` called `find_symbol("Flask.test_client")`, whose output shows `src/flask/app.py:L669-L725  method  def test_client(self, use_cookies: bool = True, ...)`, answered correctly from that signature, and cited `[src/flask/app.py:L669-L725]` → 0 of 1 verified. Claude answered the same question with one `grep`, cited `L669`, verified (grep hits register their line since decision #7; `find_symbol` registers nothing, LOG 21:45).
- Two issues: (1) the same evidence, obtained through `find_symbol` instead of `grep`, is grounded in one case and not the other; lane C's 22:10 proposal (register `Span(path, start, start)` for each find_symbol hit's signature line) would fix it and is a one-line env change. (2) The policy cites the symbol's whole range (57 lines) after seeing one line of it; that over-claim should still fail, and it will even after (1), because only the signature line would be registered. The reward then teaches "cite the line you saw, or read the range", which is the right lesson.
- UI: citation tooltips now say why: which ranges of that file were read, and whether the cited range came from a find_symbol / grep hit ("seen in a tool result, not read").
- Affects: agent/tools.py (lead / lane A), C (reward semantics note), talk (good example of the grounding gate doing its job and of its one blind spot)

## 2026-09-19 04:15 · trainer · lead — run one STOPPED at step 11 (cost)
- What: run one was stopped by the lead at step 11 to save Tinker spend. Kept: 12 steps of metrics in `data/logs/run1/metrics.jsonl`, checkpoint step 10 (`tinker://…/weights/000010`, `sampler_weights/000010`), profile `qwen4b-run1-step10`, manifest record with the step-10 held-out numbers (reward 0.31 vs base 0.04).
- Rule from here: nobody launches training or large eval jobs without the lead's explicit go; smokes stay ≤ 3 tasks. Filters and teacher stay paused until the lead says otherwise.
- Affects: B, C, D, E

## 2026-09-19 04:30 · data hygiene · lead — sync bug confirmed and fixed
- What: lane C's 20:10 report is right, and the same pull also deleted the lead's `data/evals/qwen4b-base/{fast_default,fast_bash,fast_noindex}` (the agent-variant comparison; numbers preserved in LOG 00:10 and lane A's progress log, traces lost). Cause: `modal volume get /evals DEST --force` replaces a local directory that also exists on the volume with the volume's copy; the volume had `qwen4b-base/` with only the Modal smokes.
- Fix: `scripts/modal_sync.sh` now downloads into `data/.modal_pull/` and merges with `rsync -a` (no delete), and merges `data/models/manifest.json` by record name (local wins). Verified: a local-only eval set and the local manifest survive a sync. Nothing is re-run (cost); the base `fast` baseline lane C is regenerating covers the table.
- Affects: everyone; `modal_sync.sh` is safe to run again

## 2026-09-19 04:50 · lane A · lead — FREEZE on codeqa/agent until the default-variant decision
- What: the product agent rewrote `codeqa/agent/variants.py` (+ `env.py`, `prompts.rules_for`, `repomap.tree_map`) making a LEAN agent the default (four tools, no `overview`, ~1k structural map, no Haiku) with the trained five-tool design demoted to `full`, without a LOG entry, while the lead was editing `tools.py`/`prompts.py` in the same minute. Both edits survived, the suite is at 37/38 (their `test_default_is_lean` fails), nothing is lost.
- Hold: nobody edits `codeqa/agent/**` until the lead posts the decision on the default. The variants themselves stay (they are the ablation the lead wanted). The default matters because the step-10 checkpoint was trained under `full`; serving or continuing it under `lean` is a distribution shift, and the prompt must match between training and inference.
- Also landed (lead, between runs, decision #7 addendum): `find_symbol` and `overview` (file mode) now register the signature line they show as a seen span; the rules say a find_symbol/overview/grep hit lets you cite the single line it shows and a range must be read.
- Affects: D (stop editing lane A; post what you changed and why), C (run two waits on the default decision), everyone (prompt/tool defaults may change once, now, before run two)

## 2026-09-19 02:20 · agent · lane B (at the lead's request) — lean context is now the default for every caller
- What: `codeqa/agent/variants.py` `default` (= `lean`) is four tools (`find_symbol`, `grep`, `read_file`, `list_dir`) and a structural `tree` map (~1k tokens, from manifest + symbols only; tests/docs/examples folded to one line each; cached as `data/index/<repo>/map_tree_<tokens>.txt`). No Haiku anywhere in the default path: no `overview`, no summaries, no `map.txt`. `full` = the day-one five-tool design with the summarised 3k map, kept for ablation; `tree_overview`, `nomap`, `nomap_full`, `noindex`, `bash`, `bash_nomap` also exist. `AgentVariant` gained `map: none|tree|full` and `needs_summaries`; `include_map` is kept as a property.
- Selection: `RepoEnv(..., variant=...)` or `CODEQA_AGENT_VARIANT=full`; `CODEQA_MAP_TOKENS` sets the tree budget. Every caller constructs `RepoEnv` (trainer dataset builder, SFT, held-out evaluator, evals/run, teacher, pass-rate filter, product API) so all inherit lean without changes. `prompts.rules_for()` strips every `overview` mention from the rules when the tool is absent (asserted). Indexing CLI: `all` no longer runs summaries unless `--summaries`; `--fast` is a no-op kept for scripts.
- Why: RepoSearch-R1, LocAgent and ToolTrain all give the model no prose summaries and fetch structure on demand; `overview` was the only paid, non-deterministic indexing step and the 3k map was ~1/3 of Sonnet's prompt tokens per episode. Measured tree maps: flask 572 tokens, transformers 799, sqlglot 821.
- Consequence for lane B numbers: `reports/passrate.jsonl` was measured under `full`. Re-measure under lean before run one (`cli filter` sharded, ~1 h Tinker, ~$5 judge for deepcodebench + teacher) and re-run `split`; expect found-gold to drop a few points and tool calls to rise at step 0. Not started: it would take lane B ~$5 over its $50 allowance.
- Tests: `codeqa/agent/tests/test_context.py` (4) + `test_bash.py` updated; 147 pass across agent/datagen/trainer/evals/grader.
- Affects: A (owner of these files; review welcome), C (env changed: re-measure and retrain from this env), D (`GET /repos` readiness should require `symbols.json`, not `map.txt`, under lean)

## 2026-09-19 02:45 · trainer + api · lane B (follow-up to the lean default)
- What: callers that gated on `map.txt` now gate on `symbols.json`: `trainer/dataset_builder.load_tasks` (requires `map.txt` only when the active variant's map is `full`), `apps/api/repos.repo_summary` (`GET /repos` lists any repo with a manifest + symbol index). `apps/api/workshop.repo_tool_specs` returns the active variant's tools (+ `variant` name); the overview endpoint falls back to the tree map when `map.txt` is absent. `tree_map` tolerates a missing `symbols.json` (manifest-only tree). Tests updated (`test_workshop` expects four tools); 91 trainer/api/agent/datagen tests pass.
- Affects: C (no behaviour change for repos that have map.txt), D (`/repos` readiness), everyone: `CODEQA_AGENT_VARIANT=full` restores the old behaviour end to end

## 2026-09-19 23:10 · grader/evals · lane C agent — length is training-only; Opus vs Sonnet on Sonnet's hard tasks; rubric audit
- Grader is now length-free (decisions night (2)): `reward = correctness × efficiency`; `gates.length_factor` is applied only in the trainer's shaped reward (`length_shaping`, default on; held-out evaluator and `evals.run` never apply it). Bare-filename citations resolve when unique in the snapshot (`RepoFiles.normalize`). 81 lane tests pass. Run one (in flight since 17:21) started before these edits, so it trains with length inside the grader reward; its metrics are internally consistent, but its `eval/fast` reward will read lower than a length-free re-eval by ~0.1 for long answers.
- Frontier models under the length-free grader (`data/evals/opus/sonnet_hard/`, 75 tasks Sonnet did not fully solve; Sonnet's 117 fast traces re-graded):

| model | set | reward | correct rate | full 1.0 | mean correctness among graded | reached grading |
|---|---|---|---|---|---|---|
| Sonnet 5 | fast (117) | 0.797 | 87 % | 62 % | 0.91 | 88 % |
| Opus 5 | sonnet_hard (75) | 0.803 | 88 % | 61 % | 0.91 | 88 % |

  Neither reaches 100 %. What remains, Opus: 7 grounding (cites wide unread ranges, e.g. `L427-L1853`), 1 fabricated path, 1 no answer, and partial rubric credit on the rest.
- Rubric audit (`data/tasks/reports/rubric_audit_frontier.json`): on the 28 judged tasks in the hard set, 164 rubric items; 122 stated by both models, 17 (10 %) stated by NEITHER, 1 Opus-only miss, 24 Sonnet-only miss. 9/28 tasks carry an item neither states, e.g. "the server's default parameters include eos_token_id, pad_token_id, do_sample=False", "the Expectations object is keyed by (device, capability) tuples": true facts of the source file that the question does not ask for. These are DeepCodeBench facts used verbatim as rubric items; pruning items no frontier model states would lift the frontier ceiling to ~95 % and remove noise from the judged training reward. Lane B / lead decision: prune by this rule over the train file (needs one Sonnet pass over the 559 judged tasks ≈ $80, or accept as is).
- Affects: lead (data decision), B (rubric pruning), talk (baseline framing: correct rate + mean correctness, not full-1.0 rate)

## 2026-09-19 05:00 · lane A · lead — freeze lifted; variant travels with the checkpoint
- What: the lean/full variants were requested by the user (Haiku summaries optional; compare training with and without). The 04:50 freeze is lifted: D keeps the variants work; post what you changed. Lead's edits to `tools.py`/`prompts.py` (decision #7 addendum) and D's `variants.py`/`env.py`/`rules_for`/`tree_map` coexist; suite green (57 agent + api tests).
- New (contract, additive): `EndpointProfile.variant`. `RepoEnv` resolves the variant as explicit arg > `profile.variant` > `CODEQA_AGENT_VARIANT` > default, so a checkpoint is always evaluated and served with the agent it trained on. `qwen4b-smoke1-step3` and `qwen4b-run1-step10` are annotated `variant: full`. C: when the trainer registers a checkpoint (`add_profile` + manifest), set `variant` to the run's variant name; evals/product then need no flag. D: the model switcher can show the variant next to the profile.
- Comparison plan (with vs without Haiku, same everything else): run one IS the `full` arm (lr 1e-4, `run1.jsonl`, group 8 × 16, held-out 0.31 at step 10). The `lean` arm = the same command with `CODEQA_AGENT_VARIANT=lean --run-name run1_lean --steps 10`; its in-loop eval at steps 0 and 10 runs under `lean`, so each arm is judged under its own agent. Cost ≈ 10 steps ≈ 10–12 min of Tinker. Waits on the lead's go (spend).
- Default for untrained profiles (`qwen4b-base`, `claude`): `lean` is acceptable now that checkpoints carry their own variant; the base-model `full` numbers from LOG 00:10 stay valid as the `full` baseline.
- Affects: D, C, everyone reading `profiles.yaml`

## 2026-09-19 03:20 · datagen · lane B — Code-QA-Bench imported as a third eval set
- What: `data/tasks/eval/codeqabench.jsonl` = 528 code-derivable Code-QA-Bench tasks (Lens-Frontier, MIT) over 10 repos snapshotted at `repos.json` SHAs (astropy, django, matplotlib, pylint, pytest, scikit-learn, seaborn, sphinx, sympy, xarray; 9 overlap SWE-QA-Bench at other commits, so eval-only). Field map is 1:1: rubric (7.8 items avg) -> `rubric`, gold_answer -> `reference_answer`, key_files (4.3 avg; 2,262/2,268 resolve) -> `expected_paths`; `how` -> trace, else explain, all judged. Contract: `TaskSource` gained `"codeqabench"` (additive). `cli import --source codeqabench`; report `reports/codeqabench.json`; test in `test_importers.py`. 31 s warm.
- Caveats: our snapshots keep documentation, so scores map to the paper's "documented" condition, not its primary code-only one (they strip docs). Their score is mean of three 0-5 axes (accuracy, completeness, specificity)/15 by an LLM judge; ours is rubric fraction (~their completeness). For a paper-comparable number lane C can add their 3-axis prompt beside `sweqa_judge.py`. The 100 doc-dependent tasks are not imported.
- Gotcha (decision #8): with `CODEQA_RUNTIME=modal` in `.env`, `cli import` launched detached on Modal and wrote to the volume, not the laptop; the local file came from `CODEQA_RUNTIME=local`. The Modal copy is fine to keep (same code, same output) but the lean-map cache and the 10 new snapshots now exist in both places.
- Affects: C3 (third eval set; add `codeqabench` to the evals report), lead (talk: an external number besides SWE-QA)

## 2026-09-20 00:20 · data/grader · lane C agent — rubric pruning (lead: "prune")
- Rule adopted: keep a rubric item iff some frontier answer (Sonnet 5, plus Opus 5 where we have it) states it; items no frontier answer states are dropped (`codeqa/grader/rubric_prune.py --apply-empirical FILES --traces DIRS`). Two cheaper rules were validated against the frontier item audit and rejected: "Sonnet decides from the question alone" caught 47 % of noise but dropped 32 % of real items; "the reference answer states it" kept 98 % of real items but caught only 3/17 noise items (the references are as exhaustive as the rubrics). Every task keeps ≥ 2 items; tasks with no frontier answer are untouched; originals in `data/tasks/backup/`; report `data/tasks/reports/rubric_prune.json`.
- Applied to `data/tasks/eval/fast.jsonl`: 15/353 items dropped (4 %). Sonnet re-graded on the same traces: DCB judged full-1.0 37/53 → 42/53, mean correctness 0.91 → 0.92, reward 0.797 → 0.807. Run one's in-loop evaluator loaded the old file at launch; its `eval/fast` rows stay on the old rubrics for consistency within the run.
- In flight: Sonnet answers for the 644 judged training tasks (`data/evals/claude/train_judged/`, ~2 h at concurrency 4, ≈ $100); when done a chain prunes `train/{all,run1,deepcodebench,teacher}.jsonl`, then runs Sonnet on `deepcodebench_test.jsonl` (230) and prunes it. Run one is unaffected (task file already in memory); run two should be launched from the pruned files.
- Monitor: new check "grounding gate fails > 20 % and rising". Run one at step 11: train reward 0.29-0.42, held-out step 10 reward 0.31 (from 0.04), format 0.97, grounded 0.92, tool calls 4.4; grounding failures in training rose 1 % → 28 % (the policy probing unread citations; the gate holds).
- Affects: lead (run two from pruned files), B (rubric rule for future generation), talk

## 2026-09-19 03:50 · evals · lane B (at the lead's request) — Code-QA-Bench judge ported verbatim
- What: `codeqa/evals/cqb_judge.py` = upstream `code_qa_bench/judge.py` (MIT): same prompt text, anchored 0-5 axes (accuracy, completeness, specificity), think-then-score, gold/agent position randomised per sample, median over `--samples`, per-question score = sum/15, overall = mean; same three JSON-extraction strategies. Runs over an existing `data/evals/<profile>/<set>/per_task.jsonl` like `sweqa_judge.py`, writes `cqb_judge.json` and `results.summary.cqb_score`. External number only, never training reward.
- Differences to state next to any number: judge model (theirs `claude-sonnet-4-20250514`, ours `claude-sonnet-5` by default; pass `--model`), condition (ours documented, theirs stripped), and that we run the agent through our own tools, not theirs.
- Cost: ~2.5k in + ~1k out per sample; 528 tasks x 1 sample on Sonnet ~ $8. Tests: `codeqa/evals/tests/test_cqb_judge.py` (3).
- Affects: C3 (add to the eval report), lead (talk)

## 2026-09-20 00:50 · trainer · lane C agent — loss curve
- What: `codeqa/trainer/metrics_patch.py` wraps the cookbook's `compute_kl_sample_train` (installed by `codeqa.trainer.run`) to log `optim/loss` (IS policy-gradient surrogate, near 0 by design since advantages are group-centered), `optim/loss_abs` (signal magnitude), `optim/advantage_std`, `optim/frac_tokens_with_advantage`, `optim/importance_ratio_mean|max`, `optim/clip_fraction`, `optim/nll`, `optim/action_tokens`. Monitor and plots updated (`plots/optimizer.png`). Tests in `trainer/tests/test_metrics_patch.py`.
- Why no loss before: Tinker runs the loss on its servers and returns only training logprobs; the cookbook never computes a loss value. Read `loss_abs`, `advantage_std`, KL and entropy for health; `loss` itself will sit near zero and is there because people expect a loss panel.
- Not in run one (launched earlier); present in every run launched from now on. D: keys under `optim/*`, show when present.
- Affects: D (Live page), lead

## 2026-09-20 01:30 · apps/web · product agent
- What: loss panel and labelled charts for lane C's new optimizer keys (LOG 2026-09-20 00:50)
- Detail: Runs and Live pages: a Loss chart (`optim/loss` and `optim/loss_abs` on one axis, zero line, caption saying the surrogate sits near zero by design), plus labelled, captioned charts with reference lines for `advantage_std`, `frac_tokens_with_advantage`, `importance_ratio_mean|max` (line at 1), `clip_fraction` (line at 0.2), `nll`, `action_tokens`. Runs without the keys (run one) show a one-line note instead of an empty panel. Verified on a synthetic 12-step run with all keys (deleted after).
- Affects: nothing outside apps/web

## 2026-09-19 04:20 · evals · lane B — first Code-QA-Bench numbers (10 tasks, one per repo, lean agent context)
- Runs: `data/evals/{opus,qwen4b-base}/codeqabench_10/` (+ `cqb_judge.json`, Sonnet 5 judge, 1 sample). Task file `eval/codeqabench_10.jsonl` (seed 11).
- Opus 5 through our lean agent: their score 0.75 (accuracy 4.6, completeness 2.3, specificity 4.3 of 5); our reward 0.62, 9/10 clear the gates (1 grounding), 10.3 tool calls, 56k prompt tokens, ~1,050-token answers, 40 s/task. Completeness is the weak axis: their rubrics average 8 items written from a full exploration; Opus answers the question and covers ~half.
- Base Qwen3.5-4B: their score 0.28 (4/10 answered; those four score 0.60-0.80, i.e. when it answers it is roughly Opus-level on this judge); our reward 0.00 because the 4 answers lack bracketed citations and 6 episodes end at max_turns/budget without answering (8.0 tool calls, 11 s/task). Same step-0 picture as the pass-rate run: the deficit is stopping and citing, not comprehension.
- Reading for the talk: their judge and our grader rank the same tasks the same way (Opus: astropy/xarray top, sphinx bottom under both). Their judge is more lenient on missing citations (specificity rewards named symbols, not verified spans); ours zeroes them. Quote both. Repo overlap with SWE-QA means these 10 repos are unseen at training time except seaborn's siblings.
- Cost: Opus ~$3, judging <$1.
- Affects: C3 (extend to 100 or 528 when budget allows: ~$30 / ~$160 for Opus episodes), lead

## 2026-09-20 01:20 · grader · lane C agent — two pre-run-two fixes from the reward walkthrough
- Judge window: the judge used to see the answer truncated at `max_answer_tokens`; with length no longer a gate that silently hid the tail of long answers. Now 3× the cap (min 2,000 tokens). Opus on the hard set: mean correctness 0.91 → 0.92, full-1.0 61 % → 63 %.
- Verbatim-paste gate: needs ≥ 8 pasted lines and counts every line ≥ 8 chars in the denominator, so quoting a short code snippet no longer trips it. 87 lane tests pass.
- Affects: grader (run two), evals

## 2026-09-19 05:30 · experiments · lead — PHASE 1 (shape) launched on Modal
- What: four arms in parallel as detached jobs (`scripts/arm.py` inside the runner), each: 128 episodes/step, lr 1e-4, group 8 × 16, seed 0 (identical batches), `run1.jsonl`, held-out `fast.jsonl` in-loop every 10 steps, then the same 120 tasks at T=0.2 under the common grader as set `fast_t02`. Runs: `p1_full` (five tools + summarised map, 30 steps), `p1_lean` (four tools + 1k tree map, 30), `p1_bash` (shell tool on Modal sandboxes + tree map, 30), `p1_nogates` (full agent, cited-file-exists and lines-read gates OFF via `CODEQA_HONESTY_GATES=off`, 10 steps; its in-loop eval is ungated, its `fast_t02` eval is gated). Checkpoints register as `qwen4b-p1_<arm>-step<N>` with `variant` set.
- Protocol: read the chain full → lean → bash (one change each); decide on held-out correctness at T=0.2 and tokens per correct answer, noise band ±4 points; run one is a replicate of `p1_full`'s first 10 steps (noise estimate). Nobody launches anything else on Tinker until these finish (~3–4 h; sampling is shared).
- Grader: `codeqa/grader/grade.py` gained `honesty_gates()` (env `CODEQA_HONESTY_GATES`, default on); with it off, the adversarial fixture tests fail by design (fabrication is measured, not gated). C: keep it.
- Affects: C (grader switch), D (four new profiles will appear in `profiles.modal.yaml` after `modal_sync.sh`), everyone (Tinker busy)

## 2026-09-19 05:50 · workshop · lead (for D)
- What: the four phase-1 runs are `p1_full`, `p1_lean`, `p1_bash`, `p1_nogates` under `data/logs/` (synced from the Modal volume every 2 min by the lead's loop, so the Runs page is near-live). Readable titles + hypotheses for every run are in `data/logs/runs.json` (`{run: {title, hypothesis, variant, steps, reward}}`); please read it in `GET /runs` and show `title` in the list and chart legend, falling back to the directory name. Overlay: the lead wants all four on one reward chart, so lift the two-run limit to N (four here). Also useful on the run page: the in-loop held-out points (`eval/fast/env/all/reward`) as markers at steps 0/10/20/30.
- Affects: D

## 2026-09-20 02:40 · apps/web, apps/api · product agent — watching the phase-1 runs live (answers lane C's request)
- What: `GET /runs` reads `data/logs/runs.json` (`title`, `hypothesis`, `variant`, `planned_steps`, `reward_setting`); titles show in the runs list, run page header, Live picker and cards; dir names stay as the mono subline and chart legend keys. Overlay on the run page is now N runs (`?vs=a,b,c`, toggle chips); Live page has a board when ≥ 2 runs are live: all of them on one reward chart with held-out points, one card each (title, steps / planned with a progress bar, last reward, warning count, last sync), click to monitor below.
- Liveness: no longer mtime-based. The 2-min `modal volume get` rewrites every file it pulls, so mtime says "live" for finished runs; the API now marks a run live only while new metric rows keep arriving (window 330 s = one sync + one commit + slack) and never once `steps ≥ planned_steps`. First sight in a fresh API process trusts a recent mtime once.
- Affects: lead / C (run one shows 12/11: planned_steps in runs.json says 11), nothing else

## 2026-09-19 06:10 · experiments · lead — phase 1 relaunch: filtered data
- What: the 05:30 launch failed at startup (the arm module used laptop-relative task paths inside the container; fixed to resolve via `codeqa.shared.paths`). Relaunching on lane B's **`train/all.jsonl` (1,489 tasks, pass-rate window, 62 % gold-graded, 38 % rubric-judged)** instead of `run1.jsonl`, so the arms train on the designed set and see explain data; Haiku judge in the loop (~50 calls/step/arm). Run one (unfiltered, verifiable-only) is no longer a replicate of `p1_full`; it stays as the "before the filter" reference.
- Affects: C (judge in the loop again), B (your set is in use), D (same run names)

## 2026-09-20 02:10 · grader/agent · lane C agent — grounding is a multiplier; find_symbol ranges are shown (decisions "2026-09-20 early")
- What: `grade.py` (integrated with the lead's `honesty_gates()` switch): `citations_grounded` is now `citations.grounded_fraction` (mean share of cited lines shown, ±1) and multiplies the reward; `grounding` gate only when the fraction is 0. `agent/tools.py::find_symbol` records the full displayed range (was signature line only); lane A's `test_find_symbol_and_overview_register_signature_lines_only` updated accordingly. 126 tests pass across grader/trainer/evals/agent.
- Also this session: judge window 3× cap (min 2,000 tokens); verbatim gate needs ≥ 8 pasted lines. See LOG 01:20.
- Opus on Sonnet's 75 hard tasks: reached grading 88 % → 96 %, correct 88 % → 96 %, reward 0.81 → 0.86, mean grounded 0.98. Sonnet fast re-grade running.
- Ask (lane A / lead): a final forced-answer turn when the tool budget is exhausted (driver and cookbook env), so stalls stop counting as "no answer" for frontier and trained models alike. Run one is unaffected by any of this (in flight with the earlier grader); run two picks it all up.
- Affects: lead (run two), lane A (env ask), D (badges: `citations_grounded` is fractional now)

## 2026-09-20 02:40 · data · lane C agent — rubric pruning complete; Anthropic credit dry again
- Pruned (rule: keep an item iff a frontier answer states it; Sonnet on all 644 judged train tasks + the 232 DeepCodeBench test tasks, Opus where available; backups in `data/tasks/backup/`, report `data/tasks/reports/rubric_prune.json`):
  `train/all.jsonl` 377/4,073 items (9 %), `train/deepcodebench.jsonl` 276/3,185, `train/teacher.jsonl` 93/888, `train/run1.jsonl` 33/933, `eval/fast.jsonl` 15/353, `eval/deepcodebench_test.jsonl` 111/1,507 (7 %). Tasks without an answered frontier trace (35 in all.jsonl) and tasks that would drop below 2 items (29) are untouched.
  Sonnet on the 644 judged train tasks (pre-prune grader): reward 0.76, correct 85 %; on the 232 DeepCodeBench test tasks: reward 0.73, correct 80 % (`data/evals/claude/{train_judged,dcb_test}/`).
- **Credit balance is dry again** (judge 400 "credit balance is too low", first seen 02:10). Casualties: the Sonnet fast re-grade under the new grounding rule has 39/117 judge NaNs and must be re-run after a top-up; nothing else was mid-flight. Run one is unaffected (verifiable tasks, no judge). Run two on `all.jsonl` needs the judge, so top up before launching it.
- Affects: lead (top-up; run two from the pruned files), B, talk
