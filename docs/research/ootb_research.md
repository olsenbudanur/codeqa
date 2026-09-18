# Off-the-shelf tooling survey: RL-trained code Q&A agent on Tinker

Date: 2026-09-18. All facts below were checked against live docs/repos today unless marked "(from memory, verify)".

Project recap: RL-train Qwen3.5-4B/9B via Tinker to explore a repo tarball with read-only tools (list, read range, grep, symbol lookup) and answer with `[path:L10-L20]` citations. 1500-3000 tasks with programmatic + LLM-judge graders. Serve merged weights on Modal/vLLM (OpenAI-compatible), FastAPI + React product with a live research log.

---

## 1. Tinker API (gates the schedule)

**Recommended pick and why.** Use `tinker` SDK + `tinker-cookbook` (v0.5.7, 2026-09-03, Apache-2.0, Python >=3.11) and build the environment with the cookbook's `tinker_cookbook.tool_use` library: `@tool` for our four read-only tools, `build_agent_tool_env(...)` for the multi-turn env, and a plain `reward_fn(history)` for grading. The `search_tool` recipe (Search-R1 replication) is a near-exact template: it already trains `Qwen/Qwen3.5-4B` in non-thinking mode, multi-turn, with a stateful tool object shared across a group, format penalty, and context-overflow handling. Both `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` are supported and priced. Loss masking of tool/observation tokens is handled for you. Nothing on the Tinker side should take more than half a day; the real work is our tools, task set, and reward function.

### (a) Supported models (exact ids, 64K context unless noted)

From https://tinker-docs.thinkingmachines.ai/tinker/models/ (per-million-token prices: prefill / sample / train):

| Model | Tinker id | Ctx | Prefill | Sample | Train |
|---|---|---|---|---|---|
| **Qwen3.5-4B** | `Qwen/Qwen3.5-4B` | 64K | $0.33 | $1.005 | $0.737 |
| **Qwen3.5-9B** | `Qwen/Qwen3.5-9B` | 64K | $0.66 | $1.995 | $1.463 |
| Qwen3.5-9B-Base | `Qwen/Qwen3.5-9B-Base` | 64K | $0.66 | $1.995 | $1.463 |
| Qwen3-8B | `Qwen/Qwen3-8B` | 32K | $0.195 | $0.60 | $0.44 |
| Qwen3.5-35B-A3B-Base | `Qwen/Qwen3.5-35B-A3B-Base` | 64K | $0.54 | $1.335 | $1.177 |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | 64K | $0.54 | $1.335 | $1.177 |
| Qwen3.6-27B | `Qwen/Qwen3.6-27B` | 64K | $1.86 | $5.595 | $4.103 |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` (+`:peft:262144` 256K) | 64K | $1.86 | $5.595 | $4.103 |
| Qwen3.5-397B-A17B | `Qwen/Qwen3.5-397B-A17B` (+`:peft:262144`) | 64K | $3.00 | $7.50 | $6.60 |
| GPT-OSS-20B | `openai/gpt-oss-20b` | 32K | $0.18 | $0.45 | $0.396 |
| GPT-OSS-120B | `openai/gpt-oss-120b` | 32K | $0.33 | $0.84 | $0.737 |
| Others | Inkling, Inkling-Small, Nemotron-3.x, GLM-5.3, Kimi-K2.6, DeepSeek-V3.1 | | | | |

- Deprecations page (https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/): Qwen3.5-35B-A3B, Qwen3.5-27B, Qwen3-30B-A3B, Qwen3-8B-Base, Qwen3-30B-A3B-Base retired 2026-06-12; Qwen3.6-27B retired 2026-09-02 (replacement Qwen3.8-27B). **Qwen3.5-4B and Qwen3.5-9B are not on the deprecation list.**
- Model attributes: `tinker_cookbook.model_info.get_recommended_renderer_name("Qwen/Qwen3.5-4B")` returns `"qwen3_5"`; `get_recommended_renderer_names` returns `("qwen3_5", "qwen3_5_disable_thinking")`. Both 4B and 9B are flagged `is_chat=True`, vision-capable, version "3.5".
- Context: 4B/9B are 64K on Tinker (no `:peft:262144` variant listed for them). With a 32K `max_trajectory_tokens` cap (the search_tool default) that is fine.
- Gotcha: the AGENTS.md in the cookbook says never hardcode renderer names; use `model_info.get_recommended_renderer_name()`.

### (b) RL examples in the cookbook (paths)

Repo: https://github.com/thinking-machines-lab/tinker-cookbook (Apache-2.0). Install:

```bash
pip install tinker                      # SDK (also gives the `tinker` CLI)
uv pip install tinker-cookbook          # or: uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
pip install "tinker-cookbook[wandb]"    # wandb is an optional extra, not core
export TINKER_API_KEY=...
```

Core RL (`tinker_cookbook/rl/`):
- `types.py`: `Action = list[int]`, `Observation = tinker.ModelInput`, `StepResult(reward, episode_done, next_observation, next_stop_condition, metrics, logs)`, `Transition(ob, ac, reward, episode_done, ...)`, `Trajectory(transitions, final_ob, stop_reason)`, abstract `Env.initial_observation() -> (Observation, StopCondition) | InitialObservationOverflow`, `Env.step(action, *, extra=None) -> StepResult`, `EnvGroupBuilder.make_envs() -> Sequence[Env]` plus optional `compute_group_rewards(trajectory_group, env_group)` and `logging_tags()`, `RLDataset.get_batch(index) -> Sequence[EnvGroupBuilder]`, `@chz.chz RLDatasetBuilder.__call__() -> (RLDataset, RLDataset | None)`.
- `message_env.py`: `MessageEnv` (message-level: `initial_observation() -> list[Message]`, `step(message) -> MessageStepResult(reward, episode_done, next_messages, metrics, logs, next_stop_condition)`) and `EnvFromMessageEnv(types.Env)` adapter which calls `renderer.parse_response(action)`, then `message_env.step(...)`, then **re-renders the whole conversation** with `renderer.build_generation_prompt(next_messages)`; enforces `max_trajectory_tokens` through `_exceeds_context_limit(observation_length)` at initial observation and after every step.
- `train.py`: `@chz.chz Config` with fields `learning_rate, dataset_builder, model_name, recipe_name, max_tokens, log_path, eval_every=20, save_every=20, evaluator_builders, load_checkpoint_path, renderer_name, wandb_project, wandb_name, kl_penalty_coef=0.0, kl_discount_factor=0.0, kl_reference_config, loss_fn="importance_sampling" (also "ppo", "cispo"), loss_fn_config, num_substeps=1, lora_rank=32, temperature=1.0, compute_post_kl, remove_constant_reward_groups=False, rollout_error_tolerance, termination, enable_trace, async_config: AsyncConfig(max_steps_off_policy, groups_per_batch), stream_minibatch_config: StreamMinibatchConfig(groups_per_batch, num_minibatches), base_url, ttl_seconds=604800, rolling_save_every, rolling_ttl_seconds=7200, num_groups_to_log=4, rollout_json_export=True, max_steps`. Entry point: `await tinker_cookbook.rl.train.main(config)`.
- Other files: `rollouts.py`, `rollout_runner.py`, `rollout_limits.py`, `rollout_presets.py`, `rollout_logging.py`, `problem_env.py` (single-turn Q&A helper `ProblemEnv`), `data_processing.py` (advantages, training data assembly), `metrics.py`, `multiturn_weight_assignment_test.py` (see (c)).

Tool-use library (`tinker_cookbook/tool_use/`, marked experimental): `README.md`, `tools.py`, `types.py`, `agent_tool_message_env.py`.

```python
from typing import Annotated
from tinker_cookbook.tool_use import tool, simple_tool_result, build_agent_tool_env, ToolResult

class RepoTools:                       # stateful, one per task (or shared per group)
    def __init__(self, repo_root): ...
    @tool
    async def read_file(self, path: Annotated[str, "Repo-relative path"],
                        start: Annotated[int, "1-based first line"],
                        end: Annotated[int, "1-based last line"]) -> ToolResult:
        """Read a line range of a file."""
        return simple_tool_result(text)

env = build_agent_tool_env(
    renderer=renderer, tools=[t.list_dir, t.read_file, t.grep, t.find_symbol],
    initial_messages=[{"role":"system","content":SYS},{"role":"user","content":q}],
    reward_fn=my_reward_fn,            # receives the full message history at episode end
    max_turns=12, max_trajectory_tokens=32*1024, max_generation_tokens=1024,
    failed_parse_reward=-0.1, context_overflow_reward=-0.1,
)
```

Exact signature: `build_agent_tool_env(renderer, tools, initial_messages, reward_fn, *, rollout_config=None, model_name=None, max_turns=None, failed_parse_reward=-0.1, terminate_on_parse_error=True, max_tool_calls=None, max_trajectory_tokens=None, max_generation_tokens=None, context_overflow_reward=-0.1, terminate_on_length=None, parse_error_policy=None) -> EnvFromMessageEnv`. The underlying `AgentToolMessageEnv(MessageEnv)` dataclass has `tools, initial_messages, max_turns, reward_fn, failed_parse_reward, terminate_on_parse_error, max_tool_calls, parse_error_policy, tool_execution="parallel"|"sequential", termination_policy`. `ToolResult(messages, should_stop=False, metrics, metadata)`; `to_spec()` yields `{name, description, parameters}` JSON schema from `Annotated` hints and the docstring.

Episode termination: assistant message with **no tool calls** (treated as the final answer), all tool calls failed to parse and `terminate_on_parse_error`, a tool returns `should_stop=True`, `max_turns` reached, or tool-call budget exhausted. Mixed valid/invalid calls run the valid ones with no penalty. Context overflow ends the episode with `context_overflow_reward`.

Recipes (`tinker_cookbook/recipes/`): `search_tool/` (Search-R1: `search_env.py`, `tools.py`, `train.py`, `embedding.py`, `offline_eval.py`; classes `SearchEnvGroupBuilder(EnvGroupBuilder)`, `SearchRLDataset(RLDataset)`, `@chz.chz SearchR1DatasetBuilder`, reward `TextAnswerReward(gold_answers, format_coef=0.1)`; run `python -m tinker_cookbook.recipes.search_tool.train`; defaults `model_name="Qwen/Qwen3.5-4B"`, `renderer_name="qwen3_5_disable_thinking"`, `lora_rank=32`, `learning_rate=4e-5`, `batch_size=512`, `group_size=8`, `max_turns=5`, `max_tokens=1024`, `max_trajectory_tokens=32*1024`, `format_coef=0.1`, `stream_minibatch=False`, `num_minibatches=4`, `wandb_project`, `wandb_name`), `code_rl/` (sandboxed exec), `harbor_rl/` (Harbor sandbox tasks, `HarborBashTool`), `rubric/` (LLM grader with rubric, see section 6), `verifiers_rl/` (runs Prime Intellect verifiers envs on Tinker: `python -m tinker_cookbook.recipes.verifiers_rl.train vf_env_id=env-id vf_env_args='{}'`), `multiplayer_rl/`, `math_rl/`, `preference/`, `distillation/`, `prompt_distillation/`, `sdft/`, `forecasting/`, `true_thinking_score/`, `vlm_classifier/`, `audio/`, plus `rl_basic.py`, `rl_loop.py`, `sl_basic.py`, `sl_loop.py`. Note: `twenty_questions` and `text_arena` are referenced in `tests/recipes/` but are no longer directories under `recipes/`.

Docs: https://tinker-docs.thinkingmachines.ai/cookbook/recipes/search-tool/ , https://tinker-docs.thinkingmachines.ai/cookbook/rl/ , tutorials at `/tutorials/cookbook-abstractions/env-and-envgroupbuilder/`, `/tutorials/cookbook-abstractions/rl-with-config/`, `/tutorials/advanced/rl-hyperparams/`.

### (c) Renderers, Qwen chat template, tool calls, loss masking

- Files: `tinker_cookbook/renderers/{base.py, qwen3.py, qwen3_5.py, qwen3_8.py, llama3.py, gpt_oss.py, deepseek_v3.py, kimi_k2*.py, glm5_3.py, nemotron3.py, role_colon.py, tml_v0.py}`. Classes `Qwen3_5Renderer` and `Qwen3_5DisableThinkingRenderer` registered as `"qwen3_5"` and `"qwen3_5_disable_thinking"`. Custom renderers via `register_renderer(name, factory)`.
- Methods: `build_generation_prompt(messages) -> ModelInput`, `build_supervised_example(messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE) -> (tokens, weights)`, `parse_response(tokens) -> (Message, termination)` with `.is_clean`/`.is_stop_sequence`, `get_stop_sequences()`, `create_conversation_prefix_with_tools(...)` to inject tool specs. `TrainOnWhat` options: `LAST_ASSISTANT_MESSAGE, LAST_ASSISTANT_TURN, ALL_ASSISTANT_MESSAGES, ALL_MESSAGES, ALL_TOKENS, CUSTOMIZED`.
- Tool-call format for Qwen3.5 is **XML, not JSON**: `<tool_call>\n<function=name>\n<parameter=arg>\nvalue\n</parameter>\n</function>\n</tool_call>` (`_format_tool_call_xml`, `_parse_qwen3_5_tool_call_xml`), which `parse_response` turns into `message["tool_calls"]`. Consecutive tool responses are grouped into one `<|im_start|>user` block (`groups_consecutive_tool_responses = True`). Thinking content is stripped/trimmed in `_postprocess_parsed_message`. This matches vLLM's `qwen3_coder`/`qwen3_xml` parsers at serve time.
- **Masking:** `tinker_cookbook/rl/multiturn_weight_assignment_test.py` asserts that in multi-turn trajectories only sampled action tokens get mask=1; observation tokens including tool responses get mask=0 (e.g. `expected = [0,0,0,0,1,1,1,0,0,1,1,0,1,1,1]`, `TestEndToEndToolUseRollout.test_mask_only_on_agent_tokens`). So tool results are excluded from the loss automatically in RL.
- Gotcha: `verifiers_rl` README warns that Qwen3 Instruct tokenization strips `<think>` sections from prior turns, which custom parsers may penalize. Use `qwen3_5_disable_thinking` for a short-horizon agent unless we want reasoning tokens.
- Standalone `tml-renderers` library exists (https://tinker-docs.thinkingmachines.ai/cookbook/inkling/tml-renderers/) but requires torch>=2.10; the cookbook already depends on it.

### (d) Sampling

- `service_client = tinker.ServiceClient()`; `training_client.save_weights_and_get_sampling_client(...)` or `service_client.create_sampling_client(base_model=..., model_path="tinker://...")`.
- `SamplingClient.sample(prompt: ModelInput, num_samples: int, sampling_params: SamplingParams, include_prompt_logprobs=False, topk_prompt_logprobs=0) -> Future[SampleResponse]`; `sample_async(...)`; `compute_logprobs(prompt)`; `get_tokenizer()`; `get_base_model()`. `SamplingParams(max_tokens, temperature, top_p, stop, seed)`. `ModelInput.from_ints(tokens)`; per-sample `tokens` (and logprobs) on the response. API ref: https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/
- Concurrency: issue all requests then await (`asyncio.gather(*[sampling_client.sample_async(...) ...])`); docs show 8 prompts sequential 14.2s vs concurrent 2.2s. No numeric cap is published; AGENTS.md says "Tinker is designed for high request concurrency from a single Python process". There **is** a per-account "concurrent sampler weights limit" (number of distinct checkpoints hot at once); hitting it pauses sampling with "Reason: concurrent sampler weights limit hit" (seen in a third-party issue). Keep one live sampler per run. Session metrics dashboard shows in-flight requests, tokens/s, P50/P90/P99 (https://tinker-docs.thinkingmachines.ai/tinker/session-metrics/).
- Tinker path format: `tinker://<run_id>:train:<seq>/sampler_weights/<name>`.

### (e) Export/download LoRA and merge for vLLM

```python
from tinker_cookbook import weights
adapter_dir = weights.download(tinker_path=sampler_path, output_dir="/tmp/adapter")
# Option 1: merged HF model (no LoRA dependency), W = W_base + B@A * alpha/rank
weights.build_hf_model(base_model="Qwen/Qwen3.5-4B", adapter_path=adapter_dir, output_path="/tmp/merged")
# Option 2: PEFT adapter (adapter_config.json + adapter_model.safetensors, ~146MB at rank 16)
weights.build_lora_adapter(base_model="Qwen/Qwen3.5-4B", adapter_path=adapter_dir, output_path="/tmp/peft")
# weights.publish_to_hf_hub(...) also exists
```

- Save from training: `training_client.save_weights_for_sampler(name, ttl_seconds=None)` (weights only) vs `save_state(name)` (weights + optimizer). `RestClient.list_checkpoints`, `list_user_checkpoints`, `set_checkpoint_ttl_from_tinker_path_async`, `publish_checkpoint_from_tinker_path_async`. CLI: `tinker checkpoint ...`, `tinker billing`, `tinker session`, `tinker auth`. Checkpoints expire per TTL (RL `Config.ttl_seconds` default 7 days); storage $0.10/GB-month.
- Docs: https://tinker-docs.thinkingmachines.ai/tutorials/deployment/export-hf/ , https://tinker-docs.thinkingmachines.ai/tutorials/deployment/lora-adapter/ , https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/
- **Gotcha (big):** serving a PEFT adapter on Qwen3.5 with vLLM `--enable-lora` is flaky because Qwen3.5 is a hybrid Gated DeltaNet model (vLLM issues #38085, #36478 for 2B, #36372 for 27B, #47639 regression). Plan on `build_hf_model` (merged, ~9-10GB for 4B) and serve as a plain model. Merging needs the base model download and CPU RAM for a 4B model; run it on a Modal function with a Volume, not a laptop.

### (f) LoRA hyperparameters and LR helper

- `tinker_cookbook.hyperparam_utils.get_lr(model_name, is_lora=True)`: `5e-5 * 10 (LoRA) * (2000/hidden_size)**0.0775` for Qwen (0.781 exponent for Llama); Qwen3.5 4B/9B hidden sizes are hardcoded in `_KNOWN_HIDDEN_SIZES`. `get_lora_lr_over_full_finetune_lr()` returns 10.0. `get_lora_param_count("Qwen/Qwen3.5-4B", lora_rank=32)` = 72.9M (rank 8: 18.2M, rank 128: 291.6M).
- LoRA primer (https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/): rank 32 default; apply to all matrices (attention-only underperforms); "LoRA performs equivalently to FullFT for reinforcement learning even with small ranks"; LoRA is less tolerant of very large batches; optimal LR does not depend on rank.
- RL hyperparams tutorial (`/tutorials/advanced/rl-hyperparams/`): LR range 1e-6 to 1e-4, "start with learning_rate=1e-5, group_size=4, kl_penalty_coef=0.05"; `group_size` 2-16; `num_substeps` 1-4; temperature 0.7-1.0; advantages are group-mean-centered with **no std normalization**; `remove_constant_reward_groups=True` drops zero-signal groups. The search_tool recipe uses lr 4e-5, rank 32, group 8, batch 512 for Qwen3.5-4B.
- Loss options: `importance_sampling` (default), `ppo`, `cispo`.

### (g) Pricing / limits

Per-million-token prices in (a); cached prefill is 80% off; prices rose 2026-07-17 (prefill/sample +~50%, train +~10%). Rough budget: 2,000 tasks x group 8 x 12 turns x ~4K tokens per re-rendered prompt = heavy prefill; at $0.33/M prefill for 4B, 100 steps of 64 groups is on the order of low hundreds of dollars. No RPM/TPM limits are published; contact support if throttled.

### (h) WandB from the cookbook

`Config.wandb_project` / `Config.wandb_name` feed `tinker_cookbook.utils.ml_log.setup_logging(log_dir=config.log_path, wandb_project=..., config=config, wandb_name=...)`. Set `WANDB_API_KEY`. Install the `wandb` extra. Also written to `log_path`: `metrics.jsonl`, `config.json`, `checkpoints.jsonl`, `train_iteration_NNNNNN_rollout_summaries.jsonl` (per-trajectory rewards + per-step metrics/logs), `train_iteration_NNNNNN_logtree.json`, and `train_iteration_NNNNNN.html` (human-readable rollout viewer). `num_groups_to_log=4` controls how many groups get full transcripts. (https://tinker-docs.thinkingmachines.ai/cookbook/rl/rl-logging/)

### Tinker gotchas summary
1. Env re-renders the full conversation every step: token cost grows quadratically with turns; keep `max_turns` ~8-12 and tool outputs small.
2. Only sampled tokens are trained; tool outputs are masked automatically. Do not hand-roll masks.
3. Qwen3.5 tool calls are XML; make sure any citation-format parsing does not collide with `<tool_call>` tags.
4. Use the `qwen3_5_disable_thinking` renderer for speed unless we deliberately want reasoning.
5. Sampling concurrency is fine; hot-checkpoint count is limited per account.
6. Merge (not adapter) for vLLM on Qwen3.5.
7. torch>=2.10 and transformers<=5.5.4 pins in the cookbook; use a fresh venv.

---

## 2. Repo map and symbol index

**Recommended pick and why.** `tree-sitter-language-pack` (1.20.0, released 2026-09-14, MIT, prebuilt wheels for 371 grammars) plus ~100 lines of our own symbol extractor (walk `function_definition`/`class_definition`/`method` nodes per language, or reuse aider's `tags.scm` query files). This gives a per-file `path -> [(kind, name, line)]` index in minutes and no external binaries. For the initial "file tree + top symbols under a token budget" prompt, either write a trivial ranker or drop in `RepoMapper` (standalone port of aider's PageRank repo map, MIT). `universal-ctags --output-format=json` is the zero-Python-code fallback.

Options:
1. **tree-sitter-language-pack** — `pip install tree-sitter-language-pack`; `from tree_sitter_language_pack import get_parser, get_language; parser = get_parser("python"); tree = parser.parse(src)`. Python >=3.10, abi3 wheels for mac/linux/windows. https://pypi.org/project/tree-sitter-language-pack/ . Gotchas: `tree-sitter-languages` (grantjenks) is unmaintained since Feb 2024 and fails on Python 3.13, do not use. tree-sitter core 0.26 removed `Language.query()`; older code (aider's repomap copy, grep-ast) breaks (OpenViking issue #5122). Use `tree_sitter.Query(language, src)` or pin `tree-sitter<0.26`.
2. **aider RepoMap** — `pip install aider-chat`; `from aider.repomap import RepoMap`; constructor `RepoMap(map_tokens=1024, root=None, main_model=None, io=None, repo_content_prefix=None, verbose=False, max_context_window=None, map_mul_no_files=8, refresh="auto")`; `get_repo_map(chat_files, other_files, mentioned_fnames, mentioned_idents)`. Needs stubs for `main_model.token_count(text)` and `io.read_text/tool_output/tool_error/tool_warning`, plus networkx, diskcache, grep_ast and the `aider/queries/tree-sitter-language-pack/*.scm` files. Apache-2.0. https://github.com/Aider-AI/aider/blob/main/aider/repomap.py . Gotcha: pulls the whole aider dependency tree (litellm etc.); half a day of trimming if you want it light.
3. **RepoMapper** (pdavis68) — standalone extraction of aider's map: `python repomap.py . --map-tokens 2048` or import `RepoMap`; token counting via tiktoken; MIT; last updated 2025-07-13. https://github.com/pdavis68/RepoMapper . Gotcha: unpinned deps, same tree-sitter 0.26 risk.
4. **grep-ast** — `pip install grep-ast` (0.9.0, May 2025, Apache-2.0); `TreeContext` renders matches with enclosing scopes, `filename_to_lang`. https://pypi.org/project/grep-ast/ . Useful for a "grep with structural context" tool output, not for indexing.
5. **universal-ctags** — `ctags --output-format=json --fields=+n -R .` emits JSON Lines `{name, path, line, kind, scope}` per symbol; needs the binary (brew/apt). https://docs.ctags.io/en/latest/man/ctags-json-output.5.html . Gotcha: JSON output requires a libjansson build; `-f` quirks with JSON.

---

## 3. Read-only code tools for agents

**Recommended pick and why.** Write our own four tools (~150 lines) on top of the `rg` binary (`rg --json` or `-n -C`) and the tree-sitter index from section 2, copying SWE-agent's interface conventions: 100-line windows for `read_file`, search results capped at ~50 with a "too many results, refine" message, directory listing with depth. Nothing off the shelf is both read-only and dependency-light: SWE-agent's ACI tools are bash scripts inside a container, mini-swe-agent is bash-only, OpenHands' `file_editor view` lives inside a large SDK, Serena needs per-language LSP servers.

Options:
1. **SWE-agent tool bundles** (MIT; project now maintenance-only, superseded by mini-swe-agent) — bundles `registry`, `windowed` (`open <file> [line]`, `goto`, `scroll_up/down`, 100-line windows), `search` (`search_file`, `search_dir`, `find_file`, max 50 results), `edit_*`. https://swe-agent.com/latest/config/tools/ . Copy the semantics, not the code (they are shell scripts that assume a container).
2. **python-ripgrep** — `pip install python-ripgrep` (0.0.9, MIT, Rust reimplementation, no `rg` binary): `search(patterns=[...], paths=[...], globs=[...], max_count=..., after_context=..., before_context=...)`, `files()`. https://github.com/indent-com/python-ripgrep . Gotchas: no `-C` combined context, **no case-insensitive flag, no line numbers, no count**. That makes it unusable for a `grep` tool that must return `path:line`. `ripgrep-rs` (Feb 2026) is a similar PyO3 wrapper. Prefer `subprocess.run(["rg","--json",...])` and parse.
3. **Serena** (oraios, MIT) — LSP-backed MCP server with `find_symbol`, `find_referencing_symbols`, `get_symbols_overview`, 40+ languages. https://github.com/oraios/serena . Gotcha: needs a language server installed per language and a running MCP process per repo; >half a day to make hermetic for 100s of tarballs.
4. **mcp-server-tree-sitter** (wrale) — symbol/AST queries built on tree-sitter-language-pack. https://github.com/wrale/mcp-server-tree-sitter . Steal its per-language symbol node lists.
5. **mini-swe-agent** (MIT) — bash-only; no tool-calling. https://github.com/SWE-agent/mini-swe-agent . Not applicable except for its prompt/observation truncation patterns.
6. **smolagents / OpenHands SDK** — generic `Tool` wrappers; OpenHands has a "read-only explorer" agent with `file_editor view`. Heavy dependencies; skip.

---

## 4. Existing code Q&A datasets

**Recommended pick and why.** Use **SWE-QA-Bench** (720 free-text questions, 15 Python repos, Apache-2.0, ships `repo_commit.txt` + `clone_repos.sh`) as the held-out eval and as the source of question style; use **SWE-rebench / SWE-Gym issue-PR pairs** (section 5) as the raw material for our own verifiable tasks; and copy **Code-QA-Bench's answer-first generation recipe** (a tool-equipped agent explores code, produces a verified gold answer with locations, then derives the question). Nothing public gives 1500+ repo-level questions with programmatically checkable file/line answers, so we must generate.

| Dataset | Size / languages | License | Repo snapshots? | Verifiable? | URL |
|---|---|---|---|---|---|
| SWE-QA-Bench (Peng et al., 2509.14635) | 720 Q, 15 Python repos (astropy, django, flask, matplotlib, pylint, pytest, requests, scikit-learn, sphinx, sympy, xarray, ...); 77.6% cross-file | Apache-2.0 | commit hashes + clone script | No (free text, LLM judge in `score/llm-as-a-judge.py`) | https://github.com/peng-weihan/SWE-QA-Bench , https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark |
| SWE-QA MCQ (Elkoussy & Perez, LREC 2026, 2604.24814) | 9,072 MCQ from 12 SWE-bench Python repos (declaration-and-call, interacting-entity patterns) | not stated | not stated | Yes (MCQ) | https://arxiv.org/abs/2604.24814 (no public data link found; flag) |
| Code-QA-Bench (2605.29277) | 528 code-derivable + 100 doc-dependent, 10 Python repos | CC BY 4.0 | not stated | LLM judge (accuracy/completeness/specificity) | https://arxiv.org/abs/2605.29277 (framework "open-source", link not in abstract; flag) |
| CoReQA (2501.03447) | issues+comments from 176 repos, 4 langs | not stated | no | LLM judge, 5 aspects | https://arxiv.org/abs/2501.03447 |
| RepoQA (evalplus) | 500 needle-function tests, 5 langs x 10 repos | Apache-2.0 | code context per test (not full repo) | Yes (syntactic similarity >= 0.8) | https://github.com/evalplus/repoqa , `pip install repoqa` |
| CodeQueries (HF thepurpleowl/codequeries) | 260k rows, Python (ETH Py150), 52 CodeQL queries | Apache-2.0 | full files inline | Yes (answer + supporting-fact spans) | https://huggingface.co/datasets/thepurpleowl/codequeries |
| LongCodeQA (LongCodeBench, HF Steefano/LCB) | MCQ (4 options) from GitHub issues, Python, up to 1M ctx | not shown | packed repo context | Yes (MCQ) | https://huggingface.co/datasets/Steefano/LCB , https://github.com/Zteefano/long-code-bench |
| CodeRAG-Bench | code-gen retrieval tasks | CC BY-SA 4.0 | n/a | n/a | https://github.com/code-rag-bench/code-rag-bench (not QA; skip) |
| CrossCodeEval / RepoBench | code completion | various | n/a | n/a | skip (not QA) |
| CodeRepoQA (2412.14764) | issue-thread QA, large | not checked | no | LLM judge | https://arxiv.org/pdf/2412.14764 |

Gotchas: SWE-QA-Bench answers are long prose (up to 24K chars) and include no citations, so it is style/eval only. CodeQueries questions are static-analysis style ("Unused import") and file-scoped, useful only for cheap "find the span" warmup tasks. RepoQA's needle descriptions are function-level; a `find_symbol`-only agent solves them, so use them as an easy curriculum tier.

---

## 5. Task mining from git history

**Recommended pick and why.** Do not mine GitHub yourself in a 2-day window. Load **nebius/SWE-rebench** (21,336 issue-PR rows, CC BY 4.0, per-row `license_name`, `repo`, `base_commit`, `patch`, `test_patch`, `problem_statement`, `hints_text`; Python, 3,400+ repos) and **SWE-Gym/SWE-Gym** (2,438 rows, MIT, 11 Python repos), parse touched files from `patch` headers, filter to permissive `license_name`, and pick repos with 500-5000 files. For non-Python coverage add **SWE-bench/SWE-bench_Multilingual** (300 tasks, 42 repos, 9 languages: lombok, rubocop, caddy, laravel, redis, fmt, tokio, preact, ...) and **Multi-SWE-bench** (1,632 tasks; Java, TS, JS, Go, Rust, C, C++). Each row already is an (issue, PR patch, touched files, base_commit) triple.

```python
from datasets import load_dataset
ds = load_dataset("nebius/SWE-rebench", split="test")          # 21,336 rows
ok = ds.filter(lambda r: r["license_name"] in {"MIT","Apache-2.0","BSD-3-Clause","BSD-2-Clause","ISC"})
# touched files: re.findall(r"^diff --git a/(\S+) b/", row["patch"], re.M)
```

Other options if fresh mining is needed:
- **swebench/collect** (MIT): `print_pulls.py <owner/repo> out.jsonl --token`, `build_dataset.py prs.jsonl out.jsonl --token` (emits `instance_id, patch, test_patch, problem_statement, base_commit`), `get_tasks_pipeline.py` over many repos; needs `GITHUB_TOKENS`. https://github.com/swe-bench/SWE-bench/tree/main/swebench/collect . Gotcha: SWE-bench itself took a week to scrape; API rate limits dominate.
- **SWE-smith** (NeurIPS 2025): turns any repo into task generators (localization, repair). https://github.com/SWE-bench/SWE-smith . Overkill for QA.
- **SWE-rebench pipeline** is described (permissive-license SPDX filtering) but the collection code is not open; the data is. https://nebius.com/blog/posts/swe-rebench-dataset
- **PyDriller** (`pip install pydriller`): `for c in Repository(path).traverse_commits(): c.modified_files[i].new_path/old_path/change_type/diff`; local-git only, no issue linkage. https://pydriller.readthedocs.io/
- **GitHub GraphQL**: `pullRequest.closingIssuesReferences` + `files` gives (issue, PR, files) in one query per PR; 5000 points/hour. Half a day to get robust.
- **nemo-curator**: general data curation stages; no ready GitHub issue/PR collector. Skip.

Curated permissive repo lists to reuse quickly: SWE-bench Multilingual's 42 repos (https://www.swebench.com/multilingual.html), SWE-QA-Bench's 15 Python repos with pinned commits, SWE-rebench's `repo`+`license_name` columns filtered by size.

---

## 6. LLM judge and grading

**Recommended pick and why.** Copy three tiny prompt templates instead of importing a framework: (1) Inspect AI's `DEFAULT_MODEL_GRADED_FACT_TEMPLATE` (expert answer vs submission, reasoning first, `GRADE: C/P/I` with partial credit), (2) openai/evals `fact.yaml` A-E subset/superset/disagree scheme (good for reference-anchored correctness), and (3) Tinker's `recipes/rubric` pattern (`<context>`, `<completion_to_grade>`, `<rubric>` tags, `<score>0..1</score>` per rubric item, averaged, plus a format penalty `total = format_coef*(format_score-1) + avg_score`). Run the judge through Tinker's `TinkerMessageCompleter` (as the rubric recipe does with `Qwen/Qwen3.6-35B-A3B`) or any OpenAI-compatible endpoint. Pair it with a programmatic citation verifier (regex `\[([^\]:]+):L(\d+)(?:-L?(\d+))?\]`, path exists in tarball, line range within file, cited range overlaps gold spans) so most reward is deterministic.

Templates to copy:
- Inspect AI `src/inspect_ai/scorer/_model.py`: `DEFAULT_MODEL_GRADED_QA_TEMPLATE` (Task/Submission/Criterion), `DEFAULT_MODEL_GRADED_FACT_TEMPLATE` ("Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation. Does the submission contain the content in the expert answer?"), instructions ending `GRADE: $LETTER` with `C/P/I`, grade regex `DEFAULT_GRADE_PATTERN`. `model_graded_qa(template, instructions, grade_pattern, include_history, partial_credit, model, reducer="majority")`. https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/scorer/_model.py (MIT). `pip install inspect-ai` if we want the scorer object itself.
- openai/evals `evals/registry/modelgraded/fact.yaml` (A subset, B superset, C same, D disagreement, E differ but immaterial; `choice_strings: ABCDE`) and `closedqa.yaml` (Y/N with reasoning first). https://github.com/openai/evals/blob/main/evals/registry/modelgraded/fact.yaml (MIT).
- Tinker `tinker_cookbook/recipes/rubric/data.py`: `Rubric(rubric_str, extraction_regex=r"<score>(.*?)</score>", grader_output_format_instruction="Please output your score between 0 and 1 wrapped in <score> ... </score>")`, `RubricBasedDatapoint(convo, rubric_items)`, `get_grader_prompt(convo)`, `extract_score()` returns 0.0 on parse failure; `env.py` has `RubricGradedEnv`, `RubricGradedEnvGroupBuilder`, `RubricGradedDatasetBuilder`. https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/rubric
- verifiers `vf.JudgeRubric(judge_prompt=...)` caches the judge response in rollout state so several reward functions share one call. https://docs.primeintellect.ai/verifiers/components
- DeepEval `GEval` and promptfoo `llm-rubric` exist but add dependencies for no gain here.

Anti-length-bias practices (2025-26 sources): atomic yes/no rubric criteria rather than holistic scores; include the line "Do not prefer longer answers; extra unsupported detail should lower the score" (halves verbosity bias, does not eliminate it); grade against a reference with subset/superset labels so padding cannot score; cap graded answer length (truncate at N tokens and penalize overflow programmatically); for pairwise comparisons randomize A/B order and average both orders; reasoning before verdict; sample judge at temperature 0 and majority-vote 3 if cheap; keep a small human-labeled calibration set to measure judge agreement.

---

## 7. Alternative RL frameworks (backup)

**Recommended pick and why.** If Tinker plumbing stalls, the fastest self-hosted backup is **TRL v1.13 `GRPOTrainer`** with `environment_factory=` (stateful env class exposing tool methods with Google-style docstrings) or `tools=[...]`, `max_tool_calling_iterations`, vLLM colocate on one GPU for Qwen3.5-4B LoRA. It needs `transformers>=5.2` and a GPU box, roughly half a day. Second is Tinker's own `verifiers_rl` recipe, which lets a verifiers `ToolEnv` run on Tinker compute (still Tinker). Everything else needs multi-GPU clusters or a retrieval service.

| Framework | Working multi-turn tool-calling GRPO example for small Qwen? | Setup effort | Notes / URL |
|---|---|---|---|
| **TRL GRPOTrainer** (Apache-2.0) | Yes: `tools=[fn]` (stateless) and `environment_factory=Env` (stateful, `reset()`, `get_reward()`), example with `Qwen/Qwen3-0.6B`, `chat_template_kwargs={"enable_thinking": False}`; OpenEnv hub envs; templates auto-patched for Qwen3 | ~half day on 1 GPU | https://huggingface.co/docs/trl/grpo_trainer ; vLLM server mode needs `--weight-transfer-config '{"backend":"nccl"}'` |
| **verifiers + prime-rl** (Apache-2.0) | Yes: `examples/basic/wiki-search` trains Qwen3-4B-Instruct-2507, LoRA r8/alpha32, ToolEnv multi-turn, Hermes parser; `uv run rl @ examples/basic/wiki-search/rl.toml` | >half day: 8 GPUs (6 inference/2 train) in the example, ChromaDB index, OpenAI key for judge, Python 3.12, submodules | https://github.com/PrimeIntellect-ai/prime-rl , https://github.com/PrimeIntellect-ai/verifiers . Tinker's `verifiers_rl` recipe runs the same env on Tinker instead |
| **ART (OpenPipe)** (Apache-2.0) | Yes: MCP-RL (Qwen2.5-3B), ART-E LangGraph (Qwen2.5-7B), ART-E serverless (Qwen3.6-27B), RULER judge rewards; `pip install openpipe-art`; Unsloth+vLLM | ~half day on 1 GPU (LocalBackend) or W&B ServerlessBackend | https://github.com/OpenPipe/ART . Qwen3.5 not explicitly listed; Unsloth support for hybrid Qwen3.5 is the risk |
| **SkyRL** | Yes but heavy: search example uses Qwen2.5-3B-Instruct, `generator.max_turns=4`, 16 GPUs, Ray, separate retrieval server | >1 day | https://docs.skyrl.ai/docs/examples/search |
| **veRL agent loop** | Yes: `gsm8k_tool_agent_loop.py`, `run_qwen2_5_3b_gsm8k_tool_agent_*.sh`; agent loop since v0.4.2 | >1 day (Ray, FSDP/Megatron, multi-GPU) | https://verl.readthedocs.io/en/latest/advance/agent_loop.html |
| **rLLM** | Fork of veRL; single-GPU Qwen3 multi-turn agents claimed; thinner docs | >half day | https://github.com/rllm-org/rllm (from memory, verify) |

Gotcha for all self-hosted options: Qwen3.5's Gated DeltaNet architecture requires very recent transformers/vLLM (model card says "main branch"); if a framework pins older versions, fall back to `Qwen/Qwen3-4B` for the backup path.

---

## 8. Serving on Modal

**Recommended pick and why.** Start from Modal's `vllm_inference` example (https://modal.com/docs/examples/vllm_inference): CUDA 12.9 image, `vllm==0.21.0` pinned, `@app.server` on port 8000, two Volumes (`huggingface-cache` at `/root/.cache/huggingface`, `vllm-cache` at `/root/.cache/vllm`), `scaledown_window=15*MINUTES`, `startup_timeout=10*MINUTES`, `target_concurrency=100`, `FAST_BOOT` toggling `--enforce-eager`. Swap in our **merged** Qwen3.5 model from a Volume and serve with tool calling on. Use a PEFT adapter only if the merged path is blocked. Add GPU snapshots later if cold start matters for the demo.

vLLM serve line for a merged Qwen3.5-4B (per the Qwen3.5-4B model card and vLLM recipes):

```bash
vllm serve /models/qwen35-4b-codeqa-merged \
  --served-model-name qwen-codeqa --port 8000 --tensor-parallel-size 1 \
  --max-model-len 65536 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --api-key $VLLM_API_KEY
```

- Tool-call parser: Qwen's card says `--tool-call-parser qwen3_coder`; vLLM docs now also ship `qwen3_xml` (newer streaming XML parser, "the more advanced tool call parser for qwen3 models"). Both parse the `<tool_call><function=...>` XML that Tinker's renderer trains on. Use `qwen3_xml` if streaming tool calls to the UI, `qwen3_coder` otherwise. Known bug class: tool calls emitted inside `<think>` were dropped in vLLM 0.19 (issue #39056), another reason to disable thinking.
- LoRA adapter path: `vllm serve Qwen/Qwen3.5-4B --enable-lora --lora-modules codeqa=/models/peft --max-lora-rank 32`. Gotcha: Qwen3.5 GatedDeltaNet LoRA target-module mismatches and crashes (vLLM issues #38085, #36478, #36372, #47639). Prefer merged weights.
- Cold start: example says compile takes "tens of seconds to a few minutes", ~10 s when loaded from the vllm cache Volume; weights load from the HF cache Volume. GPU memory snapshots (`enable_memory_snapshot=True`, `experimental_options={"enable_gpu_snapshot": True}`, `@modal.enter(snap=True)` starts vLLM with `--enable-sleep-mode`, warms up, calls `/sleep`; `@modal.enter(snap=False)` calls `/wake_up`) give 2-10x faster cold starts (45 s -> 5 s for a 0.5B model), but are experimental, only kick in after a few `modal deploy` cold starts, and the example pins vLLM 0.15.1. https://modal.com/docs/examples/lfm_snapshot , https://modal.com/docs/guide/memory-snapshots
- Volumes (https://modal.com/docs/guide/volumes): `vol = modal.Volume.from_name("repo-tarballs", create_if_missing=True)`; mount `@app.function(volumes={"/data": vol})`; upload with `modal volume put repo-tarballs ./tarballs /tarballs` or `with vol.batch_upload() as b: b.put_directory(...)`; `vol.commit()` / `vol.reload()` for cross-container visibility; read-only via `vol.with_mount_options(read_only=True)`; v2 volumes remove the 500k-inode cap. Good fit for both the corpus tarballs and the merged model.
- Sandboxes (https://modal.com/docs/guide/sandboxes): `sb = modal.Sandbox.create(image=img, volumes={"/data": vol}, app=app)`; `sb.exec("bash","-c",cmd)`; `sb.filesystem` API (older file ops deprecated 2026-03-09); filesystem snapshots, tunnels, timeouts. Only needed later if we allow execution.
- Modal's `modal` extra exists in tinker-cookbook (`[all]`), suggesting the merge step can run as a Modal function too.

---

## 9. Rollout / trajectory viewer

**Recommended pick and why.** Zero-setup: Tinker already writes `train_iteration_NNNNNN.html` (logtree viewer with prompts, responses, reward breakdown) and `*_rollout_summaries.jsonl` (per-trajectory `total_reward`, `final_reward`, per-step `reward/metrics/logs`) to `log_path`. Put our reward components into `StepResult.metrics`/`logs` and they show up there. For a filterable "list of tool calls + final answer + reward parts" view, an ~80-line Streamlit page over the jsonl is faster than any hosted tracer. Use WandB Tables if the team is already in WandB.

Options:
1. **Tinker logtree HTML + jsonl** — built in; open the html per iteration; `num_groups_to_log=4` limits detail. https://tinker-docs.thinkingmachines.ai/cookbook/rl/rl-logging/
2. **WandB Tables** — `wandb.log({"rollouts": wandb.Table(columns=["task","turns","tool_calls","answer","r_format","r_cite","r_judge","total"], data=rows)})` per iteration; cookbook already opens the run. Weave (`pip install weave`, `@weave.op` on the tool functions) renders tool spans with Args/Result and there is a "Weave Traces" panel for RL rollouts in W&B Models. https://docs.wandb.ai/weave/guides/tracking/view-agent-activity
3. **Inspect AI `inspect view`** — best transcript UI (messages, tool calls, scores, timeline), but only reads `.eval` logs produced by Inspect tasks; use it if we run the held-out eval through Inspect. https://inspect.aisi.org.uk/log-viewer.html
4. **Langfuse** (MIT, self-host via docker compose; acquired by ClickHouse Jan 2026) and **Phoenix** (`pip install arize-phoenix; phoenix serve`, OTLP) — full tracing + scores, but they are production observability stacks; >half a day to wire rewards as scores. https://github.com/langfuse/langfuse , https://github.com/Arize-ai/phoenix
5. **Streamlit/Gradio** — `streamlit run viewer.py` reading the jsonl; sort by reward component; render tool calls as expanders. Under an hour.

---

## 10. Product frontend pieces

**Recommended pick and why.** Code viewer: **CodeMirror 6 via `@uiw/react-codemirror`** (MIT) with `@codemirror/language-data` for lazy grammars; it is light, read-only friendly, and has clean APIs for jump-to-line (`EditorView.scrollIntoView`) and line decorations. Dictation: **Web Speech API** behind a feature check, using `react-speech-recognition` (updated 2026-01-03) or a 30-line hook. Streaming: **plain SSE** with FastAPI's built-in `fastapi.sse.EventSourceResponse` (FastAPI >= 0.135.0) or `sse-starlette`; the Vercel AI SDK adds a protocol we would have to emulate for little benefit.

Code viewer options:
1. **@uiw/react-codemirror** — `npm i @uiw/react-codemirror @codemirror/language-data @codemirror/view @codemirror/state`. Get the view via `onCreateEditor={(view)=>ref.current=view}` or `ref.current.view`. Jump: `view.dispatch({ effects: EditorView.scrollIntoView(view.state.doc.line(n).from, { y: "center" }) })`. Highlight: a `StateField<DecorationSet>` producing `Decoration.line({ class: "cm-cited" })` for lines L10-L20, toggled by a `StateEffect`. Read-only: `readOnly` prop + `EditorView.editable.of(false)`. https://github.com/uiwjs/react-codemirror . Gotcha: `height` must be set or it grows unbounded; use `basicSetup={{ lineNumbers: true }}`.
2. **Monaco (`@monaco-editor/react`)** — `editor.revealLineInCenter(n)`, `editor.createDecorationsCollection([{ range, options: { isWholeLine: true, className: "cited" } }])`. Full VS Code feel but ~MBs of workers and CSP hassle; shiki-monaco integration exists for better grammars. https://shiki.style/packages/monaco
3. **shiki / react-shiki** — static highlighting, `transformerNotationHighlight` or `meta` for line highlights; no scrolling API (give each line an id and `scrollIntoView`). Best for read-only snippets in the research log, not the main viewer. https://github.com/AVGVSTVS96/react-shiki
4. **react-syntax-highlighter** — `wrapLines` + `lineProps` for highlighting; oldest, largest bundles; skip.

Dictation: `const SR = window.SpeechRecognition || window.webkitSpeechRecognition;` Chrome/Edge/Opera full support; Safari 14.1+ (webkit prefix, can do on-device); Firefox only behind `dom.webspeech.recognition.enable`; requires HTTPS or localhost. `npm i react-speech-recognition` gives `useSpeechRecognition()` (`transcript`, `listening`, `browserSupportsSpeechRecognition`) and `SpeechRecognition.startListening({ continuous: true })`. https://github.com/JamesBrill/react-speech-recognition , https://caniuse.com/speech-recognition . Gotcha: Chrome sends audio to Google servers; show a fallback text box for Firefox.

Streaming:
- FastAPI >= 0.135.0: `from fastapi.sse import EventSourceResponse, ServerSentEvent`; `@app.get("/run/stream", response_class=EventSourceResponse) async def run() -> AsyncIterable[ServerSentEvent]: yield ServerSentEvent(data={"type":"tool_call",...}, event="step", id="1")`. Data is JSON-encoded automatically; works with POST too. https://fastapi.tiangolo.com/tutorial/server-sent-events/
- Older FastAPI: `pip install sse-starlette`; `EventSourceResponse(gen())` yielding dicts. https://github.com/sysid/sse-starlette
- Client: `EventSource` for GET, or `fetch` + `ReadableStream` reader for POST bodies. Emit typed events: `step` (tool call + result summary), `token` (answer delta), `citations`, `done`.
- Vercel AI SDK 6 (`useChat`) streams over SSE using its UI-message-stream protocol with typed data parts (`data-*`), reconcilable by id. It is nice for token streaming but our backend must emit that exact protocol; simpler to send our own SSE events. https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data

---

## 11. Corpus prep

**Recommended pick and why.** Snapshot with the GitHub tarball endpoint at a pinned SHA (no history, one HTTP call), then strip with linguist's `vendor.yml` regexes + a generated/binary/size filter applied through `pathspec`. Under an hour for a few hundred repos with a token.

Snapshot:
- `GET https://api.github.com/repos/{owner}/{repo}/tarball/{sha}` returns 302 to `codeload.github.com`; any reachable commit SHA works; 60 req/h unauthenticated, 5,000/h with a token (`Authorization: Bearer $GH_TOKEN`). Direct: `https://codeload.github.com/{owner}/{repo}/tar.gz/{sha}`. The tarball top-level dir is `{owner}-{repo}-{shortsha}/`; strip one component. https://docs.github.com/en/rest/repos/contents#download-a-repository-archive-tar
- Fallback for very large repos or non-GitHub: `git init && git remote add origin URL && git fetch --depth 1 origin <sha> && git checkout FETCH_HEAD && git archive -o snap.tar HEAD`. `git archive --remote` does not work against GitHub.
- `ghapi` (fastai) wraps the REST API but is unnecessary for one endpoint; `requests` + `tarfile` is enough.

Ignore lists:
- linguist `vendor.yml` (~130 regexes: `(^|/)node_modules/`, `(^|/)vendor/`, `(^|/)dist/`, `(\.|-)min\.(js|css)$`, `bower_components`, `\.yarn/releases/`, `jquery...js`, `bootstrap...`, `MathJax/`, `Godeps/_workspace/`, `Carthage/`, `gradlew$`, `mvnw$`, ...). Raw: https://raw.githubusercontent.com/github-linguist/linguist/main/lib/linguist/vendor.yml . Plus `generated.rb` heuristics (minified, `*.pb.go`, `package-lock.json`, `*.min.*`, Xcode project files, generated protobuf, `__generated__`), and `.gitattributes linguist-generated`/`linguist-vendored` markers inside the repo.
- Apply with `pathspec` (`pip install pathspec`, 0.12.1): `spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)`; plus `binaryornot` (`is_binary(path)`) or a NUL-byte sniff; drop files > 1 MB, lockfiles, images/fonts/archives by extension, `.git/`, `docs/_build`, `*.ipynb` outputs (or strip outputs). Also reject repos whose filtered file count is outside 500-5000.
- Cache each cleaned snapshot as `{owner}__{repo}__{sha}.tar.gz` on the Modal Volume, with a `manifest.json` (file list, sizes, language per extension, symbol counts) used by the repo-map tool.

Gotchas: GitHub tarballs exclude submodule contents; monorepos with vendored third-party dirs (e.g. `third_party/`) are not all in vendor.yml, add `third_party/`, `external/`, `deps/`, `_vendor/`. Windows line endings and BOMs will shift line numbers if you normalize after computing gold spans; normalize once at snapshot time and never again.

---

## Time-risk flags (>half a day)
- Serving a LoRA adapter (not merged) for Qwen3.5 on vLLM.
- Serena / LSP-based tools across many languages.
- verifiers+prime-rl, SkyRL, veRL self-hosted training.
- Fresh GitHub issue/PR mining with swebench/collect at 3,000-task scale.
- Aider RepoMap with full dependency tree (trimmable, but fiddly).
- Langfuse/Phoenix as the trajectory viewer.
