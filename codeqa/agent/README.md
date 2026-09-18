# codeqa/agent — the core agent design

This folder answers the three spec questions. Everything else in the repo points a model at what lives here.

## 1. How is the task presented to the agent?

`prompts.py`

- **System prompt** (`SYSTEM_RULES`): the rules. Cite every claim as `[path:L10-L20]`; cite only lines you read; use the map, prefer `overview`/`find_symbol` before `grep`; read line ranges, not files; several tool calls per turn are fine; you have N tool calls; reply without a tool call to finish; keep it under M tokens; answer shape (direct answer, details with citations, Sources block).
- **User prompt** (`user_prompt`): the repo map (`indexing/repomap.py`, token-capped ~3k) followed by `Question: ...`.
- **Budget warning** (`BUDGET_WARNING`): appended to the last allowed tool result.
- Budgets come from the task record (`Task.effective_budget()`, defaults per task type in `shared/contracts.py`).

`env.py` (`RepoEnv.initial_messages`) assembles these into the message list. **It must prepend the renderer's tool-declaration prefix** (`renderer.create_conversation_prefix_with_tools([t.to_spec() for t in tools], system_prompt)`); the cookbook env does not do this for you, and without it the model invents its own tool syntax. Five tool schemas cost roughly 600–900 prompt tokens. The same list is used by the trainer (through the cookbook), the teacher (Claude), and the product (vLLM).

## 2. What tools does the agent need?

`tools.py` — five read-only tools, declared with the cookbook `@tool` decorator so the same specs feed Tinker, vLLM, and Claude:

| Tool | Params | Returns | Cap |
|---|---|---|---|
| `overview(path=".")` | path | directory/file summary, children with one-liners, top symbols | 60 lines |
| `find_symbol(name, kind=None, file_pattern=None)` | name, kind, glob | `path:Lstart-Lend  kind  signature` | 20 hits |
| `grep(pattern, file_pattern=None)` | regex, glob | `path:L55: text`, grouped by file, deduped | 30 hits / 10 files |
| `read_file(path, start, end)` | path, range | header `path:Ls-Le`, lines `L41 \| code`, footer with total lines (or a cut note with the next line to read) | 150 lines |
| `list_dir(path)` | path | `dir/ name` / `file name (N lines)` | 50 entries |

Design points: results are always line-numbered so citations are copied, not invented; `find_symbol` and `overview` are the entry points (index-backed); `grep` is the literal fallback; no shell, no semantic search.

## 3. How are the tools implemented?

- `indexing/snapshot.py` — repo@sha → clean folder + manifest (C1). LF-normalized so line numbers are stable.
- `indexing/index.py` — tree-sitter symbols with line ranges (C2). `find_symbol` and `overview` read this, never re-parse.
- `indexing/summaries.py` — one paragraph per directory, one line per large file, from Haiku, once. `overview` reads this.
- `indexing/repomap.py` — renders the token-capped map for the prompt from the two above.
- `indexing/strip_docstrings.py` — `<repo_id>__nodoc` variant for structural tasks.
- `tools.py` — `RepoTools`: the five `@tool` methods over the snapshot folder (`rg --json` for grep when ripgrep is installed, otherwise a pure-Python scan with the same output; index lookups for symbols; plain file reads for ranges). Counts calls and errors, records every `read_file` span in `files_read`, appends the budget note.
- `curation.py` — `Caps` (defaults: 150 read lines, 30 grep hits / 10 files, 20 symbol hits, 60 overview lines, 50 list entries), ranking (source before tests/docs), nearest-path suggestions, glob-or-substring file patterns, budget note.
- `env.py` — `RepoEnv(task, profile)`: one instance per episode; `initial_messages()` (client-agnostic), `specs()`/`tools()`, `files_read()`, `make_cookbook_env(reward_fn(history, env))` for the trainer (prepends the renderer tool prefix itself), `trace_from_history()` → C6, `RepoEnv.from_question()` for ad-hoc product questions.
- `driver.py` — `run_episode(env, client, on_event)` for the product, teacher and evals: the same loop with any `ModelClient` (Tinker sampling, Anthropic, OpenAI-compatible); emits C9 events; returns a `Trace` (C6); `save_trace()` writes `data/traces/<run>/`.
- `tests/` — offline tests on the flask snapshot: `uv run pytest codeqa/agent`.

Errors are forgiving on purpose: nearest paths on a miss, first chunk on an oversize range, budget warning when one call remains. Every wasted turn is one the model must learn to avoid.

## What is fixed here and what is tunable

Fixed (contracts): tool names and output formats, the citation regex, the message roles. Tunable (config): caps, budgets per task type, the map token limit, the exact rules wording, thinking on/off via the endpoint profile.

Design rationale and precedents: `docs/agent_design.md`.
