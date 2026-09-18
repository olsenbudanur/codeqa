# Contracts

The ten seams between components. Every component reads or writes one of these and nothing else. Schemas live as pydantic models in `shared/contracts.py`; these snippets are the shapes, the models are the truth.

## The contracts (all live in `shared/contracts.py`)

### C1 · Repo id and snapshot manifest
```
repo_id = "<owner>__<repo>__<sha7>"            # pallets__flask__a1b2c3d
data/repos/<repo_id>/                            # extracted files, LF line endings
data/repos/<repo_id>/manifest.json
{
  "repo_id": "pallets__flask__a1b2c3d",
  "url": "https://github.com/pallets/flask", "sha": "a1b2c3d...",
  "files": [{"path": "src/flask/app.py", "lang": "python", "lines": 2210, "bytes": 81234}],
  "dropped": {"vendored": 40, "binary": 12, "oversize": 3}
}
```

### C2 · Index files
```
data/index/<repo_id>/symbols.json                # read it via agent.indexing.index.load_symbols()
{"repo_id": "pallets__flask__a1b2c3d",
 "symbols": [{"name": "validate_session", "kind": "function",   # function | class | method
              "path": "src/auth/session.py", "start": 41, "end": 67,
              "parent": null, "signature": "def validate_session(token: str) -> Session"}]}

data/index/<repo_id>/summaries.json
{"src/auth": "Session creation, validation and expiry. Entry point is validate_session.",
 "src/auth/session.py": "Session dataclass and the validate/expire functions."}

data/index/<repo_id>/map.txt                     # plain text, <= 3000 tokens
src/
  auth/            Session creation, validation and expiry
    session.py     Session, validate_session, SessionExpired
```

### C3 · Tool API and driver
```
overview(path=".")                          -> summary, children with one-line summaries, top symbols
list_dir(path)                              -> "dir/  name" | "file  name  (N lines)"      cap 50
find_symbol(name, kind=None, file_pattern=None)
                                            -> "path:L41-L67  function  validate_session(token)"  cap 20
grep(pattern, file_pattern=None)            -> "path:L55: text", grouped by file, deduped   cap 30 hits, 10 files
read_file(path, start, end)                 -> "L41 | code"  numbered lines, cap 150, footer "(total 312 lines)"

errors    -> "ERROR not_found: src/auth/sesion.py. Did you mean: src/auth/session.py"
budget    -> last line "1 tool call remaining. Answer next turn." when one remains

class RepoEnv:                              # one instance per episode  (codeqa/agent/env.py)
    def __init__(self, task: Task, profile: EndpointProfile, caps: Caps = DEFAULT_CAPS)
    @classmethod from_question(repo_id, question, profile, task_type="explain", budget=None)   # product, no gold
    def initial_messages(self) -> list[Message]   # system rules + map + question; client-agnostic (no tool prefix)
    def specs(self) -> list[ToolSpec]        # the five tool schemas, passed as `tools=` to any client
    def tools(self) -> list[FunctionTool]    # cookbook @tool objects
    def files_read(self) -> list[Span]       # for the grounding check; also tool_calls_made, tool_errors, budget
    def make_cookbook_env(self, reward_fn, *, max_generation_tokens=None, max_trajectory_tokens=None)
                                             # trainer path; async reward_fn(history, env) -> (float, dict)
    def trace_from_history(self, history, seconds=0.0) -> Trace   # cookbook history -> C6 (no token counts)

def run_episode(env: RepoEnv, client: ModelClient, on_event=None, *, temperature=1.0, max_tokens=None) -> Trace
def save_trace(trace: Trace, run: str = "dev") -> Path          # data/traces/<run>/<task_id>__<profile>.json

# The Tinker path prepends renderer.create_conversation_prefix_with_tools(specs, system_prompt) itself
# (TinkerChatClient.chat and RepoEnv.make_cookbook_env, via clients.tinker.with_tool_prefix); OpenAI/Anthropic get `tools=`.
```

### C4 · Prompt, answer format, citation regex
```
system rules (exact text in codeqa/agent/prompts.py):
  - Cite every factual claim as [path:L10-L20]. Cite only lines you have read.
  - Answer as soon as the evidence is sufficient. Budget: {max_tool_calls} tool calls.
  - Reply without a tool call to give your final answer. Keep it under {max_answer_tokens} tokens.

answer shape:
  <direct answer, 1-3 sentences>
  <details, each claim followed by a citation>
  Sources:
  - path:L41-L67  what it shows

CITATION_RE = r"\[([^\]\s:]+):L(\d+)(?:-L(\d+))?\]"
```

### C5 · Task record
```json
{
  "task_id": "flask-0042", "repo_id": "pallets__flask__a1b2c3d", "split": "train",
  "question": "Where is the session cookie signed, and what key is used?",
  "task_type": "trace",
  "source": "teacher",
  "source_id": null,
  "grading": {
    "expected_paths": ["src/flask/sessions.py"],
    "expected_symbols": [],
    "expected_literal": null,
    "reference_answer": "...",
    "rubric": ["names SecureCookieSessionInterface", "cites the signing call"],
    "required_citations": [{"path": "src/flask/sessions.py", "start": 310, "end": 340}]
  },
  "budget": {"max_tool_calls": 10, "max_turns": 8, "max_answer_tokens": 400}
}
```
`task_type`: locate | value | enumerate | trace | explain. `source`: deepcodebench | codescout | structural | teacher | sweqa | sweqa_pro. `source_id`: the upstream row id. `expected_symbols`: CodeScout entities as `path:Class.method`, any-of match through `find_symbol`. DeepCodeBench `facts` go into `rubric` unchanged.

### C6 · Episode trace
```json
{
  "task_id": "flask-0042", "profile": "qwen4b-run2-step40",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
               {"role": "assistant", "content": "...", "thinking": "...",
                "tool_calls": [{"name": "find_symbol", "args": {"name": "validate_session"}}]},
               {"role": "tool", "name": "find_symbol", "content": "src/auth/session.py:L41-L67 ..."},
               {"role": "assistant", "content": "Validation happens in ... [src/auth/session.py:L41-L67]"}],
  "stats": {"turns": 4, "tool_calls": 4, "prompt_tokens": 18400, "completion_tokens": 1650,
            "files_read": [{"path": "src/auth/session.py", "start": 41, "end": 67}],
            "stop_reason": "answer"},
  "answer": "..."
}
```
`stop_reason`: answer | max_turns | budget | overflow | parse_error.

### C7 · Grade result
```
def grade(task: Task, trace: Trace) -> GradeResult
def check_citations(answer: str, files_read: list[Span], repo_id: str) -> CitationReport

GradeResult = {
  "reward": 0.72,
  "components": {"format_ok": 1, "citations_parse": 1, "citations_exist": 1, "citations_grounded": 1,
                 "correctness": 0.8, "efficiency": 0.9},
  "gate_failed": null,                              # or "format" | "citations" | "budget"
  "notes": "judge: 4/5 rubric items"
}

reward = 0 if any gate fails
       else correctness * efficiency          # efficiency in [0.5, 1], = 1.0 in run one
```

### C8 · Endpoint profile
`profiles.yaml` at the repo root, loaded by `codeqa.shared.profiles.load_profiles()` / `get_profile(name)`; `make_client(profile)` picks the client by `kind`.
```yaml
claude:                   # teacher / strong product profile
  kind: anthropic         # openai | anthropic | tinker
  model: claude-sonnet-5
haiku:                    # judge, summaries
  kind: anthropic
  model: claude-haiku-4-5-20251001
qwen4b-base:              # untrained policy via Tinker sampling
  kind: tinker
  model: Qwen/Qwen3.5-4B
  renderer: qwen3_5       # cookbook renderer name, thinking on
qwen4b-run1-step40:       # trained checkpoint via Tinker sampling (appended by the trainer)
  kind: tinker
  model: tinker://<run>/sampler_weights/000040
  base_model: Qwen/Qwen3.5-4B   # owns tokenizer + renderer
qwen4b-served:            # day two: Modal vLLM, merged weights
  kind: openai
  model: qwen-codeqa      # served-model-name
  base_url: https://<modal-app>.modal.run/v1
  tool_parser: qwen3_xml  # vLLM flag
  max_context: 65536
  max_generation_tokens: 2048
```

### C9 · Product stream events (SSE)
```json
{"type": "thinking",    "text": "The map shows auth under src/auth..."}
{"type": "tool_call",   "name": "find_symbol", "args": {}, "why": "locate validate_session"}
{"type": "tool_result", "name": "find_symbol", "summary": "1 hit in src/auth/session.py", "chars": 240, "error": false}
{"type": "answer",      "markdown": "..."}
{"type": "citations",   "items": [{"path": "...", "start": 41, "end": 67, "verified": true}]}
{"type": "stats",       "tool_calls": 4, "tool_errors": 0, "prompt_tokens": 18400, "completion_tokens": 1650, "seconds": 9.8, "turns": 5, "stop_reason": "answer"}
{"type": "done"}
{"type": "error", "message": "..."}
```

### C10 · Disk layout and logs
Repo and data layout are defined once in [components.md](components.md). Summary of what lands where:
```
data/repos/<repo_id>/          snapshot + manifest.json
data/index/<repo_id>/          symbols.json  summaries.json  map.txt
data/tasks/raw|train|eval/     JSONL task records (C5)
data/traces/<run>/             one JSON per episode (C6)
data/logs/<run>/               cookbook metrics.jsonl, rollout summaries, HTML viewer
data/evals/<profile>/<set>/    results.json, per_task.jsonl, plots/
data/models/<name>/ + manifest.json   merged weights + CheckpointRecord (gap_specs §2)
profiles.yaml                  endpoint profiles (C8)
```

## Fixtures that unblock everyone

| Fixture | What it is | Unblocks |
|---|---|---|
| `mini_repo/` | 20-file Python package with auth, api, config modules; docstring on every function; one class hierarchy; one config default | B, C, E, H |
| `index/` | Hand-written symbols, summaries, map for the mini repo; becomes B's expected output | C, E |
| `tasks.jsonl` | Five records, one per task type, about the mini repo | D, F, I |
| `traces/` | Four traces: good, padded, confident wrong, fabricated citation | D, H viewer |
