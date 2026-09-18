> Final component list and repo layout: see [components.md](components.md). Component names there supersede the stream letters and numbering below.

# Workstreams and Contracts

Nine streams, ten contracts. Pick a stream, read its card and the contracts it touches, build against the fixtures, stop when the done test passes.

**Hour one, before anyone fans out:** write `codeqa/contracts.py` with every schema below as a pydantic model, and check in `tests/fixtures/`. Every stream imports the models and builds against the fixtures, so no stream waits on another to start. Changing a contract means telling everyone.

## Who depends on whom

```
A Snapshot ──C1──> B Index ──C1+C2──> C Repo Env ──C3──> F Trainer ──C10──> I Eval
                   B Index ──C2──> E Tasks ──C5──> F Trainer
                   C Repo Env ──C3 (teacher)──> E Tasks
D Grader ──C7──> F Trainer,  D Grader ──C7──> I Eval,  D check_citations ──> H Product
C Repo Env ──C3 driver, C9──> H Product
G Serve ──C8──> H Product,  G Serve ──C8──> I Eval
```

No upstream at all: A Snapshot, D Grader (fixtures), G Serve (base model). Everything else starts on fixtures and swaps in the real upstream when it lands.

## The streams

### A · Snapshot — agent
- **Start now with:** the fixed repo list. Training: DeepCodeBench's 8 repos at the commit in each row's metadata, plus the ~15 CodeScout repos with the most rows (excluding the 9 that overlap SWE-QA-Bench), each at its latest `base_commit`. Eval: SWE-QA-Bench's 15 repos at the commits in its `repo_commit.txt`. About 40 snapshots total; few on purpose because everything runs locally.
- **Produces:** C1
- **Done when:** `python -m codeqa.snapshot --repo pallets/flask@<sha>` writes folder + manifest; no vendored/binary files; LF line endings; `pytest tests/test_snapshot.py` passes.

### B · Index — agent
- **Start now with:** `tests/fixtures/mini_repo/`.
- **Consumes:** C1. **Produces:** C2
- **Done when:** symbols for mini repo match expected file; summaries for every directory; `map.txt` < 3000 tokens on a 5000-file repo; whole corpus in under an hour.

### C · Repo Env — you spec, agent fills
- **Start now with:** mini repo + hand-written index; Claude as the model via OpenAI-compatible client.
- **Consumes:** C1, C2, C8. **Produces:** C3, C4, C6
- **Done when:** every tool has tests for happy path, bad path, oversize range, cap; one end-to-end episode on mini repo with Claude yields a valid C6 trace; rendering parity test (cookbook renderer vs vLLM chat template) passes.
- **You decide first:** tool output formats, system rules text, citation regex, caps per task type.

### D · Grader — you spec, agent fills
- **Start now with:** `tests/fixtures/tasks.jsonl` + `tests/fixtures/traces/`.
- **Consumes:** C5, C6. **Produces:** C7
- **Done when:** the four adversarial traces score as intended (padded not rewarded, confident wrong ≈ 0, restated question 0, fabricated citation 0); verifiable tasks grade with no API call; judge prompt checked on five hand-graded answers.
- **You decide first:** gating order, efficiency formula, ground truth per task type.

### E · Tasks — you spec, agent fills
- **Start now with:** the DeepCodeBench importer (912 train rows → C5, `facts` → `rubric`, cited function → line range via C2) and the CodeScout deriver (rows on the chosen repos → Haiku rewrite of `problem_statement` into a where/which question, `file_changes` → `expected_paths` + `expected_symbols`). Both need only public data and C2. Structural generator on the mini repo index. Teacher waits for C.
- **Consumes:** C2, C3, public datasets (`Qodo/deep_code_bench`, `OpenHands/SWE-rebench-code-search`). **Produces:** C5
- **Done when:** every record validates; CodeScout yield is measured (rows whose gold entities resolve in the index at the chosen commit) and reported; 20 sampled records per source read as real questions; no training repo appears in SWE-QA-Bench; base pass-rate filter run; kept set ≥ 1500.
- **You decide first:** the CodeScout rewrite prompt, teacher prompt, filter thresholds, how many CodeScout repos.

### F · Trainer — agent drafts, you review
- **Start now with:** ten fixture tasks + stub grader (citation validity only).
- **Consumes:** C3, C5, C7, C8. **Produces:** C10
- **Done when:** smoke test (10 tasks, group 4, 3 steps) completes; rewards vary within a group; loss finite; metrics JSONL + cookbook HTML rollout page under log path; you have read the mask test and one raw rollout.

### G · Export and serve — agent
- **Start now with:** base Qwen3.5-4B from Hugging Face.
- **Consumes:** a Tinker checkpoint path, later. **Produces:** C8
- **Done when:** vLLM on Modal returns a tool call in Qwen XML with thinking on; after smoke test, merged checkpoint served the same way; cold start recorded.

### H · Product — you design UX, agent builds
- **Start now with:** env driver on mini repo with Claude. Swap to vLLM when G lands.
- **Consumes:** C3 driver, C7 `check_citations`, C8. **Produces:** C9
- **Done when:** typed or dictated question streams a research log, answer with clickable citations opening file at line, sources panel, verified badges, tool calls/tokens/time; model switcher across three profiles.
- **You decide first:** what the answer looks like on screen; what the research log says per step.

### I · Eval — agent
- **Start now with:** fixture tasks against Claude. Then the two held-out sets: DeepCodeBench test (232, in-repo) and SWE-QA-Bench (720, unseen repos; citations extracted by regex from answers). Optional final: 5–10 repos of SWE-QA-Pro-Bench.
- **Consumes:** C3, C5, C7, C8, C10.
- **Produces:** results table per profile × task type × held-out set; plots from metrics JSONL; SWE-QA five-dimension scores via `research/swe_qa_llm_as_a_judge.py`; a fast subset (~120 questions) for the every-10-steps eval.
- **Done when:** one command evaluates any C8 profile on any task file → CSV + plots; base vs trained vs Claude in one table; SWE-QA numbers reported next to DeepRepoQA's and SWE-QA-Pro's with the judge difference stated.

## The contracts (all live in `codeqa/contracts.py`)

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
data/index/<repo_id>/symbols.json
[{"name": "validate_session", "kind": "function",          # function | class | method
  "path": "src/auth/session.py", "start": 41, "end": 67,
  "parent": null, "signature": "def validate_session(token: str) -> Session"}]

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
read_file(path, start, end)                 -> "  41 | code"  numbered lines, cap 150, footer "total 312 lines"

errors    -> "ERROR not_found: src/auth/sesion.py. Did you mean: src/auth/session.py"
budget    -> last line "1 tool call remaining. Answer next turn." when one remains

class RepoEnv:                              # one instance per repo per episode
    def __init__(self, repo_id: str, task: Task, profile: EndpointProfile)
    def tools(self) -> list[Tool]            # cookbook @tool objects, same specs for vLLM/Claude
    def initial_messages(self) -> list[Message]
    def files_read(self) -> list[Span]       # for the grounding check

def run_episode(env: RepoEnv, client: ModelClient, on_event=None) -> Trace   # product + teacher driver
```

### C4 · Prompt, answer format, citation regex
```
system rules (exact text in codeqa/prompts.py):
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
```yaml
qwen4b-base:
  kind: openai            # openai | anthropic | tinker
  model: qwen-codeqa      # served-model-name
  base_url: https://<modal-app>.modal.run/v1
  renderer: qwen3_5       # cookbook renderer name, thinking on
  tool_parser: qwen3_xml  # vLLM flag
  max_context: 65536
  max_generation_tokens: 2048
claude-teacher:
  kind: anthropic
  model: claude-sonnet-5
```

### C9 · Product stream events (SSE)
```json
{"type": "thinking",    "text": "The map shows auth under src/auth..."}
{"type": "tool_call",   "name": "find_symbol", "args": {}, "why": "locate validate_session"}
{"type": "tool_result", "name": "find_symbol", "summary": "1 hit in src/auth/session.py", "chars": 240}
{"type": "answer",      "markdown": "..."}
{"type": "citations",   "items": [{"path": "...", "start": 41, "end": 67, "verified": true}]}
{"type": "stats",       "tool_calls": 4, "prompt_tokens": 18400, "completion_tokens": 1650, "seconds": 9.8}
{"type": "done"}
{"type": "error", "message": "..."}
```

### C10 · Disk layout and logs
```
codeqa/            contracts.py  prompts.py  snapshot.py  index.py  env.py  grader.py
                   tasks/{structural,historical,teacher,filter}.py  train.py  eval.py
                   serve/{export.py,modal_vllm.py}  app/{server.py,static/index.html}
tests/fixtures/    mini_repo/  index/  tasks.jsonl  traces/{good,padded,wrong,fabricated}.json
data/              repos/<repo_id>/  index/<repo_id>/  tasks/{train,heldout}.jsonl
logs/<run>/        metrics.jsonl  rollout_summaries.jsonl  train_iteration_N.html
traces/<run>/      <task_id>-<k>.json
models/<name>/     merged safetensors, uploaded to the Modal volume
profiles.yaml
```

## Fixtures that unblock everyone

| Fixture | What it is | Unblocks |
|---|---|---|
| `mini_repo/` | 20-file Python package with auth, api, config modules; docstring on every function; one class hierarchy; one config default | B, C, E, H |
| `index/` | Hand-written symbols, summaries, map for the mini repo; becomes B's expected output | C, E |
| `tasks.jsonl` | Five records, one per task type, about the mini repo | D, F, I |
| `traces/` | Four traces: good, padded, confident wrong, fabricated citation | D, H viewer |

## Order of operations

1. **Hour one, you.** `contracts.py`, `prompts.py`, fixtures. Decide tool output formats, citation regex, caps, gating order.
2. **Fan out.** A, B, D, G, H in parallel with no waiting. C as soon as tool formats are written.
3. **First integration.** C runs an episode on mini repo with Claude. E's teacher starts. F's smoke test with fixture tasks + stub grader.
4. **Rehearse export.** G serves the smoke-test checkpoint. H switches to it.
5. **Data and run one.** E imports DeepCodeBench, derives CodeScout, generates structural, then filters. Teacher tasks join if ready. F runs overnight, correctness only.
6. **Day two.** Read traces, add efficiency, run two. I produces the comparison. Talk.