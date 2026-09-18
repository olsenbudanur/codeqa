# Components (final)

Eight components plus a shared layer. One folder each. Anything hosted lives under `apps/`. All data is JSON under `data/`.

| # | Component | Folder | One sentence | Consumes | Produces |
|---|---|---|---|---|---|
| 0 | Shared | `shared/` | Contracts, prompts, paths, JSONL helpers. Not a component, the thing that makes the others independent. | — | schemas |
| 1 | Clients | `clients/` | Thin wrappers for every external dependency: Tinker, Modal, Anthropic, OpenAI-compatible, GitHub, Hugging Face. One `ModelClient` protocol. | secrets | client objects |
| 2 | Environment | `env/` | Everything about a repo: snapshot, index, repo map, the five tools, the prompt, the episode driver. Identical in training, teacher, product. | repo list, clients | `data/repos`, `data/index`, `RepoEnv`, `run_episode` |
| 3 | Datagen | `datagen/` | Four task sources (DeepCodeBench import, CodeScout derive, structural, teacher) plus dedupe, pass-rate filter, split. | env, grader, clients, public datasets | `data/tasks/train`, `data/tasks/eval` |
| 4 | Grader | `grader/` | Task plus trace in, reward plus components out. Gates, citation checker, verifiers, judge, efficiency. A library with no runtime of its own. | task, trace, clients (judge) | `GradeResult`, `check_citations` |
| 5 | Trainer | `trainer/` | Dataset builder and config that hand env, tasks, and grader to the cookbook loop. Thin. | env, datagen output, grader, clients (Tinker) | checkpoints, `data/logs`, `data/traces` |
| 6 | Serving | `serving/` | Checkpoint to merged weights to Modal volume, and the vLLM app. | Tinker checkpoint, clients (Modal) | an endpoint profile |
| 7 | Evals | `evals/` | Any profile on any task file to a table and plots, including the SWE-QA judge score. | env, grader, profiles, task files | `data/evals/<profile>/<set>/` |
| 8 | Product | `apps/api/`, `apps/web/` | FastAPI backend that runs one episode and streams events; Vite plus React plus shadcn single-page app that renders the research log and the cited answer. | env, grader (`check_citations`), profiles | the demo |

## Why grader and trainer stay separate

They are closely associated, but the grader has three consumers, not one. The trainer calls it for reward. The product calls its citation checker to show verified badges. Evals call it to score held-out sets. If the grader lives inside the trainer, the product and evals import the trainer, which drags Tinker into both. Keep the grader a pure library. The trainer is its first and thinnest consumer.

The same logic keeps the environment out of the trainer: the teacher and the product need it without Tinker.

## Why indexing lives inside the environment

Snapshot, symbol index, summaries, and repo map exist only so the tools and the prompt can use them. Nothing else reads them except the structural generator, which reads the symbols file through the contract. So the environment owns the whole lifecycle of a repo: fetch it, index it, expose it.

## Repo layout

```
action_worktrial/
  pyproject.toml                   # one uv project, one lockfile
  profiles.yaml                    # base | trained | claude endpoint profiles
  .env                             # keys (gitignored); .env.example lists them

  codeqa/                          # the Python package
    shared/                        # 0. imported by everyone, imports nothing
      contracts.py                 #    Task, Trace, GradeResult, EndpointProfile, IndexSymbol, Manifest, SSEEvent, CheckpointRecord
      paths.py  jsonl.py  profiles.py   #    profiles.yaml -> EndpointProfile (get_profile / add_profile)
    clients/                       # 1. every external dependency, thin
      base.py                      #    ModelClient protocol + make_client(profile)
      openai_compat.py  anthropic.py  tinker.py  github.py  hf.py  (modal.py)
    agent/                         # 2. THE CORE: how the task is presented, which tools, how they work
      README.md                    #    answers the spec's three agent questions, maps them to files
      prompts.py                   #    system rules, user prompt (map + question), budget warning
      tools.py                     #    overview, find_symbol, grep, read_file, list_dir  (@tool)
      curation.py                  #    rank, dedupe, collapse, cap
      env.py                       #    RepoEnv: initial_messages, specs/tools, files_read, make_cookbook_env, trace_from_history, from_question
      driver.py                    #    run_episode(env, client, on_event) -> Trace + save_trace  (product + teacher + evals)
      tests/                       #    offline tests on the flask snapshot (tools, indexing)
      indexing/
        snapshot.py                #    repo@sha -> data/repos/<repo_id>/ + manifest.json
        index.py                   #    tree-sitter symbols -> data/index/<repo_id>/symbols.json
        summaries.py  repomap.py  strip_docstrings.py  cli.py   #    cli: snapshot|index|summarize|map|nodoc|all
    datagen/                       # 3. four sources in, filtered tasks out
      sources/                     #    import_deepcodebench.py  derive_codescout.py  import_sweqa.py  structural.py  teacher.py  rewrite.py
      filter.py  split.py
    grader/                        # 4. task + trace -> reward; a pure library
      gates.py  citations.py  verifiers.py  judge.py  efficiency.py  grade.py  README.md (threat model)
    trainer/                       # 5. tasks + env + grader -> cookbook loop
      dataset_builder.py  config.py  run.py
    serving/                       # 6. checkpoint -> merged weights -> Modal volume + manifest
      export.py
    evals/                         # 7. profile x task file -> table, plots, SWE-QA score
      heldout_evaluator.py  run.py  report.py  plots.py

  apps/                            # 8. anything hosted
    inference/                     #    Modal vLLM app
    api/                           #    FastAPI: repos, on-demand index, /ask -> SSE
    web/                           #    Vite + React + shadcn SPA (lane D chose a SPA over a Next.js fork; see apps/web/README.md)
    trainer/                       #    optional detached Modal runner

  scripts/                         # smoke tests: smoke_data, smoke_clients, smoke_chat, smoke_episode, smoke_driver, smoke_train, smoke_grade
  tests/fixtures/                  # mini_repo/  index/  tasks.jsonl  traces/<adversarial>.json
  data/                            # gitignored; repos/ index/ tasks/{raw,train,eval} traces/ logs/ evals/ models/

  docs/                            # ONE docs folder
    README.md                      #    reading order
    components.md  agent_design.md  contracts.md  data_sources.md  gap_specs.md  decisions.md
    research/                      #    evidence: tool survey, dataset shapes, census, paper notes
    archive/                       #    superseded drafts, do not read
```

Import paths are `codeqa.<component>.<module>`, e.g. `from codeqa.shared.contracts import Task`, `from codeqa.agent.indexing.index import build_index`. Scripts run from the repo root: `uv run python -m scripts.smoke_data`.

## Import rules

- Everyone may import `shared` and `clients`.
- `env` and `grader` import nothing else. They are the core.
- `datagen`, `trainer`, `evals`, `apps/api` import the core. They never import each other.
- `serving` and `apps/inference` import only `shared` and `clients`.
- No component reads another's files directly. It goes through `shared/paths.py` and `shared/contracts.py`.

## Build order

1. `shared/contracts.py`, `shared/prompts.py`, `tests/fixtures/`. One hour, you.
2. In parallel: `clients`, `agent/indexing`, `grader` on fixtures, `serving` plus `apps/inference` with base Qwen, `apps/web` shell against Claude.
3. `env` tools and driver, run one episode on the mini repo with Claude.
4. `trainer` smoke test on fixture tasks with a stub grader, then rehearse export.
5. `datagen` import, derive, generate, filter.
6. Run one overnight. Day two: traces, efficiency term, run two, `evals`, talk.

## Mapping to the workstreams page

Streams A and B (snapshot, index) are `agent/indexing`. Stream C is the rest of `env`. D is `grader`. E is `datagen`. F is `trainer`. G is `serving` plus `apps/inference`. H is `apps/api` plus `apps/web`. I is `evals`. `clients` and `shared` are the hour-one work that the page calls contracts.
