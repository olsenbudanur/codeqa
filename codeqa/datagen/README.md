# datagen — Lane B

Turns four task sources into C5 task records under `data/tasks/`, with every path and line range verified against
the pinned snapshot (C1) and index (C2) the agent will actually see.

## Consumes
- **C1 / C2 on disk** via `codeqa.agent.indexing` (`repos.ensure_repo` snapshots + indexes on demand).
- **Public datasets** through `codeqa.clients.hf`, paged and cached once in `data/cache/hf/`.
- **SWE-QA commits** from `repo_commit.txt` in the SWE-QA-Bench GitHub repo (cached in `data/cache/`).
- **Haiku** (`codeqa.clients.anthropic`) for CodeScout question rewrites and structural paraphrases; **Sonnet** for the teacher.

## Produces (C5)
| File | Source | Split | What |
|---|---|---|---|
| `data/tasks/raw/deepcodebench.jsonl` | deepcodebench | train | 912 natural questions, facts as rubric, symbol spans as `required_citations` |
| `data/tasks/eval/deepcodebench_test.jsonl` | deepcodebench | eval | 232 held-out in-repo questions |
| `data/tasks/eval/sweqa.jsonl` | sweqa | eval | 720 unseen-repo questions with resolved citation spans |
| `data/tasks/raw/codescout.jsonl` | codescout | train | locate/trace with programmatic gold (B3) |
| `data/tasks/raw/structural.jsonl` | structural | train | locate/value/enumerate/trace from the index (B4) |
| `data/tasks/raw/teacher.jsonl` | teacher | train | judged tasks authored by Sonnet with our tools (B5) |
| `data/tasks/train/all.jsonl`, `eval/fast.jsonl` | mixed | | after filter + split (B6) |
| `data/tasks/reports/<source>.json` | | | counts and resolution rates per import |

## Modules
- `cache.py` — HF page cache (`fetch_all` resumable, rate-limit aware; `load_cached`).
- `repos.py` — `ensure_repo(owner, repo, sha)` -> (Manifest, symbols); `load_repo(repo_id)`.
- `resolve.py` — `RepoIndex`: exact / unique-basename / unique-suffix / stripped-prefix path resolution, symbol-hint
  disambiguation, span clipping, symbol -> span (`Cls.meth` before `meth`, unique only; class spans clipped to 40 lines).
- `llm.py` — Haiku/Sonnet profiles, `ask`, bounded async runner, JSONL cache under `data/cache/rewrites/`.
- `sources/` — row -> Task per source: `import_deepcodebench`, `import_sweqa`, `derive_codescout` + `rewrite` (Haiku),
  `structural` (tree-sitter facts -> four task types), `teacher` (Sonnet authoring loop over the real tools).
- `importers.py` (B1/B2), `derive.py` (B3), `generate.py` (B4), `teach.py` (B5), `filter.py` + `split.py` (B6), `cli.py`.

## Run
```
uv run python -m codeqa.datagen.cli import --source all               # B1 + B2  (~10 s warm, ~2 min cold)
uv run python -m codeqa.datagen.cli derive --repos 15                 # B3  (3 min; 1,240 Haiku calls, cached)
uv run python -m codeqa.agent.indexing.cli all --repos data/repo_list_codescout.txt --nodoc   # lane A's CLI: summaries, map, nodoc
uv run python -m codeqa.datagen.cli generate                          # B4  (80 s; ~1,000 Haiku paraphrases, cached)
uv run python -m codeqa.datagen.cli teach --seeds-per-repo 30         # B5  (~1 h at concurrency 6; ~$0.11 per kept task; resumable)
uv run python -m codeqa.datagen.cli filter --samples 4                # B6a (Tinker base model x 4 per task; resumable)
uv run python -m codeqa.datagen.cli split                             # B6b (window, per-repo cap, train/ + eval/fast.jsonl + counts)
uv run pytest codeqa/datagen -q
```
Every long step prints progress unbuffered, caches or logs incrementally, and can be re-run to resume.

## Semantics worth knowing
- `required_citations` are the spans of the symbols the reference answer names (DeepCodeBench) or the line ranges
  the annotator cited (SWE-QA). They are *evidence spans*: a correct answer can cite different lines, so the grader
  should score overlap, not exact equality, and never gate on them.
- `expected_paths` only contains paths that exist at the pinned commit. Mentions that could not be resolved are dropped
  and counted in the report.
- `task_type` for imported sources is a regex guess from the question's first words. Judged sources always carry
  `reference_answer` (and `rubric` for DeepCodeBench) so the judge path works regardless of the type guess.
- Structural tasks point at `<repo_id>__nodoc` (docstrings blanked, line numbers identical) so a paraphrased-docstring
  locate question cannot be solved by grepping the docstring. Facts are read from the original snapshot.
- `expected_literal` is the source literal with string quotes removed (`utf-8`, `300`, `-1`, `True`); `None` and empty
  defaults are never used as gold.
- Teacher tasks: `required_citations` are the evidence ranges the teacher named, intersected with what it actually read.
  A task is kept only if blind Haiku, running the real environment, satisfies >= 50% of the rubric.
- Pass-rate filter: `difficulty_score` = correctness among format-valid samples when >= 2 exist, else ungated
  correctness. Window [0.1, 0.9] keeps; below -> `raw/reserve_hard.jsonl`, above -> `raw/reserve_easy.jsonl`.
