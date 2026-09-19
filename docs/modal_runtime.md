# Modal runtime

What moved to Modal on 2026-09-19 (decision #8 in `decisions.md`), how to use it, and what to watch for.

## What runs where

| piece | runs on | why |
|---|---|---|
| sampling and LoRA training | Tinker | unchanged; throughput is Tinker's, not ours |
| training loop, evals, pass-rate filter, teacher (the "agent runtime jobs") | Modal, detached functions | laptop no longer in the loop; parallel jobs on cheap CPU containers |
| `bash` tool (variant) | Modal sandboxes, one pool per job | real containment for a policy that writes shell commands |
| the five default tools, grader | inside whichever job runs them (Modal container or laptop) | file reads and index lookups; nothing to isolate |
| judge (Haiku) | Anthropic | unchanged |
| product API + web | EC2 (lane E) | serving, not compute |

Data: the Modal volume **`codeqa-data`** is the source of truth for jobs. Layout mirrors `data/` and the repo root:

```
/repos/<repo_id>/         snapshots (+ __nodoc variants)      pushed up
/index/<repo_id>/         symbols, summaries, map              pushed up
/tasks/{raw,train,eval}/  task files                           pushed up
/profiles.yaml            endpoint profiles                    pushed up; jobs append checkpoint profiles here
/logs/<run>/              cookbook metrics, checkpoints, rollouts   written by jobs, pulled down
/evals/<profile>/<set>/   results, per-task rows, traces       written by jobs, pulled down
/traces/, /models/        episode traces, checkpoint manifest  written by jobs, pulled down
```

Inside a job: `CODEQA_DATA_DIR=/data` and `CODEQA_PROFILES=/data/profiles.yaml` (set by the image), so every module that goes through `codeqa.shared.paths` reads and writes the volume without knowing.

## Commands

```
scripts/modal_push.sh                 # index + tasks + profiles to the volume (after adding tasks / repos)
scripts/modal_push.sh repo <repo_id>  # one new snapshot + index (after `indexing.cli all` on the laptop)
scripts/modal_push.sh repos           # every snapshot (~1 GB the first time)

modal run --detach apps/trainer/modal_runner.py --module codeqa.trainer.run \
  --args "--tasks data/tasks/train/run1.jsonl --profile qwen4b-base --run-name run1 --steps 50 \
          --group-size 8 --groups-per-batch 16 --lr 1e-4 --eval-tasks data/tasks/eval/fast.jsonl --eval-every 10 --save-every 10"
modal run --detach apps/trainer/modal_runner.py --module codeqa.evals.run \
  --args "--profile qwen4b-base --tasks data/tasks/eval/fast.jsonl --set fast --concurrency 8"
modal run --detach apps/trainer/modal_runner.py --module codeqa.datagen.cli --args "filter --samples 2 --sources structural"
modal run --detach apps/trainer/modal_runner.py --module codeqa.trainer.run \
  --env CODEQA_AGENT_VARIANT=bash,CODEQA_BASH_EXECUTOR=modal --args "..."      # bash agent, sandboxed

modal app logs codeqa-jobs            # logs of detached jobs
scripts/modal_sync.sh                 # pull /logs /evals /traces /models down into data/ (laptop or EC2)
```

`--args` takes exactly the arguments you would pass locally; `data/...` paths are rewritten to `/data/...`. The runner spawns the job and returns in seconds with its call id; add `--wait` to block and stream (smokes). With `CODEQA_RUNTIME=modal` in `.env` (the default now) you do not type any of this: the plain `uv run python -m codeqa.<cli> …` command redirects itself; `CODEQA_RUNTIME=local` in front runs it on the laptop, `CODEQA_RUNTIME_WAIT=1` blocks.

## Adding things

- **A repo:** on the laptop, `uv run python -m codeqa.agent.indexing.cli all --repos <file>` (needs GitHub + Haiku), then `scripts/modal_push.sh repo <repo_id>`. Or run the indexing CLI as a Modal job (writes to the volume directly; needs `GITHUB_TOKEN` in the `codeqa` secret for tarballs at scale).
- **Tasks or eval sets:** write the JSONL under `data/tasks/...`, `scripts/modal_push.sh`.
- **A profile:** edit `profiles.yaml`, push. Checkpoint profiles that jobs append on the volume come back as `data/profiles.modal.yaml` on sync; merge them into the repo's `profiles.yaml` by hand.
- **Code:** nothing to push. The runner mounts `codeqa/`, `apps/`, `scripts/` from the laptop at job start. Dependency changes (pyproject/uv.lock) rebuild the image, about a minute.

## How the pieces fit

- `apps/trainer/modal_runner.py`: app `codeqa-jobs`; image = debian_slim + ripgrep + `uv_sync` of the lockfile + local source; one function `job(module, argv, env)` that runs `python -m module argv`, commits the volume every 30 s and at exit. 4 CPU, 8 GB, 24 h timeout.
- `codeqa/agent/modal_shell.py`: the bash executor; app `codeqa-sandbox`; pool of `CODEQA_MODAL_SANDBOXES` (default 4) sandboxes with the volume read at `/data`, network blocked, 1 h lifetime, 10 min idle timeout. One sandbox runs many commands concurrently. The pre-check and seen-lines extraction in `codeqa/agent/shell.py` are shared with the local executor, so grounding is identical.
- `codeqa/agent/variants.py`: `default | noindex | nomap | bash | bash_nomap`, chosen per `RepoEnv(variant=)` or `CODEQA_AGENT_VARIANT`.

## Measured

- Eval job, 3 flask tasks, five tools: 39 s inside the container; image build 32 s once.
- Training job, 10 graphiti tasks × 4, one step, five tools: 100 s incl. checkpoint save.
- Bash sandbox pool: warm-up 12 s, 0.7 s per command, 32 concurrent commands in 4.5 s.
- Bash-agent eval job inside Modal (container spawns its own sandboxes): 3 flask tasks in 38 s.

## Gotchas

- **The volume is not a live shared filesystem.** A job's writes are visible to others only after its commit (every 30 s); the laptop sees them only after `scripts/modal_sync.sh`.
- **`modal volume get … --force` replaces a local directory that also exists on the volume** (it deleted local eval sets on 2026-09-19). The sync script therefore stages under `data/.modal_pull/` and merges with rsync without deletes, and merges the models manifest by record name. Never call `modal volume get` directly into `data/`.
- `modal volume put` of the whole `data/repos` (~1 GB, 65 dirs) failed with `stream timeout`; push one snapshot per call (`scripts/modal_push.sh repos` does). `/index` (150 MB) uploads fine in one call.
- Two jobs writing the same run name clobber each other's files. Use distinct `--run-name` / `--set`.
- A task file smaller than `steps × groups_per_batch` gives one batch per epoch; pass `--epochs` (lane C's trainer).
- The runner does not tear down the sandbox pool at job end; Modal reaps it on idle (10 min) or lifetime (1 h).
- The trainer does not yet append the final checkpoint to `profiles.yaml` / the manifest (asked of lane C, LOG 01:45); until then add the `qwen4b-<run>-step<N>` profile by hand from `checkpoints.jsonl`.
- **Default is Modal since 2026-09-19 02:40**: `.env` has `CODEQA_RUNTIME=modal`, so `uv run python -m codeqa.trainer.run …` (and `evals.run`, `datagen.cli`) launches the same command as a detached Modal job and prints the job id. `CODEQA_RUNTIME=local` in front runs it on the laptop. Inside Modal and under pytest the hook is a no-op. The hook is the first line of each CLI's `main()` (`codeqa.shared.runtime.maybe_redirect_to_modal`).
