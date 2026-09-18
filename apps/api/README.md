# apps/api

FastAPI backend for the product (lane D2). One process, no database: it reads `data/` through `codeqa.shared.paths`, runs
lane A's environment and driver, and streams C9 events.

## Consumes / produces

| Contract | Use |
|---|---|
| C1 manifests | `GET /repos` lists every `data/repos/*/manifest.json` that has an `index/<repo_id>/map.txt`; `__nodoc` variants are hidden. |
| C3 `RepoEnv.from_question` + `run_episode` | `POST /ask` runs one episode with a cached `ModelClient` per profile. |
| C7 `check_citations` | The `citations` event on `/ask` comes from the grader, not the driver's light check. |
| C8 `profiles.yaml` | `GET /profiles` lists it with a display label and note; `POST /ask` takes a profile name. |
| C9 `SSEEvent` | `/ask` streams `text/event-stream`, one `data: {"type": ..., ...payload}` frame per event, then `done`. |
| C6 traces | Every `/ask` writes `data/traces/product/<task_id>__<profile>.json` (`CODEQA_TRACE_RUN` to change the run). |
| gap_specs §8 | `POST /repos {url, sha?, fast?}` → `{job_id, repo_id}`; `GET /repos/{job_id}/status` → `{repo_id, stage, progress, seconds, message}`. Fast mode (default) marks the repo ready after snapshot + index + map, then fills in summaries and rebuilds the map in the background. |

`GET /file?repo_id&path[&start&end]` returns snapshot text for the file viewer; paths outside the repo are refused.

## Run

```
uv run uvicorn apps.api.server:app --reload --port 8000
cd apps/web && VITE_API_URL=http://localhost:8000 pnpm dev
uv run pytest apps/api/tests            # throwaway repo + scripted client, no network
```

Environment: `CODEQA_WARM_CLIENTS=0` skips creating Tinker sampling clients at startup (they are created on first use
either way); `CODEQA_ASK_TIMEOUT` (default 600 s) bounds one episode.

## Measured (2026-09-18)

- flask, "Where is the Flask application class defined, and what does it inherit from?": `claude` 3 calls, 9.1 s, 29k prompt tokens, 2 of 3 citations verified (the third cites lines it only grepped); `qwen4b-smoke1-step3` 4 calls, 18.4 s, 23k prompt tokens, bracketed citations present, 0 of 2 verified (grepped, never read). Tinker client warm-up 3.9 s.
- `POST /repos https://github.com/pallets/itsdangerous` (38 files): ready in 6.3 s, summaries in 7.8 s; a question on it answered in 13 s with 4 verified citations.
