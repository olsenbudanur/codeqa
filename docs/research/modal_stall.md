# Sampling stall (2026-09-18): Tinker LoRA sampling hangs for this account; the client waits silently

Scope: SDK `tinker` 0.29.1, `tinker-cookbook` 0.5.7, Modal logs, free Tinker metadata only (no sampling).

## What we know

- Not Modal, not billing, not the transport. Sampling from LoRA samplers hangs from Modal jobs, from four
  laptop `scripts.arm` runs and from the product API; base-model one-token probes return (slowly, ~20 s).
  The billing pause logs "The job is paused due to billing status" every 60 s and there are none. The
  Tinker session heartbeat and Anthropic calls kept working during every stall (Modal log
  ap-zy2reWq00Fpi0ebV0mRN6e: batches 0-4 at ~2 min each, "Sampling batch 5" frozen at 23/32 from 19:54:45
  until stopped at 20:10:07, zero warnings, process still responsive).
- Silence is by design. Server flag `sample_use_retrieve_futures=true` (free `/api/v1/client/config`):
  each sampler has one long-poll for completion ids and a sample future just waits
  (`lib/session_futures_poller.py`). Queue states ("paused_rate_limit" = "concurrent sampler weights limit
  hit", "paused_capacity") are only surfaced on the legacy path, never here. A 429 on `/asample`
  (backpressure header) makes the SDK back off 1-5 s and resubmit forever without logging
  (`sampling_client._sample_async_impl`). The only client bound is `RetryConfig.progress_timeout` = 2 h.
  Cookbook default for non-Inkling models is `FailFast` with no per-rollout timeout, so one unserved
  request freezes a batch. `TINKER_LOG=debug` prints every request and status ('HTTP Response: POST
  .../asample "429 ..."'), which would show submit-side backpressure vs. never-completing results.
- Every job creates LoRA samplers: `save_checkpoint_and_get_sampling_client` makes a fresh sampler each
  step (and at step 0, even before any training), the held-out evaluator and the product profiles
  (`sampler_weights/final`, `sampler_weights/000010`) sample from LoRA samplers too. Free census of the
  account (GET /sessions, /sessions/{id}): 160 sessions today, 234 samplers, 40 training runs; after the
  cleanup 153 are `client_finished`, 4 `cancelled` (server-side status, detail null), 2 `active` (probes).
  Public cookbook guidance: Tinker "limits how many distinct sampler weights can be hot at once, per
  account"; no number is published, and the docs say nothing about per-project scope.
- Orphaned work keeps running server side. `retrieve_futures` (metadata) on finished sessions' samplers
  shows completed requests (cursors up to 895), and the newest active probe sampler went from 0 to 1
  completion between two probes ~10 min apart, so hung LoRA requests do eventually complete. Recovery by
  waiting is therefore bounded by the backlog draining, not infinite, but the drain rate for LoRA samplers
  is currently minutes per request.

## (a) How killed clients leave state on the server (SDK facts)

- Session: `POST /create_session` at `ServiceClient` construction; finished only by `ServiceClient.close()`
  (`POST /sessions/{id}/finish`, "Marks the session terminal"). `InternalClientHolder.__del__` cleans local
  resources only. The cookbook's `train.main` never calls close (rl/train.py:1935), so every SIGKILL,
  OOM, preemption or `modal app stop` left an `active` session until our hygiene wrapper/cleanup.
- Sampler: `POST /create_sampling_session` or `save_weights_for_sampler`; ids look like
  `<session>:sample:<n>`; there is no delete or unload endpoint.
- Requests: keyed by (sampling_session_id, seq_id); an explicit `POST /cancel_future {request_id}` exists
  but the SDK only sends it when dynamic flag `sample_cancel_enabled` is true, and `/client/dynamic_config`
  returns `{}` for this account (default false). SDK comments say abandoned work is otherwise stopped by
  "the server's SDK-heartbeat fallback" (heartbeat every 10 s; client warns after 2 min; server expiry
  undocumented). The 129 orphans stayed `active` for hours, so that fallback is slow or absent.
- Per-process client caps: 400 in-flight dispatches per holder, 2000 per sampler
  (`sample_max_concurrent_requests`), 128-256 in flight per job here.

## (b) Docs: limits, queueing, status

tinker-docs (data-model, session-metrics, async-patterns, under-the-hood, restclient) publish no per-account
in-flight or hot-sampler numbers and no queueing semantics; the console (tinker.thinkingmachines.ai/sessions/
<id>) shows "In-flight sample requests" and "Rate-limit events/sec" per session. No public status page found.

## (c) Cancel / terminate APIs and recovery

- Finish sessions: `ServiceClient.close(status, detail)` or `ServiceClient(session_id=...)` then close
  (done: 129 marked "cleanup: orphaned session"). Move: `RestClient.assign_session_project(session_id,
  project_id)` (one way, moves runs and samplers). Delete checkpoints: `delete_checkpoint_from_tinker_path`.
  Cancel one request: `POST /api/v1/cancel_future` with a request_id (only the dead processes had them).
  A server-side `cancelled` status exists (4 sessions), likely from the console; try cancelling remaining
  `active` sessions there. Nothing unloads sampler weights.
- Cross-project: checkpoints and samplers inherit the session's project; use
  `ServiceClient(project_id=NEW).copy_weights("tinker://.../sampler_weights/000010")` to reuse old weights.
  If the hot-sampler cap is per account, a new project will not help; the probe below tells in a minute.
- Recovery if we wait: unknown upper bound; evidence says requests complete at minutes-per-request pace,
  so with ~600 orphaned in-flight requests (4 kills x 128-256) expect hours, and it worsens with every
  killed job. Ask Tinker support (session ids in scratchpad census) to drop the org's queued sampling work.

## Fixes (minimal)

1. Keep the session hygiene in `codeqa/clients/tinker.py` and the 20-min watchdog in `scripts/arm.py`; add
   `TINKER_LOG=debug` for the next diagnostic run so submit-side 429s become visible.
2. `codeqa/trainer/config.py` `build_config`: `rollout_error_tolerance=RetryOnFailure(max_retries=8,
   per_rollout_timeout=600)` so a hung rollout is cancelled, logged and dropped instead of freezing.
3. Never kill a job with in-flight sampling without close(): stop via SIGINT/SIGTERM (hygiene handler),
   never `modal app stop` while sampling; run one job at a time until the cap is understood.
4. Canary before any launch (one token, under a cent): LoRA sampler `sample_async` in `wait_for(60)`.
