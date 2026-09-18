# Demo script (lane D4)

Six beats, about six minutes. Everything below was run on 2026-09-18 with the numbers shown; re-run once before the talk
and update them.

## Before the talk

```
uv run uvicorn apps.api.server:app --port 8000          # terminal 1; warms Tinker clients in the background
cd apps/web && VITE_API_URL=http://localhost:8000 pnpm dev   # terminal 2 → http://localhost:5173
```

- Open `http://localhost:5173/app`, pick pallets/flask, ask one throwaway question with the trained profile so the
  sampling client is warm (first call otherwise adds ~4 s).
- Check `GET http://localhost:8000/profiles` lists `qwen4b-base`, the trained checkpoint you want to show, and `claude`.
- Fallback with no network or keys: `pnpm dev` without `VITE_API_URL` replays the recorded episode from
  `apps/web/mock/events.json` on every page. The home page always uses the recording.
- Second fallback: `data/traces/product/` holds every episode the API has run; `evals.report` can read them.

## 1. Home page (30 s)

`http://localhost:5173/`. Let the hero replay finish: the question, the activity rows, "Researched in 3 calls, 9.8 s",
the answer with two verified chips and one flagged. Say: every claim is a link to lines the agent read, and the amber
one is a claim it did not read, caught by the same check the training reward uses.

## 2. Locate, with the teacher (45 s)

Workbench, model **Claude Sonnet 5**, first sample chip:

> Where is the Flask application class defined, and what does it inherit from?

Measured: 3 calls (grep, read 1–100, grep), 9.1 s, 2 of 3 citations verified. Click the green chip → the file pane
opens at `src/flask/app.py:L81` with the range highlighted; click "Open on GitHub". Click the amber chip: it points at
`sansio/app.py:L59`, which the agent only grepped. Say: verified means read, not true.

## 3. Trace, with the trained model (60 s)

Model **Qwen3.5-4B, trained**, second chip:

> Trace what happens when a request raises an exception.

Watch the research log: verbs, one row per call, budget tally. Expand a thinking row. Point at the stats bar: tool
calls and cumulative prompt tokens are the efficiency term.

## 4. Explain (45 s)

Same model, third chip:

> How does Flask decide which session interface to use?

Show the "Also read" list next to "Cited": reads that did not make it into the answer are the waste the reward prices.

## 5. Live repository (60 s)

Left rail, paste `https://github.com/pallets/itsdangerous`, press +. The stepper runs Snapshot → Index → ready.
Measured: ready in 6.3 s, summaries filled in at 7.8 s (map rebuilt in the background). Ask:

> How does the Signer decide which digest method to use?

Measured with Claude: 2 calls, 13 s, 4 verified citations. Say: nothing was precomputed for this repo; the index is
the same one training uses.

## 6. Base vs trained (60 s)

Header → **Compare** (`/compare`). Left `Qwen3.5-4B, untrained`, right the trained checkpoint. Ask the locate question
again. Measured with `qwen4b-smoke1-step3` (a 3-step plumbing checkpoint, not a result): base 3 calls, 21.4k prompt
tokens, 14.1 s, no bracketed citations; trained 1 call, 8.5k tokens, 12.1 s, one bracketed citation, not verified.
Say what the reward did in three steps (format), what it has not done yet (grounding), and what the run-one checkpoint
shows here. Swap in the run-one profile before the talk; the picker lists everything in `profiles.yaml`.

## If something breaks

- API down: the workbench shows the error inline ("The agent stopped before answering") and the log line in terminal 1
  says why. Restart terminal 1; clients rebuild on first use.
- Tinker slow or 402: switch the picker to Claude; the flow is identical.
- Indexing a URL fails: `gh auth token` must work in terminal 1's shell; large repos take longer (2k files ≈ 1–2 min).
