# apps/web

The product UI: a home page and a workbench where a user picks a repository, asks a question (typed or dictated), watches the agent research live, and reads a cited answer with verified citations. Vite + React 19 + Tailwind v4 + shadcn (Radix), styled after shadcn-admin but not a fork of it.

## Consumes / produces

| Contract | How the web app uses it |
|---|---|
| C9 `SSEEvent` | `POST /ask` stream, one row of the research log per `thinking` / `tool_call` / `tool_result`; `answer`, `citations`, `stats`, `done`, `error` fill the answer panel. TS mirror in `src/lib/contracts.ts`. |
| C4 `CITATION_RE` | Copied verbatim into `src/lib/contracts.ts`; `tests/test_web_contracts.py` (repo root) fails if it drifts from `codeqa/shared/contracts.py`. It is the only citation parser in the app. |
| C1 manifests | `GET /repos` → repo list (`repo_id`, `url`, `sha`, `files`, `lines`, `symbols`, `stage`). |
| C7 `check_citations` | The `citations` event carries `verified` per item. Until it arrives the UI applies the same grounding rule locally: a citation is verified only if a `read_file` call this episode covers its range. |
| C8 profiles | `GET /profiles` → model switcher (`name`, `kind`, `model`, `label`, `note`). |
| gap_specs §8 | `POST /repos {url, sha?}` → `{job_id}`; `GET /repos/{job_id}/status` → `{repo_id, stage, progress, seconds, message?}` drives the index stepper. |
| file viewer | `GET /file?repo_id&path` → full file text (the viewer scrolls to and highlights the cited range itself). |

Produces nothing on disk. Holds no secrets; the API does. Every route except `/` sits behind a password screen (`components/password-gate.tsx`); the password is kept in localStorage and sent as `Authorization: Bearer` on every API call (`lib/auth.ts`). A 401 from any endpoint clears it and shows the gate again. In mock mode the check is local against the same default (`Action!`).

## Run

```
pnpm install
pnpm dev                         # http://localhost:5173, replays mock/events.json (no backend needed)
VITE_API_URL=http://localhost:8000 pnpm dev   # against apps/api
pnpm test                        # vitest: citation helpers, C9 reducer over the mock stream
pnpm typecheck && pnpm lint && pnpm build
```

Routes: `/` home page (hero replays the recorded episode), `/app` workbench, `/compare` one question to up to four models side by side, first column as baseline (demo beat 6; `?models=a,b,c` prefills), `/workshop/live` one-screen training monitor (auto-picks the run writing metrics, 10 s refresh, tab title shows step and reward, toasts on new warnings), `/workshop/{runs,checkpoints,data,traces}` the read-only workshop (D6; needs `VITE_API_URL`).

## Layout

```
mock/                events.json (C9 stream from data/smoke_episode3.log, one citation deliberately unverified),
                     repos.json, profiles.json, files/<repo_id>/... (the two flask files the sample answer cites)
src/lib/             contracts.ts (TS mirror), citations.ts (one parser), api.ts (HttpApi | MockApi, picked by VITE_API_URL), sse.ts, router.ts, history.ts (saved conversations, localStorage, capped at 100)
src/state/episode.ts C9 events -> research rows, answer, citations, stats
src/hooks/           use-episode (ask/stop), use-dictation (Web Speech API), use-replay (home hero), use-media-query
src/pages/           home.tsx, workbench.tsx, compare.tsx, workshop/ (index shell, runs, checkpoints, data, traces)
src/components/      workshop/ (charts on recharts using the dataviz reference palette: --series-1..4; shared panels), repo/ (repo switcher dropdown, conversations list, index stepper), ask/ (question box + mic), research/ (ledger), answer/ (markdown, chips, sources, stats), file/ (CodeMirror 6 viewer)
tests/               vitest
```

## Home page design

One screen: header, headline with a self-verifying citation chip, one-line summary, the product replaying the recorded episode in a chat frame (`components/chat/transcript.tsx`, fixed height, pinned to the newest turn), four one-line steps, footer. Dark matte field with an emerald atmosphere confined to the hero (from the `tech-green-dark-mode-modern`, `framed-tech-dark-border-gradient` and `atmosphere-background` skills in `.agents/skills/`). Motion: masked word reveal, elements rising in order, brackets closing in, chips stamping in when verified; final states under `prefers-reduced-motion`. Fits a 900px viewport; a `short:` variant compacts it under 800px.

## Decisions

- Conversations are saved per browser in localStorage (`lib/history.ts`) so history works in mock mode and against the API alike; the store is one module so it can be swapped for an API-backed one (the API already keeps every episode as a C6 trace in `data/traces/product/`).

- Vite SPA, not Next.js: the app is a pure client of the FastAPI SSE backend and needs no SSR or server routes.
- Research log rows are verbs ("Read src/flask/app.py, lines 81–85"). Thinking rows are shown one line, muted, expandable.
- Answer panel stacks: answer, then cited / also-read lists, then stats. The `Sources:` block at the end of the answer is parsed for the per-citation notes and not rendered as prose.
- Citation chips show the last two path segments (both flask files are `app.py`).
- Chromatic accent is only the verdict: green verified, amber unverified.
