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

Produces nothing on disk. Holds no secrets; the API does.

## Run

```
pnpm install
pnpm dev                         # http://localhost:5173, replays mock/events.json (no backend needed)
VITE_API_URL=http://localhost:8000 pnpm dev   # against apps/api
pnpm test                        # vitest: citation helpers, C9 reducer over the mock stream
pnpm typecheck && pnpm lint && pnpm build
```

Routes: `/` home page (hero replays the recorded episode), `/app` workbench, `/compare` one question on two profiles side by side (demo beat 6).

## Layout

```
mock/                events.json (C9 stream from data/smoke_episode3.log, one citation deliberately unverified),
                     repos.json, profiles.json, files/<repo_id>/... (the two flask files the sample answer cites)
src/lib/             contracts.ts (TS mirror), citations.ts (one parser), api.ts (HttpApi | MockApi, picked by VITE_API_URL), sse.ts, router.ts, history.ts (saved conversations, localStorage, capped at 100)
src/state/episode.ts C9 events -> research rows, answer, citations, stats
src/hooks/           use-episode (ask/stop), use-dictation (Web Speech API), use-replay (home hero), use-media-query
src/pages/           home.tsx, workbench.tsx
src/components/      repo/ (repo switcher dropdown, conversations list, index stepper), ask/ (question box + mic), research/ (ledger), answer/ (markdown, chips, sources, stats), file/ (CodeMirror 6 viewer)
tests/               vitest
```

## Home page design

Direction from four installed skills (`.agents/skills/`, symlinked into `.claude/skills/`): `tech-green-dark-mode-modern` and `framed-tech-dark-border-gradient` (matte dark field, emerald signal, 1px gradient-border frames, corner brackets, mono rails), `atmosphere-background` (slow drifting light folds and a bloom, CSS only, confined to the hero), `product-proof-saas` (the real product as the hero proof, labeled sample, no fake proof) and `landing-page-design` (above-the-fold formula, section order). Plus `light-mode-paper-technical` and `beautiful-shadows`, installed for an earlier paper-toned pass and kept for reuse.

Concept: the page behaves like a cited answer. The headline ends in a live citation chip that verifies itself; the hero frame is a chat transcript (`components/chat/transcript.tsx`) replaying the recorded episode: the question as the user's turn, activity rows streaming into the assistant turn, collapsing to "Researched in 3 calls, 9.8 s" when the answer lands. The transcript viewport is fixed-height and pinned to the newest content, so the hero never resizes. "How it works" uses the product's own components stepping through the episode; the footer is a `Sources:` block naming where every number on the page comes from. Motion: masked word reveal, elements rising in sequence, brackets closing in, chips stamping in when verified, section rules drawing in. Everything renders its final state under `prefers-reduced-motion`. The hero is sized to fit a 900px viewport with the chat frame fully visible.

## Decisions

- Conversations are saved per browser in localStorage (`lib/history.ts`) so history works in mock mode and against the API alike; the store is one module so it can be swapped for an API-backed one (the API already keeps every episode as a C6 trace in `data/traces/product/`).

- Vite SPA, not Next.js: the app is a pure client of the FastAPI SSE backend and needs no SSR or server routes.
- Research log rows are verbs ("Read src/flask/app.py, lines 81–85"). Thinking rows are shown one line, muted, expandable.
- Answer panel stacks: answer, then cited / also-read lists, then stats. The `Sources:` block at the end of the answer is parsed for the per-citation notes and not rendered as prose.
- Citation chips show the last two path segments (both flask files are `app.py`).
- Chromatic accent is only the verdict: green verified, amber unverified.
