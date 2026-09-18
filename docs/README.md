# Docs index (read in this order)

1. `components.md` — the final component list, repo layout, import rules, build order. Start here.
2. `agent_design.md` — how the agent works: index, prompt, five tools, loop, rules, budgets, precedents. Code lives in `codeqa/agent/` (see its README).
3. `contracts.md` — the ten seams (C1–C10): task record, trace, grade result, tool API, profiles, SSE events. Code: `codeqa/shared/contracts.py`.
4. `data_sources.md` — which datasets feed training and eval, counts, licenses, how each becomes a task.
5. `gap_specs.md` — specs for the deliverable-level items (checkpoint manifest, threat model, judge failure, in-loop eval, shaping variants, metrics, on-demand indexing) and the parallel-vs-sync split.
6. `decisions.md` — dated decision log and the talk outline. Append as you go.

`agents/` is the onboarding page, the append-only agent log, and one brief per lane under `agents/lanes/`. Agents start there.
`research/` holds evidence (tool survey, dataset shapes, census, paper notes, the SWE-QA judge script), not specs.
`archive/` holds superseded drafts. Ignore it.
