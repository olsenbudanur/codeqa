# Reward red-team (independent reviewer agent, 2026-09-19)

Scope: `codeqa/grader/*`, `codeqa/trainer/group_rewards.py`, `codeqa/trainer/dataset_builder.py::compute_group_rewards`, prompts, tools, gap_specs §3/4/6/10, decisions, lane C log entries. VERIFIED = a crafted trace was run through the real grader (KeywordJudge offline) or the code path was read definitively. Scripts with raw output were in the reviewer's scratchpad (`redteam/{exploits,exploits2,live_judge,prefix_tokens,data_stats,scan_lines}.py`).

Training-file facts (1,489 tasks): 559 judged (all rubric mode, none with `expected_symbols`); 334 single-gold-symbol locate (any-of); 306 multi-symbol callee tasks (symbol F1); 161 path-only tasks; 12 value literals (`src/diffusers, 3, False, 0, 0., 4, {{ name }}, False, ., False, 0, keras`); 4,073 rubric items, 7 % location-type; 0 tasks without gold. Cookbook: advantages are group-centered only (no std normalisation); the env adds its own −0.1 for parse errors and context overflow; each transition's observation is the entire rendered conversation so far.

## Part A — ways the reward is bad

| # | claim | mechanism | sev | status |
|---|---|---|---|---|
| A1 | Grounding is all-or-nothing and grep spans are one line, so the case decision #7 meant to fix still scores 0 | `tools.py:195` records `Span(ln, ln)`; `citations.py:62` needs every cited line read; `grade.py:47-49` zeroes the whole answer on any failing citation | HIGH | VERIFIED: cite L35-L36 after grep hit L35 → 0 (grounding); cite L35 → 1.0; 5 good + 1 off-by-one → 0 |
| A2 | Symbol credit needs the `def` line inside a cited range; citing the body or the call site scores 0 | `verifiers.py:51-59` → `repo.py:82-85` (`start <= s.start <= end`); naming the caller enters pred and costs precision | HIGH (306 callee tasks) | VERIFIED: body L36-L42 (def L35) → 0; L35-L42 → 1.0; names both callees, cites caller body → 0; opened both callee defs → 0.8 |
| A3 | Content-free answers score 1.0 on 161 path-only tasks; shotgun name lists score 1.0 on 334 single-gold locate tasks | `verifiers.py:43` never inspects text; `verifiers.py:76-77` any-of has no precision term | MED | VERIFIED: "No idea. [routes.py:L1]" → 1.0; name every symbol + cite L1-L48 → 1.0; 2-gold enumerate with every name → 0.44 |
| A4 | Run-two efficiency budgets mis-scaled 3-4×: `multiplicative`/`token_cost` floor at 0.5, `hard_cap` zeroes nearly everything | `dataset_builder.py:52-55` sums `ob.length` = cumulative context; budget = calls × 3000 (`efficiency.py:20,38`); prefix alone 1.7k-4.2k tokens (median 3.7k / 46 snapshots); smoke_lr1e4 explain: 37.7k-49.6k tokens at 5-6.7 calls vs 36k budget | HIGH for run two, none for run one | VERIFIED from logs |
| A5 | Under `multiplicative` a grep after a read counts every hit as a redundant call | `efficiency.py:25-35`: one-line grep span inside a read range is "fully covered"; 30 hits = +30 calls | MED (run two) | VERIFIED: read + grep(30) → eff 0.5 |
| A6 | 8/12 value literals are trivial tokens; `.` can never match | `verifiers.py:24` rstrips `.;,` → "." normalises to empty; `False`/`0` match anywhere as whole tokens | LOW | VERIFIED: literal "." → 0; "defaults to True, not False" with gold False → 1.0 |
| A7 | Grounded credit is earned by any read line, content irrelevant | `group_rewards.py:21-27`; bounded at 0.05 by design; intended ladder | LOW | VERIFIED: grep-hit line cited, no content → shaped 0.05 |
| A8 | CodeScout multi-gold F1 caps a perfect answer near 0.5 | 264/444 have 2-8 gold symbols from the fix PR, some unrelated; with A2, 1.0 needs citing defs of entities the question never mentions | MED | PLAUSIBLE |
| A9 | Haiku judge is robust; residuals small | live probe (3-item rubric): good 1.0; hedge listing all alternatives 0 (×3); injection 0; fake "CONFIRMED" echo 0.33; vague 0; contradiction-then-correct 1.0. `judge.py:132` `bool(it.get("satisfied"))` makes the string "false" satisfied; judge temperature not settable | LOW | VERIFIED (parse) / live |
| A10 | Signal structure: ladder is weak relative to a correct answer (right order); NaN fill correct; a credit outage silently makes ~37 % of groups constant | typical step-0 group advantages [-0.056×4, 0.044×3, 0.094]; with one correct [-0.181×4, -0.081×2, -0.031, 0.919]; NaN advantage 0.0 | LOW-MED | VERIFIED |
| A11 | Verifiable 0/1 vs judged fractions: judged groups differ by one item (±0.08) vs ±0.9, so judged tasks (37 %, the only ones with 33 % step-0 group variance) give ~1/10 the gradient per signal-bearing group | | MED | PLAUSIBLE |
| A12 | Misc | `identifier_grounded` metric-only (no judged train task has symbols); unclosed `<think>` at answer start drops the answer; wrong-case path passes `exists` on macOS but fails grounding; 25/66,028 snapshot files (binary .fits) have splitlines/`\n` disagreement; one django path with spaces is uncitable; judge ~100 calls/step (~$0.30), worst-case sample latency ~141 s behind semaphore 16; format gate binary at cap (801 tokens = 0); reference mode unused | LOW | VERIFIED |

## Part B — improvements, ranked

(a) Safe before run one (each < 1 h, offline-testable):
- **B1** One-line tolerance on grounding coverage (`citations.py:31`: `range(start-1, end+2)`). Fixes A1. Test: A1/A3 → 1.0; unread/fabricated/no_tools fixtures still 0.
- **B2** Symbol match on body overlap (`repo.py:85`: `start <= s.end and end >= s.start`) and drop symbols named in the question from pred. Fixes A2 M1 and N2 precision. Whole-file cites already covered every def line, so no new exposure.
- **B3** Callee tasks accept a grounded call-site citation: gold symbol predicted when named AND some grounded cited line's text contains the name. Fixes A2 N1 (306 tasks).
- **B4** Precision guard on single-gold any-of: path 1.0 × min(1, 2/|cited paths|); symbol 1.0 × min(1, 3/|pred|). Fixes A3 D1 (12 names → 0.25).
- **B5** Path-only tasks must name the gold path or basename outside a bracket, else × 0.5. Fixes A3 E1.
- **B6** Value literals: drop or re-gold the 8 trivial ones, or require a citation intersecting `required_citations`. Fixes A6.
- **B7** Judge boolean parse: `satisfied is True or str(...).lower() == "true"`. Fixes A9.

(b) Run two:
- **B8** Fix token accounting before any efficiency variant: prompt tokens = final context (`transitions[-1].ob.length + len(ac.tokens)`, typically 4k-15k) or budget = prefix_tokens + 2500 × max_tool_calls with prefix measured per repo. Target median usage 0.5-0.7 on the smoke traces (now 1.05-1.4). Fixes A4.
- **B9** Exclude one-line grep spans from `redundant_reads`. Fixes A5.
- **B10** Rescale judged correctness to score² so within-group gaps widen at the top where Haiku is reliable. Risk: fewer partial-credit gradients on hard explain tasks. (Lane C: not blind; measure first.)
- **B11** CodeScout multi-gold: min(1, |pred ∩ gold| / min(|gold|, 2)) × min(1, 4/|pred|). Fixes A8's ceiling.
- **B12** Grounded fraction instead of all-or-nothing (gate only when < 0.5; keep `citations_exist` strict). (Lane C: only after B15 data.)

(c) Research: **B13** rubric items from the gold file's docstring for the 161 path-only tasks; **B14** re-judge only groups whose judged scores span ≥ 2 items; **B15** log `grounded_off_by_one` per rollout so B12 is decided on data.

Bottom line: the judge is sound; the gates are where correct answers die (three verified false-zero classes: off-by-one citations of grep-shown lines, body-not-def citations, call-site citations on callee tasks). B1-B3 turn those into signal. Do not enable any efficiency variant before B8.
