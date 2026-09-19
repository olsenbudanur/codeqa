# Decisions log

One line per decision, dated, with what changed our mind. This is the raw material for the talk's "how my thinking evolved" section. Append as you go.

## 2026-09-18 (day 0, ideation)

- **Read-only tools, no sandbox, no Docker.** Model never runs code. A repo is a folder, a tool is a function. Removes the hardest infra piece.
- **Harness = tinker-cookbook `tool_use` + `search_tool` recipe.** Found after checking whether Tinker/Modal publish one. Replaces a planned custom harness.
- **Thinking ON.** User wants the product to be good with a small model. Efficiency term counts tool calls and context tokens, not thinking tokens, in early runs.
- **Pre-index every repo** (tree-sitter symbols + directory summaries + token-capped map). Prompted by DeepRepoQA's index-backed actions and its ablation showing semantic search matters least. `overview` tool replaces embeddings.
- **Five tools:** overview, find_symbol (kind, file_pattern), grep, read_file (line range), list_dir. Tool outputs curated (rank, dedupe, collapse, cap), always line-numbered.
- **Reward = outcome, computed from the whole trajectory.** Gates (format, citations exist and were read) → correctness (verifier or rubric judge) → efficiency multiplier. Run one correctness-only; run two adds efficiency. Precedents (RepoSearch-R1, SWE-QA-Pro) use judge-only rewards with no grounding or efficiency; that is our gap to fill.
- **Q&A only, no issue-to-patch tasks.** Issue-derived data is used only as a source of realistic *questions* with location gold (CodeScout), never as patch tasks.
- **CoReQA dropped**: never released, no commits. Found by the research agent.
- **Data sources fixed after a census** (2,705 verified natural repo Q&A with pinned commits exist publicly; 5k is not reachable from natural data). Train: DeepCodeBench train (912, facts=rubric), CodeScout-derived locate tasks (~15 repos), structural generator, Claude teacher (answer-first, blind verify). Eval: DeepCodeBench test (in-repo), SWE-QA-Bench (unseen repos), optional SWE-QA-Pro-Bench / Code-QA-Bench.
- **Few repos on purpose** (~23 train + 15 eval snapshots). Everything runs locally. SWE-QA-Pro's 464-repo RL set skipped for this reason.
- **Serve merged weights, not LoRA adapters.** vLLM LoRA on Qwen3.5 is buggy (research agent).
- **Traces on disk**; the cookbook's HTML rollout viewer is the viewer. WandB optional, one config line.
- **Runtimes local**; the only containers are Modal images for vLLM. Overnight run protected with caffeinate + tmux; Modal detached driver is the fallback.
- **Repo layout**: one folder per component, `shared/` contracts, `clients/` for external deps, `apps/` for hosted things, `data/` all JSON. Grader kept separate from trainer because product and evals also consume it.
- **Product UI**: shadcn dashboard fork (Next.js) over a FastAPI SSE backend.
- **Grader refinements from RepoSearch-R1 review:** symbol-set F1 for multi-target tasks, identifier grounding via the index, per-group diversity metrics, docstring-stripped snapshot variant for structural tasks. MCTS rollouts deferred behind a diversity trigger. See gap_specs.md §10.

## Open questions carried into day 1

- CodeScout license (HF field unset) — confirm from the OpenHands repo or switch to Long Code Arena bug localization (Apache-2.0, 7,479 rows).
- CodeScout yield under the one-commit-per-repo policy — measure, do not assume.
- Whether the base 4B can emit the citation format at all before training — decides the SFT warm-start.
- Judge-failure policy in the grader (retry, then exclude the sample from its group; never zero, never group-average).

## Talk outline (fill as you go)

1. What I built (one slide: the three-lane diagram).
2. How I sequenced two days and why (contracts first, fan out, smoke test before data).
3. What "good" means here and how the grader defends it (gates, grounding, efficiency; the four adversarial traces).
4. Where the data came from (census result; why generation; the few-repos constraint).
5. Reward curve + held-out curves + citation validity + tool calls per correct answer.
6. Before/after rollouts on the same question. Live demo: base vs trained vs Claude.
7. What I learned / would do differently / how thinking evolved (this log).
8. What's next (LSP tool, SFT warm-start ablation, run three with a different shaping, more repos on Modal).

## 2026-09-18 (day 0, sync setup outcomes)

- **Proven sync:** contracts validated against real rows (DeepCodeBench, CodeScout, SWE-QA); flask snapshot at the SWE-QA commit + tree-sitter index in ~3s; Tinker capabilities/renderer/sampling OK; Modal profile + `codeqa` secret + `codeqa-data`/`codeqa-models` volumes; GitHub + HF OK. One full cookbook episode ran with real tools over the snapshot (`scripts/smoke_episode.py`).
- **Step-0 behavior observed:** untrained Qwen3.5-4B answered without any tool call, copied the prompt's placeholder template verbatim, fabricated `[path:L9-L18]`; grounding gate → reward 0. Matches RepoSearch-R1 / SWE-QA-Pro's "small models get worse with tools until trained." Prompt changed to a concrete example + explicit "call a tool before answering".
- **Gotchas recorded:** `.env` with empty `MODAL_TOKEN_*` lines overrides the Modal profile (removed); anthropic SDK ≥1.7 rejects `temperature`; Tinker SDK retries 402 for minutes (client now `max_retries=1`); cookbook `@tool` resolves type hints at decoration time, so tool signature names must be module-level under `from __future__ import annotations`; macOS has no `timeout` binary.
- **Cookbook facts:** `EnvGroupBuilder.compute_group_rewards` is the hook for the NaN→group-mean judge-failure policy; `train.Config.remove_constant_reward_groups` is the built-in zero-variance filter; `RewardFn = (history) -> (reward, metrics)`.
- **Blocked on account:** Anthropic key in `.env` is invalid (401). Modal secret needs the real value once available.
- **Step-0 with tool declarations injected (`renderer.create_conversation_prefix_with_tools`)**: untrained Qwen3.5-4B called tools correctly in Qwen XML (read_file 1-50 → find_symbol Flask → read_file 81-85), answered correctly (Flask at src/flask/app.py L81, inherits App) in 4 turns / 3 calls / 10s, but wrote the citation as `src/flask/app.py:L81` without the bracket format → format gate → reward 0. Conclusions: (a) tool use works out of the box once declarations are in the prompt, so the SFT warm-start is likely unnecessary; (b) the pass-rate filter must score format and correctness separately or it will reject everything at step 0; (c) the bracket format is a dense, easy signal RL should pick up in the first steps; (d) the first action was a blind 50-line read before find_symbol — the efficiency target.
- **Env rule:** the cookbook tool env does not inject tool specs; `RepoEnv.initial_messages()` must prepend `renderer.create_conversation_prefix_with_tools([t.to_spec() ...], system_prompt)`. Prompt grew from 263 to 740 tokens with two tools' schemas; budget for five.

## 2026-09-18 16:45 · first training proof (lane A, A6)
- Whole chain proven once: real tasks → `RepoEnv` → cookbook loop (LoRA 32, lr 4.9e-4) → sampler checkpoint → `run_episode` from the checkpoint → cited answer. 10 tasks × 4 × 3 steps, 132 s.
- Step-0 base model on graphiti enumerate tasks: format gate 47.5 %, grounded citations 12.5 %, reward 0.10, 40 % of groups had mixed reward → there is a learning signal without SFT warm-start (open question from ideation, now closed).
- After 3 steps on the same 10 tasks: format 87.5 %, grounded 45 %, reward 0.37, tool calls 6.0 → 2.6. Plumbing proof only (repeated tasks), but the checkpoint cited correctly on an unseen flask question.
- Decisions taken on the way: renderer keeps thinking in history; `read_file` lines are `L41 | code`; pure-Python grep fallback (no ripgrep on the Mac); `profiles.yaml` names `claude`, `haiku`, `qwen4b-base`, `qwen4b-<run>-step<N>`; efficiency term for run two must be applied in `compute_group_rewards` where token counts are visible.
- Talk material: `data/logs/smoke1/metrics.jsonl` (first curve), `data/traces/dev/sweqa-flask-001__{qwen4b-base,claude,qwen4b-smoke1-step3}.json` (same question, three models).

## 2026-09-18 evening · five decisions for run one (lead)
1. **Learning rate: 1e-4** for run one, not `hyperparam_utils.get_lr` (4.9e-4). Lane C showed the default collapses when few groups survive `remove_constant_reward_groups` (judged explain tasks); the 1e-4 control learned normally (`data/logs/smoke_lr1e4`).
2. **No-answer penalty: adopted.** An episode that ends without a final answer (`stop=budget|max_turns`) gets −0.1; format failure stays 0, so a bad answer still beats stalling. Same magnitude as the cookbook's `failed_parse_reward` / `context_overflow_reward`. Lane C implements it in the trainer's group hook; it applies from the first run that starts after it lands (run two, or run one if run one restarts). Reward block in `docs/agents/README.md` updated.
3. **Judges:** Haiku 4.5 for training reward and the pass-rate filter; Sonnet 5 only for the final SWE-QA-Bench number in the evals table. Never Sonnet inside the loop.
   *Measured 2026-09-18 22:10 (lane C), on Claude Sonnet 5's 20 judged answers from the fast set:* Haiku 4.5 re-judging the same answers agrees with itself to a mean absolute difference of **0.01** in rubric fraction; Haiku vs Sonnet 5 differ by **0.07** on average (means 0.93 vs 0.92), and every disagreement is a single rubric item. Judge cost per call ≈ $0.003 (Haiku) vs ≈ $0.007 (Sonnet). Conclusion: the cheap judge is as consistent as the expensive one for atomic yes/no rubrics, so Haiku stays in the loop; the SWE-QA five-dimension judge (Sonnet) is a different instrument for the external number only. Same audit showed Claude's low fast-set correctness (0.22) was 76/120 answers over the old 450-token cap, not judge error: among answers that reached grading, mean correctness was 0.82. (LOG 22:10.)
4. **Run-two efficiency formula: accepted as implemented by lane C.** First half of the budget free, then linear 1.0 → 0.5 at 100 % of budget, over tool calls and prompt tokens (prompt-token budget = max_tool_calls × 3000); redundant reads count as extra calls. Gating order stays: format → citations parse → exist → grounded → budget → correctness × efficiency.
5. **Pass-rate filter scale (B6): two samples per task, not four**, judged types filtered after verifiable ones, per-repo cap 120 as written. Run one does NOT wait for it: it starts on the verifiable raw tasks (structural + CodeScout + DeepCodeBench enumerate/locate/value with gold paths); the filtered `train/all.jsonl` feeds run two.
Run-one config: `codeqa.trainer.run`, profile `qwen4b-base`, group 8, 16 groups per step, 50 steps, lr 1e-4, `remove_constant_reward_groups=True`, variant `none`, eval every 10 on `eval/fast.jsonl` (or `smoke_sweqa_flask` until B6 writes it). The lead launches it; nobody else starts training runs.

## 2026-09-18 evening · answer caps raised (decision #6)
- `DEFAULT_BUDGETS.max_answer_tokens`: locate 250 → 400, value 200 → 300, enumerate 350 → 500, trace 400 → 600, explain 450 → 800. Reason: lane C's baseline showed Claude failing the format gate on SWE-QA explain purely on length (p50 671 tokens, 0/46 passing at 450), which also would have discarded most teacher traces for judged types. Base Qwen answers average ~190 tokens, so run-one cost is unchanged. The cap is in the prompt, so it is fixed now, before run one, and must not change between run one and inference.
- Consequence: lane C re-runs the two `fast` baselines with the new caps (same 120 tasks, same seed) so the talk table is consistent; the Sonnet SWE-QA judge number is unaffected (no length gate).

## 2026-09-18 evening · grounding counts grep-shown lines (decision #7)
- Claude's held-out losses, per task: 82/120 failed the length cap (fixed by #6), 7 failed grounding, 2 had no citation, 29 reached correctness. Of the 20 judged answers that got through, all scored 0.67–1.0 (15 at 1.0), so the Haiku rubric judge is not the reason the strong model scored low. The three enumerate zeros include a task with a rubric but no gold paths (lane B to fix; grader now falls back to the judge for such tasks).
- The grounding failures were citations of single lines the model had seen in `grep` output (`path:Lnn: text`) but never opened with `read_file`. The rule is "cite only lines you have seen"; a grep hit shows the line's content, so it is now recorded as a seen span (`RepoTools.grep`). `find_symbol` ranges are NOT recorded: they show a range without its content. Reward change → applies from run one.

## 2026-09-19 afternoon · run-one signal (lead instruction "keep ≥ 1k data, make the rest of the improvements"; implemented by lane C)
- **Data stays at 1,489.** Re-windowing on our reward would leave < 300 tasks (only 8 % of raw tasks ever scored under it in base samples).
- **32 groups per step** (was 16): the training-file probe (40 tasks × 8, temperature 1.0, current caps + grep rule) showed reward variance in 12 % of groups; 32 groups gives ≈ 4 signal-bearing groups per step.
- **Grounded-citation credit +0.05**: an answer that passes every gate (formatted, bracket citations that exist and were read) but is wrong gets a floor of 0.05. Ladder: stall −0.1 < no citations 0 < grounded-but-wrong 0.05 < correct. Purpose: give the bracket-citation format a gradient (41 % of base episodes answer without any citation). Bounded: a policy that games it caps at 0.05. Evals report the unshaped reward. `--grounded-credit 0` turns it off.
- **SFT warm start prepared, not run** (`codeqa/trainer/sft.py`): trigger is `citations_parse` flat by step 10. Needs Claude traces over training tasks (~$30 for 200); lane B kept no teacher traces.

## 2026-09-19 · where things run (decision #8)
- **Tinker**: sampling and LoRA training, unchanged.
- **Modal**: the agent runtime jobs — the cookbook training loop with `RepoEnv` tools and the grader, evals, the pass-rate filter, the teacher — as detached functions over the `codeqa-data` volume (`/data` = `CODEQA_DATA_DIR`), with the `codeqa` secret for keys. The bash sandbox pool lives there too. The volume is the source of truth for repos, index, tasks, logs, evals, traces, models manifest and `profiles.yaml`; the laptop and the EC2 box sync down from it.
- **EC2**: the product only (API + web), per `docs/agents/lanes/E_deploy.md`; product episodes run on the box against synced snapshots and Tinker sampling.
- Why: the laptop was the bottleneck for parallel evals/filters and had to stay awake for runs; Tinker throughput is unchanged by any of this. Runner: `apps/trainer/modal_runner.py`.

## 2026-09-19 evening · reward fixes from the red-team (lead: "fix the ones you agree with"; lane C)
Adopted, all offline-tested against the adversarial fixtures and the reviewer's exploit scripts: (1) grounding tolerance of one line around any shown span, near misses logged not rewarded; (2) symbol credit when a citation overlaps the definition body, or when a grounded cited line contains a gold symbol's name (call sites), with symbols named in the question excluded from precision; (3) precision discounts on single-gold any-of tasks; (4) path-only tasks must name the file outside the bracket or score half; (5) trivial value literals need a grounded citation on the task's evidence lines; (6) judge accepts string booleans; (7) run-two efficiency measured on the final context with budget = prefix + 1,500 × calls (smoke rollouts: median usage 0.52, was 1.88 under the cumulative count), grep hits never counted as redundant reads.
Not adopted: squaring judged scores (B10) and grounded-fraction instead of the gate (B12) until `grounded_by_tolerance` from run one shows how often near misses occur.

## 2026-09-19 night · answer length is a soft term, not a gate (lead: "don't care about conciseness, reward based on length")
- Before: an answer over `max_answer_tokens` (400/300/500/600/800 by type) scored 0. Sonnet 5 exceeded the cap on 38 % of fast-set answers, all cited and mostly correct: its reward was 0.48 with 52 % of answers reaching grading, while the same traces with the cap lifted scored 0.79 with 88 % reaching grading and only 2/103 judged wrong.
- Now: `length_factor = min(1, cap / tokens)` (floor 0.1) multiplies the reward; the prompt still says "keep it under N tokens", so the guidance is unchanged and no data changes. Re-graded on the same Sonnet traces: reward 0.73, 88 % reach grading, 87 % correct; over-cap answers keep 0.82 on average.
- The threat the hard cap defended against (pasting tool output to hit rubric words) is now a separate gate: > 50 % of the answer's substantive lines verbatim from a tool result → format failure, reward 0 (`gates.verbatim_share`, fixture `verbatim.json`).
- What still keeps Sonnet below 90 %: partial rubric credit (31/117 answers at ~0.8, one missing fact each), 7 episodes that hit `max_turns` without answering (an env budget, not a grader rule), 4 grounding failures, 2 driver errors. Raising `max_turns` for explain/trace is the lever for "reached grading"; partial credit is the judge doing its job.
