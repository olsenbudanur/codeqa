# Project timeline: events and decisions, in order

Compiled from `docs/decisions.md`, `docs/agents/LOG.md`, the lane files and the user's notes. LOG timestamps were written by different agents on different clocks (some local, some UTC-ish, some with the date rolled forward), so entries below are ordered by what actually followed what; the bracketed stamp is the LOG entry to look up, not a wall-clock time. Decision numbers (#6, #9, ...) are `docs/decisions.md`'s.

## Day 0 morning (Thu Sep 18): ideation

Constraints chosen before any code:

- Rely on OSS and prior papers/data instead of rebuilding (RepoSearch-R1, SWE-QA-Pro, DeepRepoQA, LocAgent as references; CoReQA dropped, never released).
- Read-only tools, no sandbox, no Docker: a repo is a folder, a tool is a function. Removes the hardest infra piece.
- Harness = tinker-cookbook `tool_use` / `search_tool` recipe, not a custom loop.
- Pre-index every repo declaratively (tree-sitter symbols, Haiku directory summaries, token-capped map); train only the exploration. `overview` replaces embeddings.
- Five tools: overview, find_symbol, grep, read_file (line ranges), list_dir. Outputs curated and always line-numbered.
- Single-shot Q&A, no follow-ups, no issue-to-patch tasks. Issue data (CodeScout) used only as a source of questions with location gold.
- Reward = outcome over the whole trajectory: gates (format, citations exist and were read) -> correctness (verifier or rubric judge) -> efficiency multiplier. Run one correctness-only, run two adds efficiency. Grounding and efficiency are the gap vs prior judge-only rewards.
- Data from a census (2,705 verified natural repo Q&A exist publicly; 5k is not reachable): DeepCodeBench train (facts = rubric), CodeScout-derived locate tasks, a structural generator over our own index, a Claude teacher (answer first, blind Haiku verify). Eval: DeepCodeBench test, SWE-QA-Bench, Code-QA-Bench. Target pass rate 0.3 to 0.7 of 8 samples.
- Few repos on purpose (~23 train + 15 eval snapshots). Thinking on. Traces on disk, no WandB. Tinker for training and inference; Anthropic for teacher/verifier/judge/summaries; HF datasets-server and GitHub tarballs for data.
- Repo layout: one folder per component, `shared/` contracts are law, `clients/`, `apps/`, `data/`. Product UI = shadcn dashboard over a FastAPI SSE backend.
- Grader refinements from the RepoSearch-R1 review: symbol-set F1, identifier grounding via the index, per-group diversity metrics, docstring-stripped snapshots for structural tasks.
- Open questions carried in: CodeScout license; can base 4B emit the citation format at all (decides SFT warm-start); judge-failure policy.

## Day 0 ~13:00: sync setup, then fan-out

- Contracts validated against real rows; flask snapshot + index in 3 s; Tinker, Modal, GitHub, HF proven; one full cookbook episode with real tools. [LOG 09-18 13:00]
- Step-0 behaviour: untrained Qwen3.5-4B answered without tools, copied the prompt template, fabricated a citation -> reward 0. With tool declarations injected it used tools correctly and answered correctly but wrote `src/flask/app.py:L81` without brackets -> format gate. Conclusion: tool use works out of the box, only the citation format is missing, so no SFT warm-start is needed to get a learning signal. [decisions day 0; LOG 13:05]
- Rule found: the cookbook env does not inject tool specs; `RepoEnv` must prepend `create_conversation_prefix_with_tools` (prompt 263 -> 740 tokens with two tools).
- Work split into lanes with a shared LOG as the data bus: A agent runtime (lead), B data, C training + grader + evals, D product; E deploy scoped later. Contracts first, fan out, smoke test before data.

## Day 0 afternoon: every component exists once

- Contracts get three additive fields (`Message.usage`, `Message.parse_error`, `EndpointProfile.base_model`); renderer keeps thinking in history; `read_file` lines become `L41 | code` because the Qwen renderer strips leading whitespace. [14:20, 14:50]
- D1: web shell on a recorded episode; Vite SPA instead of a Next.js fork; `CITATION_RE` mirrored in TypeScript with a parity test. [13:25]
- C: Tinker re-verified; cookbook facts; NaN policy for judge failures (fill with the group mean, advantage 0). [14:10]
- A2: indexing CLI (snapshot -> index -> Haiku summaries -> map <= 3k tokens; `__nodoc` variants). [14:50]
- B1/B2: 23 repos snapshotted and indexed in 70 s; DeepCodeBench 912/232, SWE-QA 720; citation path resolution 65% -> 92%; CodeScout adopted (N = 15 repos, attributed to SWE-rebench). [15:10, 15:15]
- A3/A4: `RepoTools` and `RepoEnv`; no ripgrep on the Mac, pure-Python grep fallback. [15:15]
- C1: grader done (56 offline tests + live Haiku check); efficiency shaping: first half of budget free, then linear to 0.5. [15:20]
- A5: `run_episode` drives Tinker and Anthropic clients identically; `profiles.yaml` naming fixed (`claude`, `haiku`, `qwen4b-base`, `qwen4b-<run>-step<N>`). [15:40]
- C2/C3: trainer and evals run end to end on real Tinker. Finding: the cookbook's recommended LoRA lr (4.9e-4) collapses the policy after one step on judged explain tasks (only 2 groups with variance survived the constant-reward filter). Sonnet 5 spends all tokens thinking unless thinking is disabled for judges. [16:40]
- A6: first training proof, `smoke1`: 10 tasks x 4 x 3 steps, reward 0.10 -> 0.37, format 47% -> 88%, tool calls 6.0 -> 2.6; the step-3 checkpoint cites correctly on an unseen flask question. A7: the env/driver/client API is frozen. [16:45]
- B3/B4: 1,017 CodeScout tasks, 1,328 structural tasks over 23 repos; teacher loop running. [17:05, 18:25]
- C: the lr 1e-4 control does not collapse (0.18 -> 0.28, std ~0.3). Recommendation: 1e-4 and a task mix with step-0 group variance. [17:30]
- Anthropic credits exhausted the first time (~18:35): teacher, judge and summaries blocked; Tinker unaffected. [18:50]
- Five run-one decisions (lead): lr 1e-4; no-answer penalty -0.1 (stall < bad answer < correct); Haiku in the loop, Sonnet only for the final external number; run-two efficiency formula accepted; pass-rate filter at two samples per task. Run one starts on the verifiable raw tasks without waiting for the filter. Only the lead launches training. [18:40]
- `eval/fast.jsonl` (120 tasks, 60 DeepCodeBench test + 60 SWE-QA) fixed as the held-out set. [18:50]
- No-answer penalty lands; step time shown to be sampling-bound (grader + judge < 5%); a CLI monitor with collapse checks. [19:30]
- D2: the API streams real episodes with `claude` and `qwen4b-smoke1-step3`. D4: `/compare` page and the demo doc (base vs step-3 on flask: format learned first, grounding next). [19:40, 20:05]
- B: interim step-0 pass rates by source: structural is the only source where the base model earns reward often enough for group variance (strict 29%); CodeScout 1%; DeepCodeBench 13%. [19:40]
- `run1.jsonl` = 1,841 tasks: all structural + DeepCodeBench verifiable + CodeScout trace. [20:20]
- Anthropic balance near zero again during baselines. [20:10]

## Day 0 evening: baselines expose two grader problems

- C baselines on `fast`: Claude Sonnet 5 reward 0.20 / correct 22%; base 0.00. 82/120 of Claude's answers fail the format gate purely on length (p50 671 tokens vs a 450 cap); every over-cap answer had bracket citations. [21:00]
- Decision #6: answer caps raised (explain 450 -> 800, locate 250 -> 400, ...). The cap is in the prompt, so it is fixed before run one. [21:20]
- SWE-QA external judge (Sonnet, five dimensions): Claude 73 vs base 41. Under our own reward both were ~0 because of the cap: the strongest argument for deciding the cap before run one. [21:40]
- Decision #7: grep hits count as read (all 7 of Claude's grounding failures were citations of lines seen in grep output but never opened). `find_symbol` still records nothing. [21:45]
- C audit "is the judge or the data wrong?": neither. Haiku agrees with itself to 0.01 and with Sonnet to 0.07; three DeepCodeBench rows were bad and lane B fixed them at import time. [22:10, 22:40]
- D6 "Workshop" specified: runs, checkpoints, data, traces pages, read-only over `data/`. [22:30, 22:45, 22:55]
- Lane B handoff: `train/all.jsonl` = 1,489 tasks after the pass-rate window [0.1, 0.9] on two to four base-model samples per task (62% programmatic, 38% judged); eval files final; ~$52 of Anthropic spend. Caveat recorded then: the window was measured on a lenient metric and under the old caps and grounding rule. [23:20]
- Agent variants added for ablations, off by default: `noindex`, `nomap`, `bash` (one read-only sandboxed shell tool). [23:20]
- E1 deploy scoped; secret scan script. [23:40]
- B pins `fast.jsonl` to the ids already evaluated; more DeepCodeBench fixes (the C++ twins). [23:50]

## Night 0 -> 1: frontier validation, Modal, Workshop, run one

- B validates the fast set with frontier models: Sonnet clears every gate on 61/120 (mean correctness 0.86 among those); Opus on the 18 hardest; a bad-test hunt removes 3 tasks and repairs 6. The limiter is the cap and answering without a final message, not the judge. Recommendation: make length proportional, not a gate. [00:05, 00:30, 00:55, 01:20]
- The `bash` variant runs on Modal sandboxes (0.7 s per command, 32 concurrent). [00:10]
- D6 Workshop done (runs, checkpoints, data, traces; 944 traces listed; run overlay shows the lr collapse). [00:30]
- Decision #8: agent runtime jobs (train, evals, filter, teach) move to Modal as detached functions over the `codeqa-data` volume; Tinker unchanged; product on EC2. Runner proven for evals, then for a training step (100 s), then for both variants; `CODEQA_RUNTIME=modal` becomes the default in `.env`. [00:40, 01:20, 01:45, 02:30, 02:50]
- D: optimizer metrics + Health panel on the run page; `/workshop/live` one-screen monitor; password gate on API and web. [01:10, 01:40, 03:10]
- Lane B (at the lead's request) makes a lean context the default: four tools, ~1k structural tree map, no Haiku summaries anywhere; the day-one five-tool design becomes `full`. Reason: RepoSearch-R1, LocAgent, ToolTrain give the model no prose summaries; `overview` was the only paid, non-deterministic step and the map was a third of Sonnet's prompt tokens. Callers gate on `symbols.json` instead of `map.txt`. [02:20, 02:45]
- Run one launched on the laptop: `run1.jsonl`, 50 steps, 16 groups x 8, lr 1e-4, eval every 10. Filters paused to give Tinker throughput to training. [03:00]
- Meanwhile lane C (their own clock, 15:00 -> 23:10): baselines re-run under the new caps and grep rule (Claude 0.45 / 48%, base 0.04 / 4%); training-file audit (under our reward only 8% of kept tasks ever scored > 0; step-0 group variance 12%; recommend 32 groups per step, a +0.05 grounded-citation credit, SFT warm start as a fallback); all three land (`groups_per_batch` 32, `grounded_credit`, `sft.py` prepared but not run). [15:00, 15:40, 16:30, 16:50]
- Independent reward red-team: 12 failure modes, 9 verified with crafted traces; the judge is sound; the gates produce three classes of false zeros (off-by-one around grep spans, symbol credit needing the `def` line, content-free path answers scoring 1.0). Seven rules changed the same evening; run-two token accounting fixed (budget measured on the final context, `TOKENS_PER_CALL` 3000 -> 1500). [18:20, 19:30]
- Data hygiene: `modal_sync.sh --force` replaced local eval outputs with the volume's copy and deleted the base baselines and the agent-variant comparison; fixed to pull into a staging dir and merge without delete. [20:10, 04:30]
- Answer length becomes a soft term (`length_factor = min(1, cap/tokens)`), then training-only shaping: the grader is length-free, evals measure content. Verbatim tool-output pasting gets its own gate. Sonnet on the same traces: 0.48 -> 0.73 -> 0.80; Opus 0.80 on Sonnet's hard tasks. [21:40, 23:10; decisions night (1) and (2)]
- Lane B imports Code-QA-Bench (528 tasks, 10 repos) as a third eval set and ports its judge verbatim; first numbers: Opus 0.75 on their scale through our lean agent, base 0.28. [03:20, 03:50, 04:20]
- Prompt caching for the Anthropic client (turn 2 reads 6,198 tokens from cache); an opt-in tool-output trimming knob for the product. [03:40]
- Run one, step 10: held-out reward 0.04 -> 0.31, correct 0.04 -> 0.31, format 0.53 -> 0.97, grounded citations 0.05 -> 0.92, tool calls 7.6 -> 4.4, max-turn stalls 0.28 -> 0.00. Profile `qwen4b-run1-step10` becomes the demo checkpoint. [04:05]
- Run one stopped at step 11 to save Tinker spend. Rule: nobody launches training or large evals without the lead. [04:15]
- D: every saved checkpoint is askable; per-event timing; format failures shown as errors; compare takes four models with an Opus referee that researches the question itself; repo page with a tool console; "everything the model saw" trace tab; five typed example questions per repo. [04:20, 05:10, 06:20, 08:10]
- Product finds a grounding blind spot: `find_symbol` shows a signature but registers nothing, so the same evidence is grounded via grep and not via find_symbol. Lead's addendum: find_symbol and overview register the signature line they show. [09:00, 04:50]
- Freeze on `codeqa/agent` after the lean default landed without a LOG entry while the lead was editing the same files; lifted the same night with a contract addition: `EndpointProfile.variant`, so a checkpoint is always served and evaluated with the agent it trained on. [04:50, 05:00]
- Rubric pruning rule adopted: keep an item iff a frontier answer states it (17 of 164 items in the hard set were true facts the question did not ask for). Applied to fast (4% dropped) and, after Sonnet on all 644 judged train tasks, to the train files (9%). Credits run dry again mid-way. [09-20 00:20, 02:40]
- Loss curve: Tinker never returns a loss, so lane C wraps the cookbook's KL hook to log the IS surrogate, `loss_abs`, advantage std, importance ratios, clip fraction, NLL. D adds the panels. [00:50, 01:30]
- Grounding becomes a multiplier (share of cited lines shown), the gate fires only at zero; find_symbol registers the full displayed range; judge window 3x cap; verbatim gate needs >= 8 lines. Opus reaches grading on 96%, mean reward 0.86. Open ask to lane A: a forced final answer turn when the tool budget is exhausted. [01:20, 02:10; decisions 09-20 early]

## Day 1 (Fri Sep 19): phase 1 and the Tinker incident

- Phase 1 (shape) launched on Modal: four arms in parallel, `p1_full`, `p1_lean`, `p1_bash`, `p1_nogates`, 30 steps each, identical batches; `honesty_gates()` switch added for the nogates arm. Relaunched on `all.jsonl` after a path bug. `runs.json` titles for the Workshop. [05:30, 05:50, 06:10]
- The arms ran 32 groups (256 episodes per step) because `scripts/arm.py` never passed `--groups-per-batch`. [08:05, 08:10]
- Phase 1 stopped: Tinker balance flat, then a Modal preemption restarted an arm from scratch and `--if-exists delete` wiped its run directory. Fixes: 16 groups, save every 5 steps, resume on restart, 16 GB containers, per-process index caching. [08:30]
- Concurrency test with two arms, then the product's measurement: 47 s for an 8-token sample; sampling queued, not refused. [08:50, 08:55]
- Incident: Tinker sampling hangs account-wide. First root cause: 133 sessions from the day still "Active" with their processes dead; all finished via the SDK; the product's session had been cancelled server-side. Rules: never kill a Tinker job mid-flight, one `ServiceClient` per process closed on exit. [09:10, 09:45]
- Product finds the org has `sample_cancel_enabled=False`, so abandoned samples keep computing server-side; adds a dead-session self-heal to the API. Then: 8 tokens in 6.8 s but 256 tokens not done in 150 s under four arms, so it is throughput, not admission. All arms stopped on the user's instruction; one arm at a time from here. [09:20, 09:30, 09:40]
- Lead: LoRA sampling paused on the account (base sampling works, any saved LoRA sampler times out); `scripts/queue.py` runs arms sequentially behind a one-token canary; per-rollout timeout added to the trainer; session hygiene and a no-progress watchdog; support draft written. Census: 160 sessions / 234 samplers, ~600 orphaned requests draining at one every few minutes. A new Tinker org is created for training; the old org keeps the demo checkpoints. [10:00, 10:10, 10:15, 10:25, 10:40]
- Product: with everything stopped and every session finished, five base models probed from one session: Qwen3-8B 1.7 s, Qwen3.5-9B 1.6 s, gpt-oss-20b 1.7 s, Qwen3.5-4B 52 s. The user reproduces it from the new org. Conclusion: the shared Qwen3.5-4B pool itself is congested; `qwen9b-base` added to the product as a comparison. [09:55, 10:00]
- Phase-1 queue runs in the new org, one arm at a time, ~8 min per step; step-1 funnel on the full set: answered 45%, cited 7%, correct 6%. [21:41]
- Decision #9: the full task set is a dead end for a cold start (held-out 1.7% after 10 steps vs run one's 66% on the same tasks; winners hit the turn cap like losers, so no "answer earlier" gradient). Phase 1 trains on `run1.jsonl`; caps raised to >= 20 calls; the prompt states both limits and the exact citation form. `p1_full_allset` kept as the exhibit. [22:13]
- Incident: the relaunch command wiped the afternoon's four-arm runs. Rule: relaunch = rename, never delete. [22:32]
- 4B LoRA sampling blocked in the second org too (a fresh 4B LoRA probe took 257 s); phase 1 restarted on Qwen3.5-9B with the canary always on. [22:42]
- Bug fix and Decision #10: since `RetryOnFailure` went in, rewards were paired with the wrong trajectory (completion order vs creation order; 35 to 67% of records misaligned in every run after run one). Every verdict from those runs is void. Fixed run at the same steps: cited 76% vs 35% before, correct 69% vs 27%, reward +0.53 -> +0.65 by step 2. The prompt now leads with the citation rule. [23:06]
- 9B queue stopped by the user on cost (~$100 for ~25 steps, ~$4 per step; the fixed prefix re-sent every turn is half of all prefilled tokens). [23:40]
- Product hands lane C the client failure modes (dead session; wedged per-client poller) and three training changes: `MinViableGroup` instead of 8 retries, resume-on-crash in the queue, canary gated on latency. [12:00]

## Day 1, second half (LOG dated 09-20): phases 2 to 7

- Phase 2: four harnesses on 4B with a cheap config (12 groups x 8, caps 12/14, tool outputs capped), `merged.jsonl` (run1 U all minus value, 2,340 tasks), a $100 Tinker guard that writes STOP at a step boundary; cost model ~$0.55 to 0.75 per step. Phase 3 `p3_bash`: train reward 0.20 -> 0.65, correct 33% -> 81%, calls 12.9 -> 6.7; held-out 0.40 -> 0.48. The first clean, transferring curve since the pairing fix. [(late), 09:50]
- Phase-3 queue crashed on a volume `reload()` race during the final eval; reload made opt-in, mkdir retries. [09:50]
- Decision #11, phase 4: stratified batches (same task-type mix every step; 8-task draws alone moved reward +-0.15), Sonnet 5 as judge (Haiku's rubric scores were the noisiest component), 40-task final eval, `2>/dev/null` allowed; relaunched after an ordering bug; per-run sync because `data/logs` passed 500 MB. [10:14]
- Phase 5: full + nogates on 9B in the second org while the first org's 4B pool stalled again; stopped at step 9 (9B answers got longer under this reward, 4B's shortened); 4B pool back, `p4_full` resumed from its checkpoint. [11:52, 12:12]
- Lane B: natural-question sources become judged-only ("how" -> trace, else explain; 66 train + 165 eval rows had been path-F1 against paths regexed from prose); train files rubric-only, no references. [09-20 datagen entries]
- Reward v2 and bash fixes from a 1,536-episode trace audit of `p4_bash`: the blocklist regex matched `python` inside paths and rejected 799 legal commands; ~220 citations overshot EOF by 1 to 3 lines; the prompt's `src/...` example was copied into 147 fake paths. v2: per-claim citation credit instead of the all-or-nothing existence gate, length floor for explain only, judge scores `(satisfied - contradicted)/items`. New files `all_noteacher.jsonl`, `eval/balanced.jsonl`. [13:05]
- Local probes on 9B and 4B bash (8, 12, 24 tasks) fix grader false negatives: unnumbered `sed`/`head`/`cat` output now registers the printed range; verbatim thresholds relaxed; gold validation over all training tasks finds 0 missing paths. 24 tasks at 24 calls: 4B 0.22, 9B 0.37; remaining low scores are grader-correct or CodeScout gold. [13:40 to 15:15]
- Decision #12, the reward mistake, written for the talk: R paid for being right and honest, never for being complete; every extra claim carried gate risk and the token-summed loss punished long wrong answers more than short ones, so the policy learned to say less (grounding 54 -> 72% and 82 -> 90%, calls -20%, answer length -40%, explain rubric coverage down). Run-two fix: graded rubric coverage as the explain correctness term, per-claim credit, a length floor. [11:10]
- Phase 6: four 4B arms on `all_noscout.jsonl` (CodeScout excluded: PR-touched gold disagrees with the question about half the time), reward v1 control vs v2, a 24-call efficiency arm, the product harness. The queue silently ran its default arm list instead (fixed: unknown arm names now exit). `p6_full`: 12 steps in 21 min, held-out balanced 0.43/63% -> 0.52/87% at step 6 -> 0.47/83% final, $23. `qwen4b-p6_full-step12` becomes the product's default model. [15:40, 13:50, 14:05]
- Decision #13, phase 7 (`bash_v3`): rounds (a message may carry several commands), a context cap (32k prompt tokens) and message cap instead of a call cap, a remaining-budget trailer after every round, a forced tool-free final answer when the budget runs out, grep self-healing; reward v3 = c x (0.75 + 0.25 grounded), efficiency on true token cost only after step 8. Queued after the running bash arm. [15:05, 15:20]
- The STOP file on the volume is invisible to a running container without `reload()`; stop signal moved to a Modal Dict. [15:45]

## Mistakes, in one place

From the user's notes and the LOG:

- Paid only for being right and honest, never for being complete; the token-summed loss then shortened answers (decision #12).
- Filtered the training set before the agent and grader were final: the pass-rate window was measured under the `full` variant, the old caps and the old grounding rule, on a lenient metric; under our reward only 8% of kept tasks had ever scored.
- Too much variance per step from the task-type draw (8 to 12 tasks per batch moved reward +-0.15); fixed late with stratified batches.
- `overview` and Haiku summaries were a dead end; the lean context replaced them.
- grep needed to degrade gracefully (regex errors, zero hits); self-healing only arrived in phase 7.
- Could not iterate fast enough: the Qwen3.5-4B sampler pool, orphaned sessions from killed jobs, two Anthropic credit outages, volume syncs that deleted local outputs, a relaunch that deleted runs.
- The cookbook's recommended lr collapsed the policy in one step on sparse-signal batches.
- Reward pairing by index with a completion-order retry strategy scrambled credit in every run between run one and the fix.
- The hard answer cap zeroed most frontier answers, so early baselines were uninterpretable until length became a soft, training-only term.
- `find_symbol` showed evidence it did not register, so identical evidence was grounded through grep and not through find_symbol.
- Jobs launched with omitted flags (32 groups by default; the queue's default arm list) and infra assumptions (`--if-exists delete` on a preempted restart, a STOP file no container could see).
- CodeScout gold is "entities the fix PR touched", which disagrees with the question often enough to be excluded from the reward A/B.
