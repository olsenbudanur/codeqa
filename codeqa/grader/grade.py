"""task + trace -> GradeResult (C7). Gates in order, then correctness, then efficiency.

  gates:   format -> citations parse -> citations exist -> grounding (only if NO cited line was shown) -> budget    any failure = 0
  correct: verifiers (zero API calls) for locate|value|enumerate|codescout; judge for trace|explain with rubric/reference
  reward:  correctness * efficiency * grounded_fraction (share of cited lines shown)     judge failure -> NaN (gate_failed = judge_error)
"""
from __future__ import annotations

import asyncio
import math

from codeqa.grader import gates
from codeqa.grader.citations import check_citations, grounded_fraction, grounded_only_by_tolerance
from codeqa.grader.efficiency import context_tokens, efficiency, redundant_reads, token_sum, token_sum_efficiency
from codeqa.grader.judge import JudgeClient, judge
from codeqa.grader.repo import RepoFiles, load_repo
from codeqa.grader.verifiers import uses_judge, verify
from codeqa.shared.contracts import GradeComponents, GradeResult, Task, Trace

import os

# Ablation switch (decisions.md, phase-1 no-gates arm): when off, cited-file-exists and cited-lines-read no longer zero the
# reward; the components still record the fractions so fabrication shows up in env/all/citations_{exist,grounded}.
def honesty_gates() -> bool:   # read per call so a job can train ungated and still score its checkpoint with the common grader
    return os.environ.get("CODEQA_HONESTY_GATES", "on").lower() != "off"


def reward_version() -> str:
    """CODEQA_REWARD=v1 restores the 2026-09-19 reward (all-or-nothing citation-existence gate, no length floor) for controls.
    v3 (2026-09-20, bash_v3): correctness first, grounded fraction and token efficiency as secondary weights (see grade())."""
    return os.environ.get("CODEQA_REWARD", "v2")


V3_GROUNDED_WEIGHT = 0.25


def v3_efficiency_weight() -> float:
    """CODEQA_EFF_WEIGHT (default 0.15) applies from training step CODEQA_EFF_WEIGHT_FROM_STEP (default 0); the trainer
    publishes the current step as CODEQA_TRAIN_STEP. Before that step the weight is 0 and the term is only measured."""
    w = float(os.environ.get("CODEQA_EFF_WEIGHT", "0.15"))
    start = int(os.environ.get("CODEQA_EFF_WEIGHT_FROM_STEP", "0"))
    step = int(os.environ.get("CODEQA_TRAIN_STEP", "0"))
    return w if step >= start else 0.0


def _fail(gate: str, note: str, comps: GradeComponents) -> GradeResult:
    return GradeResult(reward=0.0, components=comps, gate_failed=gate, notes=note)


async def grade(task: Task, trace: Trace, variant: str = "none", judge_client: JudgeClient | None = None,
                repo: RepoFiles | None = None, reward: str | None = None) -> GradeResult:
    """`reward` pins the reward version (v1|v2|v3); None reads CODEQA_REWARD. The held-out evaluator pins v2 so its
    numbers stay comparable across phases whatever the training reward is."""
    rv = reward or reward_version()
    budget = task.effective_budget()
    repo = repo or load_repo(task.repo_id)
    comps = GradeComponents(efficiency=1.0)
    answer = gates.extract_answer(trace)

    ok, why = gates.format_gate(answer, trace, budget)
    if not ok:
        return _fail("format", f"format: {why}", comps)
    comps.format_ok = 1.0

    ok, why = gates.citations_parse_gate(answer)
    if not ok:
        return _fail("citations", f"citations: {why}", comps)
    comps.citations_parse = 1.0

    report = check_citations(answer, trace.stats.files_read, task.repo_id, task.grading.expected_symbols, repo=repo)
    n = max(len(report.citations), 1)
    if honesty_gates():
        if rv == "v1" and not report.all_exist:                          # v1: one bad path zeroes the answer
            bad = [f"{c.path}:L{c.start}-L{c.end}" for c in report.citations if not c.exists]
            return _fail("citations", f"citations: not in snapshot: {', '.join(bad[:3])}", comps)
        comps.citations_exist = sum(c.exists for c in report.citations) / n      # v2 (2026-09-20): per-claim credit; a bad path
        gf = grounded_fraction(report, trace.stats.files_read, repo)             # is simply an ungrounded claim in the fraction
        if gf == 0.0:                                     # nothing cited was shown: still a gate
            bad = [f"{c.path}:L{c.start}-L{c.end}" for c in report.citations if not c.grounded]
            return _fail("grounding", f"grounding: nothing cited was read: {', '.join(bad[:3])}", comps)
        comps.citations_grounded = gf                     # fraction of cited lines shown; multiplies the reward (2026-09-20)
    else:  # ablation (CODEQA_HONESTY_GATES=off): fabrication is not gated, only measured
        comps.citations_exist = sum(c.exists for c in report.citations) / n
        comps.citations_grounded = sum(c.grounded for c in report.citations) / n
    comps.identifier_grounded = 1.0 if (not task.grading.expected_symbols or any(c.anchors_symbol for c in report.citations)) else 0.0

    ok, why = gates.budget_gate(trace, budget)
    prefix, final = context_tokens(trace)
    eff, over_cap = efficiency(trace.stats, budget, variant, prefix, final or None)
    if not ok or over_cap:
        return _fail("budget", f"budget: {why or 'over hard cap'}", comps)
    comps.efficiency = eff

    if uses_judge(task):
        g = task.grading
        v = await judge(task.question, answer, g.rubric, g.reference_answer, budget.max_answer_tokens, client=judge_client)
        if v.failed:
            comps.correctness = float("nan")
            return GradeResult(reward=float("nan"), components=comps, gate_failed="judge_error", notes=f"judge failed: {v.error}")
        comps.correctness = v.score
        note = f"judge: {sum(v.satisfied)}/{len(v.satisfied)} items"
    else:
        comps.correctness, note = verify(task, answer, report, repo)

    ground = comps.citations_grounded if honesty_gates() else 1.0     # partial grounding scales the reward; the ablation arm ignores it
    if rv == "v3":
        # correctness first; grounded fraction and token efficiency are secondary weights that only a correct answer earns.
        # R = c x ((1 - wg - we) + wg x grounded + we x efficiency); length and the forced-answer factor are trainer shaping.
        w_e = v3_efficiency_weight()
        comps.efficiency = token_sum_efficiency(trace, answer_tokens=gates.approx_tokens(answer))
        reward = comps.correctness * ((1.0 - V3_GROUNDED_WEIGHT - w_e) + V3_GROUNDED_WEIGHT * ground + w_e * comps.efficiency)
        note += f"; v3 grounded {ground:.2f} eff {comps.efficiency:.2f} (w_e {w_e:.2f})"
        return GradeResult(reward=reward, components=comps, gate_failed=None, notes=note)
    reward = comps.correctness * comps.efficiency * ground             # length is NOT here: the trainer applies gates.length_factor as shaping
    if honesty_gates() and ground < 0.999:
        note += f"; grounded {ground:.0%} of cited lines (x{ground:.2f})"
    return GradeResult(reward=reward, components=comps, gate_failed=None, notes=note)


def grade_sync(task: Task, trace: Trace, variant: str = "none", judge_client: JudgeClient | None = None,
               repo: RepoFiles | None = None, reward: str | None = None) -> GradeResult:
    return asyncio.run(grade(task, trace, variant, judge_client, repo, reward))


def metrics(result: GradeResult, trace: Trace, task: Task) -> dict[str, float]:
    """Flat metrics for the trainer and evals. Aggregated by the cookbook under env/all/ and env/<tag>/."""
    c = result.components
    nan = math.isnan(result.reward)
    m = {
        "reward": 0.0 if nan else result.reward,
        "format_ok": c.format_ok,
        "citations_parse": c.citations_parse,
        "citations_exist": c.citations_exist,
        "citations_grounded": c.citations_grounded,
        "identifier_grounded": c.identifier_grounded,
        "correctness": 0.0 if math.isnan(c.correctness) else c.correctness,
        "efficiency": c.efficiency,
        "judge_error": 1.0 if nan else 0.0,
        "tool_calls": float(trace.stats.tool_calls),
        "tool_errors": float(trace.stats.tool_errors),
        "prompt_tokens": float(trace.stats.prompt_tokens),
        "completion_tokens": float(trace.stats.completion_tokens),
        "answer_tokens": float(gates.approx_tokens(gates.extract_answer(trace))),
        "redundant_reads": float(redundant_reads(trace.stats.files_read)),
        "turns": float(trace.stats.turns),
        "correct": 1.0 if (not nan and result.reward > 0) else 0.0,
        "length_factor": gates.length_factor(gates.extract_answer(trace), task.effective_budget(), task.task_type),
        "answer_over_cap": 1.0 if gates.approx_tokens(gates.extract_answer(trace)) > task.effective_budget().max_answer_tokens else 0.0,
        "verbatim_share": gates.verbatim_share(gates.extract_answer(trace), trace),
        "grounded_by_tolerance": float(grounded_only_by_tolerance(gates.extract_answer(trace), trace.stats.files_read, load_repo(task.repo_id))) if c.citations_parse else 0.0,
        "prefix_tokens": float(context_tokens(trace)[0]),
        "context_tokens": float(context_tokens(trace)[1]),                  # binary; tool_calls / correct = calls per correct answer
        "stalled": 1.0 if trace.stats.stop_reason in ("budget", "max_turns") else 0.0,   # ended without answering
        "forced_answer": 1.0 if trace.stats.forced_answer else 0.0,                     # v3: answered on the injected final turn
        "token_sum": float(token_sum(trace)),                                            # true cost: prompt+completion over every turn
        "calls_per_turn": float(trace.stats.tool_calls) / max(trace.stats.turns, 1),
    }
    for g in ("format", "citations", "grounding", "budget", "judge_error"):
        m[f"gate_{g}"] = 1.0 if result.gate_failed == g else 0.0
    for reason in ("answer", "max_turns", "budget", "overflow", "parse_error", "error"):   # every key present, so means are true rates
        m[f"stop_{reason}"] = 1.0 if trace.stats.stop_reason == reason else 0.0
    return m
