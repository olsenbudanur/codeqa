"""task + trace -> GradeResult (C7). Gates in order, then correctness, then efficiency.

  gates:   format -> citations parse -> citations exist -> citations grounded -> budget    any failure = reward 0
  correct: verifiers (zero API calls) for locate|value|enumerate|codescout; judge for trace|explain with rubric/reference
  reward:  correctness * efficiency     judge failure -> NaN (gate_failed = judge_error)
"""
from __future__ import annotations

import asyncio
import math

from codeqa.grader import gates
from codeqa.grader.citations import check_citations, grounded_only_by_tolerance
from codeqa.grader.efficiency import context_tokens, efficiency, redundant_reads
from codeqa.grader.judge import JudgeClient, judge
from codeqa.grader.repo import RepoFiles, load_repo
from codeqa.grader.verifiers import uses_judge, verify
from codeqa.shared.contracts import GradeComponents, GradeResult, Task, Trace


def _fail(gate: str, note: str, comps: GradeComponents) -> GradeResult:
    return GradeResult(reward=0.0, components=comps, gate_failed=gate, notes=note)


async def grade(task: Task, trace: Trace, variant: str = "none", judge_client: JudgeClient | None = None,
                repo: RepoFiles | None = None) -> GradeResult:
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
    if not report.all_exist:
        bad = [f"{c.path}:L{c.start}-L{c.end}" for c in report.citations if not c.exists]
        return _fail("citations", f"citations: not in snapshot: {', '.join(bad[:3])}", comps)
    comps.citations_exist = 1.0
    if not report.all_grounded:
        bad = [f"{c.path}:L{c.start}-L{c.end}" for c in report.citations if not c.grounded]
        return _fail("grounding", f"grounding: cited but not read: {', '.join(bad[:3])}", comps)
    comps.citations_grounded = 1.0
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

    reward = comps.correctness * comps.efficiency        # length is NOT here: the trainer applies gates.length_factor as shaping
    return GradeResult(reward=reward, components=comps, gate_failed=None, notes=note)


def grade_sync(task: Task, trace: Trace, variant: str = "none", judge_client: JudgeClient | None = None,
               repo: RepoFiles | None = None) -> GradeResult:
    return asyncio.run(grade(task, trace, variant, judge_client, repo))


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
        "length_factor": gates.length_factor(gates.extract_answer(trace), task.effective_budget()),
        "answer_over_cap": 1.0 if gates.approx_tokens(gates.extract_answer(trace)) > task.effective_budget().max_answer_tokens else 0.0,
        "verbatim_share": gates.verbatim_share(gates.extract_answer(trace), trace),
        "grounded_by_tolerance": float(grounded_only_by_tolerance(gates.extract_answer(trace), trace.stats.files_read, load_repo(task.repo_id))) if c.citations_parse else 0.0,
        "prefix_tokens": float(context_tokens(trace)[0]),
        "context_tokens": float(context_tokens(trace)[1]),                  # binary; tool_calls / correct = calls per correct answer
        "stalled": 1.0 if trace.stats.stop_reason in ("budget", "max_turns") else 0.0,   # ended without answering
    }
    for g in ("format", "citations", "grounding", "budget", "judge_error"):
        m[f"gate_{g}"] = 1.0 if result.gate_failed == g else 0.0
    for reason in ("answer", "max_turns", "budget", "overflow", "parse_error", "error"):   # every key present, so means are true rates
        m[f"stop_{reason}"] = 1.0 if trace.stats.stop_reason == reason else 0.0
    return m
