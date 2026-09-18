"""Run monitor: training curve + collapse checks from data/logs/<run>/metrics.jsonl, and the plots.

  uv run python -m codeqa.evals.monitor --run run1            # table, warnings, writes data/logs/run1/plots/*.png
  uv run python -m codeqa.evals.monitor --run run1 --no-plots

Collapse checks (from the smoke runs): group reward std -> 0, unique tool sequences per group -> 1, stop-by-budget or
max-turns rate climbing, format gate spiking, reward falling while tool calls rise, judge error rate.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from codeqa.evals.plots import plot_runs, read_metrics
from codeqa.shared import paths

TRAIN_COLS = ("reward", "reward_shaped", "correctness", "format_ok", "citations_grounded", "tool_calls", "answer_tokens",
              "group_reward_std", "unique_tool_sequences_per_group", "stop_budget", "stop_max_turns", "gate_format",
              "gate_citations", "gate_grounding", "judge_error", "no_answer_penalty")
EVAL_COLS = ("reward", "correctness", "format_ok", "citations_grounded", "tool_calls", "stop_budget", "stop_max_turns")
OPTIM_COLS = ("optim/lr", "optim/entropy", "optim/kl_sample_train_v1", "optim/kl_sample_train_v2", "optim/post_kl", "kl_ref/kl", "time/total")
SHORT = {"reward": "rew", "reward_shaped": "shaped", "correctness": "corr", "format_ok": "fmt", "citations_grounded": "grnd",
         "tool_calls": "calls", "answer_tokens": "ans_tok", "group_reward_std": "std", "unique_tool_sequences_per_group": "uniq",
         "stop_budget": "s_bud", "stop_max_turns": "s_turn", "gate_format": "g_fmt", "gate_citations": "g_cit",
         "gate_grounding": "g_grd", "judge_error": "j_err", "no_answer_penalty": "pen"}


def _v(row: dict[str, Any], prefix: str, key: str) -> float | None:
    return row.get(f"{prefix}/{key}")


def _fmt(v: float | None) -> str:
    if v is None:
        return "   -"
    return f"{v:6.0f}" if abs(v) >= 100 else f"{v:6.2f}"


def optim_table(rows: list[dict[str, Any]]) -> str:
    """Optimizer-side signals the cookbook logs per step: lr, policy entropy, sampler-vs-trainer KL (how far the policy moved
    from the weights that produced the batch), post-update KL (only with compute_post_kl), KL to the reference model (only
    with a KL penalty), and wall time. Tinker's hosted forward/backward does not report gradient norms."""
    present = [c for c in OPTIM_COLS if any(c in r for r in rows)]
    if not present:
        return "(no optim/* keys yet)"
    lines = ["step " + " ".join(f"{c.split('/')[-1][:12]:>12s}" for c in present)]
    for r in rows:
        if "optim/lr" not in r:
            continue
        cells = []
        for c in present:
            v = r.get(c)
            cells.append(f"{'-':>12s}" if v is None else (f"{v:12.2e}" if (abs(v) < 1e-2 and v != 0) else f"{v:12.3f}"))
        lines.append(f"{int(r.get('step', -1)):4d} " + " ".join(cells))
    return "\n".join(lines)


def table(rows: list[dict[str, Any]], prefix: str, cols: tuple[str, ...]) -> str:
    lines = ["step " + " ".join(f"{SHORT.get(c, c)[:7]:>7s}" for c in cols)]
    for r in rows:
        if _v(r, prefix, "reward") is None:
            continue
        lines.append(f"{int(r.get('step', -1)):4d} " + " ".join(f"{_fmt(_v(r, prefix, c)):>7s}" for c in cols))
    return "\n".join(lines)


def checks(rows: list[dict[str, Any]], prefix: str = "env/all") -> list[str]:
    """Human-readable warnings. Empty list = nothing alarming."""
    rs = [r for r in rows if _v(r, prefix, "reward") is not None]
    if len(rs) < 2:
        return []
    out: list[str] = []
    last, prev, first = rs[-1], rs[-2], rs[0]
    step = int(last.get("step", -1))

    def g(r, k, d=0.0):
        v = _v(r, prefix, k)
        return d if v is None else v

    if g(last, "group_reward_std") < 0.05:
        out.append(f"step {step}: group_reward_std={g(last, 'group_reward_std'):.3f} (<0.05): almost no groups carry a gradient")
    if g(last, "unique_tool_sequences_per_group", 9) <= 1.5:
        out.append(f"step {step}: unique_tool_sequences_per_group={g(last, 'unique_tool_sequences_per_group'):.2f}: rollouts have collapsed to one behaviour")
    stall = g(last, "stop_budget") + g(last, "stop_max_turns")
    stall0 = g(first, "stop_budget") + g(first, "stop_max_turns")
    if stall > 0.5 or (stall > 0.25 and stall > stall0 + 0.15):
        out.append(f"step {step}: {stall:.0%} of episodes end without answering (budget+max_turns; step 0 was {stall0:.0%})")
    if g(last, "gate_format") > 0.7 and g(last, "gate_format") > g(first, "gate_format") + 0.2:
        out.append(f"step {step}: format gate fails {g(last, 'gate_format'):.0%} (step 0: {g(first, 'gate_format'):.0%})")
    if g(last, "reward") < g(prev, "reward") - 0.1 and g(last, "tool_calls") > g(prev, "tool_calls") + 1:
        out.append(f"step {step}: reward fell {g(prev, 'reward'):.2f}->{g(last, 'reward'):.2f} while tool calls rose {g(prev, 'tool_calls'):.1f}->{g(last, 'tool_calls'):.1f}")
    kl = last.get(f"optim/kl_sample_train_v1")
    if kl is not None and abs(kl) > 0.05:
        out.append(f"step {step}: sampler-vs-trainer KL {kl:.3f} (>0.05): the update moved the policy a lot; consider a lower lr")
    ent, ent0 = last.get("optim/entropy"), first.get("optim/entropy")
    if ent is not None and ent0 and ent < 0.4 * ent0:
        out.append(f"step {step}: entropy {ent:.3f} is <40 % of step 0 ({ent0:.3f}): policy is sharpening fast (collapse risk)")
    if g(last, "judge_error") > 0.1:
        out.append(f"step {step}: judge_error={g(last, 'judge_error'):.0%}: those samples get the group mean (no signal)")
    if len(rs) >= 4:
        recent = [g(r, "reward") for r in rs[-4:]]
        if max(recent) - min(recent) < 0.01 and all(g(r, "group_reward_std") < 0.05 for r in rs[-4:]):
            out.append(f"steps {int(rs[-4]['step'])}-{step}: reward flat at {recent[-1]:.2f} with zero group variance: dead run")
    return out


def report(run: str, plots: bool = True) -> str:
    rows = read_metrics(run)
    if not rows:
        return f"no metrics.jsonl for run {run!r} under {paths.LOGS}"
    parts = [f"run {run}: {len(rows)} steps logged", "", "train (env/all)", table(rows, "env/all", TRAIN_COLS)]
    if any(_v(r, "eval/fast/env/all", "reward") is not None for r in rows):
        parts += ["", "held-out (eval/fast)", table(rows, "eval/fast/env/all", EVAL_COLS)]
    parts += ["", "optimizer (per step)", optim_table(rows)]
    warns = checks(rows)
    parts += ["", "checks: " + ("OK" if not warns else "")] + [f"  ! {w}" for w in warns]
    if plots:
        for p in plot_runs([run], paths.LOGS / run / "plots"):
            parts.append(f"wrote {p}")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    print(report(a.run, plots=not a.no_plots))
    return 0


if __name__ == "__main__":
    sys.exit(main())
