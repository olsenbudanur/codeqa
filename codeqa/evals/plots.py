"""Curves for the talk (gap_specs §7): reward, correctness, citation validity, tool calls per correct.

  uv run python -m codeqa.evals.plots --run run1                     # from data/logs/run1/metrics.jsonl -> data/logs/run1/plots/
  uv run python -m codeqa.evals.plots --run run1 --run run2          # overlay runs
  uv run python -m codeqa.evals.plots --set fast --profiles qwen4b-base claude qwen4b-run1-step40   # bars from data/evals/

Training curves read `env/all/<key>` (train) and `eval/fast/env/all/<key>` (held-out) per step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from codeqa.shared import paths

HEADLINE_KEYS = ("reward", "correctness", "citations_grounded", "tool_calls_per_correct")
EXTRA_KEYS = ("format_ok", "citations_parse", "prompt_tokens", "answer_tokens", "judge_error", "group_reward_std", "unique_tool_sequences_per_group")
OPTIM_KEYS = ("optim/loss", "optim/loss_abs", "optim/advantage_std", "optim/entropy", "optim/kl_sample_train_v1", "optim/clip_fraction", "optim/post_kl", "kl_ref/kl")
TITLES = {"optim/loss": "surrogate loss (near 0 by design)", "optim/loss_abs": "learning signal |ratio x adv|", "optim/advantage_std": "advantage std",
          "optim/clip_fraction": "clip fraction (|ratio-1| > 0.2)", "optim/entropy": "policy entropy", "optim/kl_sample_train_v1": "KL sampler vs trainer", "optim/post_kl": "KL after update",
          "kl_ref/kl": "KL to reference model", "reward": "reward", "correctness": "correctness", "citations_grounded": "citation validity (grounded)",
          "tool_calls_per_correct": "tool calls per correct answer", "format_ok": "format ok", "citations_parse": "citations parse",
          "prompt_tokens": "prompt tokens / episode", "answer_tokens": "answer tokens", "judge_error": "judge error rate",
          "group_reward_std": "group reward std", "unique_tool_sequences_per_group": "unique tool sequences / group"}


def read_metrics(run: str) -> list[dict[str, Any]]:
    p = paths.LOGS / run / "metrics.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in rows if isinstance(r, dict)]


def series(rows: list[dict[str, Any]], prefix: str, key: str) -> tuple[list[int], list[float]]:
    """(steps, values) for `<prefix>/<key>`. tool_calls_per_correct is derived: tool_calls / max(correct rate, eps)."""
    xs, ys = [], []
    for r in rows:
        step = r.get("step", r.get("progress/batch"))
        if step is None:
            continue
        if key.startswith(("optim/", "kl_ref/", "time/")):          # absolute keys, no prefix
            v = r.get(key)
            if v is None:
                continue
            xs.append(int(step)); ys.append(float(v)); continue
        if key == "tool_calls_per_correct":
            calls, rew = r.get(f"{prefix}/tool_calls"), r.get(f"{prefix}/reward")
            if calls is None or rew is None:
                continue
            v = calls / rew if rew and rew > 0 else float("nan")
        else:
            v = r.get(f"{prefix}/{key}")
            if v is None:
                continue
        xs.append(int(step))
        ys.append(float(v))
    return xs, ys


def plot_runs(runs: list[str], out_dir: Path, keys: tuple[str, ...] = HEADLINE_KEYS + EXTRA_KEYS) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    data = {run: read_metrics(run) for run in runs}
    written: list[Path] = []

    def draw(keys_: tuple[str, ...], name: str, ncols: int) -> None:
        nrows = (len(keys_) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
        for ax, key in zip(axes.flat, keys_):
            for run, rows in data.items():
                for prefix, style in ((("env/all", "-"), ("eval/fast/env/all", "--")) if not key.startswith(("optim/", "kl_ref/")) else (("", "-"),)):
                    xs, ys = series(rows, prefix, key)
                    if xs:
                        ax.plot(xs, ys, style, marker="o", ms=3, label=f"{run} {'train' if prefix == 'env/all' else 'eval'}")
            ax.set_title(TITLES.get(key, key), fontsize=10)
            ax.set_xlabel("step", fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
        for ax in list(axes.flat)[len(keys_):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    draw(HEADLINE_KEYS, "headline", 4)
    draw(keys, "all_curves", 4)
    if any(any(k in r for k in OPTIM_KEYS) for rows in data.values() for r in rows):
        draw(OPTIM_KEYS, "optimizer", 4)
    return written


def plot_eval_bars(set_name: str, profiles: list[str], out_dir: Path) -> Path | None:
    """Grouped bars across profiles for the headline metrics of one eval set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from codeqa.evals.report import load_results

    rows = [(p, load_results(p, set_name)) for p in profiles]
    rows = [(p, r["summary"]) for p, r in rows if r]
    if not rows:
        return None
    keys = ("reward", "correct_rate", "format_ok", "citation_valid", "tool_calls_per_correct")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2))
    for ax, key in zip(axes, keys):
        vals = [s.get(key, float("nan")) for _, s in rows]
        vals = [v if v == v and v != float("inf") else 0.0 for v in vals]
        ax.bar(range(len(rows)), vals, color="#4C72B0")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([p for p, _ in rows], rotation=30, ha="right", fontsize=8)
        ax.set_title(key.replace("_", " "), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"eval set: {set_name}", fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"eval_{set_name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="run name under data/logs (repeatable)")
    ap.add_argument("--set", default=None, help="eval set under data/evals/<profile>/")
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.run:
        out = Path(a.out) if a.out else paths.LOGS / a.run[0] / "plots"
        for p in plot_runs(a.run, out):
            print("wrote", p)
    if a.set:
        from codeqa.evals.report import profiles_with_set
        out = Path(a.out) if a.out else paths.EVALS / "plots"
        p = plot_eval_bars(a.set, a.profiles or profiles_with_set(a.set), out)
        print("wrote", p) if p else print("no results for set", a.set)
    return 0


if __name__ == "__main__":
    sys.exit(main())
