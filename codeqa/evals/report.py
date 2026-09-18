"""Tables from eval outputs: one profile's results.json, or several profiles side by side (C3).

  uv run python -m codeqa.evals.report --set smoke_sweqa_flask                  # every profile that has this set
  uv run python -m codeqa.evals.report --set fast --profiles claude qwen4b-base --markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from codeqa.shared import paths

HEADLINE = ("reward", "correct_rate", "format_ok", "citation_valid", "tool_calls_per_correct", "prompt_tokens_per_correct", "tool_calls", "n")
GROUP_COLS = ("n", "reward", "correct_rate", "format_ok", "citation_valid", "tool_calls")


def load_results(profile: str, set_name: str) -> dict[str, Any] | None:
    p = paths.EVALS / profile / set_name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def profiles_with_set(set_name: str) -> list[str]:
    if not paths.EVALS.exists():
        return []
    return sorted(d.name for d in paths.EVALS.iterdir() if (d / set_name / "results.json").exists())


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return "–"
        return f"{v:.0f}" if v >= 100 else f"{v:.2f}"
    return str(v)


def table(rows: list[dict[str, Any]], cols: tuple[str, ...], first_col: str, markdown: bool = False) -> str:
    """rows: [{first_col: label, **values}]. Plain aligned text or a markdown table."""
    head = [first_col, *cols]
    body = [[str(r.get(first_col, "")), *[_fmt(r.get(c, "")) for c in cols]] for r in rows]
    if markdown:
        lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        lines += ["| " + " | ".join(b) + " |" for b in body]
        return "\n".join(lines)
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h) for i, h in enumerate(head)]
    fmt = lambda cells: "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))  # noqa: E731
    return "\n".join([fmt(head), *(fmt(b) for b in body)])


def headline_table(set_name: str, profiles: list[str] | None = None, markdown: bool = False) -> str:
    profiles = profiles or profiles_with_set(set_name)
    rows = []
    for p in profiles:
        r = load_results(p, set_name)
        if r:
            rows.append({"profile": p, **r["summary"]})
    return table(rows, HEADLINE, "profile", markdown)


def group_table(profile: str, set_name: str, markdown: bool = False, kind: str = "source/type") -> str:
    """Rows are source x task_type (default), or `source:` / `type:` marginals."""
    r = load_results(profile, set_name)
    if not r:
        return f"(no results for {profile}/{set_name})"
    rows = []
    for key, s in r["by_group"].items():
        is_cross = "/" in key
        if (kind == "source/type" and is_cross) or (kind == "source" and key.startswith("source:")) or (kind == "type" and key.startswith("type:")):
            rows.append({"group": key.split(":", 1)[-1], **s})
    return table(rows, GROUP_COLS, "group", markdown)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    a = ap.parse_args(argv)
    profiles = a.profiles or profiles_with_set(a.set)
    parts = [f"## {a.set}: headline by profile", headline_table(a.set, profiles, a.markdown)]
    for p in profiles:
        parts += [f"\n## {p} / {a.set}: by source x task_type", group_table(p, a.set, a.markdown)]
    text = "\n".join(parts)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
