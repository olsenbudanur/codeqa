"""Code-QA-Bench's LLM judge, ported verbatim from Lens-Frontier/code-qa-bench `code_qa_bench/judge.py` (MIT), for the
external comparison number only. Never used for training reward.

Three anchored 0-5 axes (accuracy, completeness, specificity), think-then-score, gold/agent order randomized per sample,
median over N samples; per-question score = (accuracy + completeness + specificity) / 15; overall = mean. Their default
judge was claude-sonnet-4-20250514; we default to claude-sonnet-5, so quote the judge model with the number.

  uv run python -m codeqa.evals.cqb_judge --profile qwen4b-base --set codeqabench --samples 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from codeqa.clients.anthropic import AnthropicClient
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Message, Task
from codeqa.shared.jsonl import read_all

AXES = ("accuracy", "completeness", "specificity")
DEFAULT_MODEL = "claude-sonnet-5"

# --- prompt: identical to upstream `_JUDGE_PROMPT_TEMPLATE` ---------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """\
You are an expert judge evaluating answers about a code repository.

## Question
{question}

## Answer A
{answer_a}

## Answer B
{answer_b}

## Key Points the Answer Should Cover (Rubric)
{rubric}

{position_instruction}

## Step 1: Analyze

Before scoring, reason through each axis carefully:

- **Accuracy analysis**: Compare {target_label}'s factual claims against the reference. \
List any incorrect statements about the code (wrong file paths, wrong function behavior, \
wrong class relationships).
- **Completeness analysis**: Go through each rubric point and note whether {target_label} \
covers it, partially covers it, or misses it entirely.
- **Specificity analysis**: List the specific code references {target_label} makes \
(file paths, function/class names, code patterns). Note any vague or generic statements.

## Step 2: Score

After your analysis, score {target_label} on three axes using these anchored scales:

### Accuracy (0-5)
- **5**: All factual claims are correct; no errors about code behavior or structure
- **4**: Minor inaccuracy that does not affect overall understanding
- **3**: Partially correct; some claims are wrong but core understanding is right
- **2**: Significant errors that undermine the answer's reliability
- **1**: Mostly wrong; fundamental misunderstanding of the code
- **0**: Completely incorrect or fabricated

### Completeness (0-5)
- **5**: All rubric points are thoroughly covered
- **4**: Most rubric points covered; one minor point missing
- **3**: About half of the rubric points covered
- **2**: Only a few rubric points addressed
- **1**: Barely touches on the topic; most points missing
- **0**: Does not address any rubric point

### Specificity (0-5)
- **5**: References specific files, functions, classes, and code patterns throughout
- **4**: Most claims cite specific code locations; one or two vague statements
- **3**: Mix of specific and generic statements
- **2**: Mostly generic with a few specific references
- **1**: Almost entirely vague; one or two names mentioned without context
- **0**: Entirely vague; no specific code references at all

## Output Format

First write your analysis, then output a JSON block:

```json
{{
  "accuracy": <0-5>,
  "completeness": <0-5>,
  "specificity": <0-5>,
  "explanation": "<brief justification summarizing your analysis>"
}}
```
"""
POSITION_GOLD_FIRST = ("Answer A is the reference answer. "
                       "Your job is to evaluate **Answer B** (the agent's answer) against Answer A and the rubric.")
POSITION_AGENT_FIRST = ("Answer B is the reference answer. "
                        "Your job is to evaluate **Answer A** (the agent's answer) against Answer B and the rubric.")


def build_prompt(question: str, gold_answer: str, agent_answer: str, rubric: list[str], gold_first: bool = True) -> str:
    rubric_text = "\n".join(f"- {p}" for p in rubric) if rubric else "(no rubric)"
    a, b = (gold_answer, agent_answer) if gold_first else (agent_answer, gold_answer)
    return JUDGE_PROMPT_TEMPLATE.format(question=question, answer_a=a, answer_b=b, rubric=rubric_text,
                                        position_instruction=POSITION_GOLD_FIRST if gold_first else POSITION_AGENT_FIRST,
                                        target_label="Answer B" if gold_first else "Answer A")


# --- parsing: identical strategies to upstream ---------------------------------------------------------------------
def _sanitize(s: str) -> str:
    s = s.replace("\\'", "'")
    return re.sub(r",\s*([}\]])", r"\1", s)


def _loads(s: str):
    for text in (s, _sanitize(s)):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def parse_response(text: str) -> dict[str, Any]:
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        r = _loads(m.group(1).strip())
        if r is not None:
            return r
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        r = _loads(m.group(0))
        if r is not None and "accuracy" in r:
            return r
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        r = _loads(text[i:j + 1])
        if r is not None:
            return r
    raise ValueError(f"Could not extract JSON scores from judge response ({len(text)} chars)")


def aggregate(samples: list[dict[str, float]]) -> dict[str, float]:
    """Median per axis over samples (upstream `_aggregate_samples`); score = sum/15."""
    agg = {ax: statistics.median([s[ax] for s in samples]) for ax in AXES}
    agg["score"] = sum(agg[ax] for ax in AXES) / 15.0
    return agg


async def judge_once(client: AnthropicClient, question: str, gold: str, answer: str, rubric: list[str], gold_first: bool) -> dict[str, float]:
    reply = await asyncio.wait_for(client.chat([Message(role="user", content=build_prompt(question, gold, answer or "(no answer)", rubric, gold_first))],
                                               max_tokens=2048), timeout=90)
    data = parse_response(reply.content)
    out = {ax: float(data[ax]) for ax in AXES}
    for ax, v in out.items():
        if not 0 <= v <= 5:
            raise ValueError(f"{ax}={v} out of range")
    out["position"] = "gold_first" if gold_first else "agent_first"
    out["explanation"] = str(data.get("explanation", ""))[:300]
    return out


async def judge(client: AnthropicClient, task: Task, answer: str, num_samples: int = 1, retries: int = 3,
                rng: random.Random | None = None) -> dict[str, Any] | None:
    rng = rng or random
    samples: list[dict[str, float]] = []
    for _ in range(max(1, num_samples)):
        gold_first = rng.choice([True, False])          # position randomization per sample, as upstream
        for attempt in range(retries):
            try:
                samples.append(await judge_once(client, task.question, task.grading.reference_answer or "", answer, task.grading.rubric, gold_first))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    print(f"  judge failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    return None
                await asyncio.sleep(2 ** attempt)
    agg = aggregate([{ax: s[ax] for ax in AXES} for s in samples])
    agg["samples"] = samples
    return agg


async def run(profile: str, set_name: str, model: str, concurrency: int, max_tasks: int | None, num_samples: int) -> dict[str, Any]:
    ev_dir = paths.EVALS / profile / set_name
    rows = [json.loads(l) for l in (ev_dir / "per_task.jsonl").read_text().splitlines() if l.strip()]
    results = json.loads((ev_dir / "results.json").read_text())
    tasks = {t.task_id: t for t in read_all(Path(results["tasks_file"]), Task)}
    rows = [r for r in rows if tasks.get(r["task_id"]) and tasks[r["task_id"]].grading.reference_answer][:max_tasks]
    client = AnthropicClient(EndpointProfile(name="cqb-judge", kind="anthropic", model=model, max_generation_tokens=2048, thinking=False))
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        async with sem:
            s = await judge(client, tasks[r["task_id"]], r.get("answer", ""), num_samples)
        print(f"  {r['task_id']:28s} " + (f"score={s['score']:.2f} " + " ".join(f"{ax[:3]}={s[ax]:.0f}" for ax in AXES) if s else "FAILED"), flush=True)
        return {"task_id": r["task_id"], "scores": s}

    print(f"code-qa-bench judge: profile={profile} set={set_name} model={model} samples={num_samples} n={len(rows)}", flush=True)
    scored = await asyncio.gather(*(one(r) for r in rows))
    ok = [s["scores"] for s in scored if s["scores"]]
    summary: dict[str, Any] = {"n": len(rows), "scored": len(ok), "model": model, "samples": num_samples}
    for k in (*AXES, "score"):
        summary[k] = sum(s[k] for s in ok) / len(ok) if ok else float("nan")
    (ev_dir / "cqb_judge.json").write_text(json.dumps({"summary": summary, "per_task": scored}, indent=2))
    results.setdefault("summary", {})["cqb_score"] = summary["score"]
    (ev_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"done: score={summary['score']:.3f} " + " ".join(f"{ax}={summary[ax]:.2f}" for ax in AXES), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--set", default="codeqabench")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--samples", type=int, default=1, help="judge calls per answer, median per axis (upstream judge_samples)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=None)
    a = ap.parse_args(argv)
    asyncio.run(run(a.profile, a.set, a.model, a.concurrency, a.max_tasks, a.samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
