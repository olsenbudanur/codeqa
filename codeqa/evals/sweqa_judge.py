"""SWE-QA's five-dimension LLM judge (docs/research/swe_qa_llm_as_a_judge.py), for the external comparison number only.

Never used for training reward. Scores correctness, completeness, relevance, clarity, reasoning on 1-20 each (total 100),
strictly against the reference answer, exactly like the SWE-QA paper so our number is comparable to theirs.

  uv run python -m codeqa.evals.sweqa_judge --profile qwen4b-base --set sweqa --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from codeqa.clients.anthropic import AnthropicClient
from codeqa.shared import paths
from codeqa.shared.contracts import EndpointProfile, Message, Task
from codeqa.shared.jsonl import read_all

DIMENSIONS = ("correctness", "completeness", "relevance", "clarity", "reasoning")
DEFAULT_MODEL = "claude-sonnet-5"

PROMPT = """You are a STRICT and RIGOROUS evaluator. You must rate the candidate answer STRICTLY against the reference answer. Be CONSERVATIVE with high scores - only award high scores (16-20) when the candidate answer truly matches the reference answer's quality and completeness.

CRITICAL EVALUATION PRINCIPLES:
1. Compare the candidate answer DIRECTLY with the reference answer point by point
2. Any deviation, omission, or inaccuracy should result in score reduction
3. High scores (16-20) should be RARE - reserve them only for answers that are nearly perfect
4. Be strict about factual accuracy - even minor errors should lower the correctness score
5. Missing key points from the reference answer should significantly reduce completeness score
6. Vague or imprecise language should lower clarity scores
7. When in doubt between two score ranges, choose the LOWER one

Evaluation Criteria and Scoring Guidelines (each scored 1 to 20, total score 100):
        1. Correctness (STRICT - penalize any inaccuracies):
            20 — ONLY if completely correct with ALL core points and details accurate, matching reference answer precisely
            16-19 — Mostly correct but must have only TRIVIAL inaccuracies; any noticeable error reduces to 15 or below
            12-15 — Partially correct; has some errors or omissions that affect understanding; main points may be accurate but details are wrong
            8-11 — Several errors or ambiguities that significantly affect understanding of core information
            4-7 — Many errors; misleading or fails to convey key information correctly
            1-3 — Serious errors; completely wrong or misleading
        2. Completeness (STRICT - penalize missing information):
            20 — ONLY if covers ALL key points from reference answer without ANY omission; must match reference in depth
            16-19 — Covers most key points but missing some non-trivial information; minor omissions are acceptable
            12-15 — Missing several important key points; content is noticeably incomplete compared to reference
            8-11 — Important information largely missing; content is one-sided or superficial
            4-7 — Covers very little relevant information; seriously incomplete
            1-3 — Covers almost no relevant information; completely incomplete
        3. Relevance (STRICT - penalize off-topic content):
            20 — ONLY if content is fully focused on question topic with NO irrelevant information whatsoever
            16-19 — Mostly focused but may have minor peripheral information; any significant off-topic content reduces score
            12-15 — Generally on topic but contains some off-topic content that detracts from answer
            8-11 — Topic not sufficiently focused; contains considerable off-topic or tangential content
            4-7 — Content deviates from topic; includes excessive irrelevant information
            1-3 — Majority of content irrelevant to the question
        4. Clarity (STRICT - penalize unclear expression):
            20 — ONLY if language is exceptionally fluent, clear, and precise; very easy to understand without any ambiguity
            16-19 — Mostly fluent and clear but may have minor unclear points; any significant ambiguity reduces score
            12-15 — Generally clear but some expressions are unclear or not concise; may require effort to understand
            8-11 — Expression somewhat awkward; has ambiguity or lacks fluency that hinders understanding
            4-7 — Language obscure; sentences are not smooth; significantly hinders understanding
            1-3 — Expression confusing; very difficult to understand
        5. Reasoning (STRICT - penalize weak logic):
            20 — ONLY if reasoning is exceptionally clear, logical, and well-structured; argumentation is excellent and matches reference quality
            16-19 — Reasoning is clear and logical with solid argumentation; minor logical gaps may exist
            12-15 — Reasoning generally reasonable but has noticeable logical jumps or organization issues
            8-11 — Reasoning is average; has logical jumps or organization problems that affect understanding
            4-7 — Reasoning unclear; lacks logical order; difficult to follow
            1-3 — No clear reasoning; logic is chaotic

INPUT:
    Question:{question}
    Reference Answer:{reference}
    Candidate Answer:{candidate}

OUTPUT:
    Please output ONLY a JSON object with 5 integer fields in the range [1,20], corresponding
    to the evaluation scores:
        {{
        "correctness": <1-20>,
        "completeness": <1-20>,
        "relevance": <1-20>,
        "clarity": <1-20>,
        "reasoning": <1-20>
        }}

SCORING INSTRUCTIONS:
- Read the reference answer carefully and identify ALL key points, details, and structure
- Compare the candidate answer systematically against the reference answer
- For each criterion, start with a conservative score and only increase if the candidate truly deserves it
- If the candidate answer is significantly shorter, less detailed, or less precise than the reference, reduce scores accordingly
- If the candidate answer contains information not in the reference (unless it's clearly relevant and accurate), consider reducing relevance score
- When scoring, ask yourself: "Does this candidate answer match the quality and completeness of the reference answer?" If not, reduce scores
- Average or mediocre answers should receive scores in the 8-15 range, not higher
- Only truly excellent answers that closely match the reference should receive 16-20 scores

REQUIREMENT:
    No explanation, no extra text, no formatting other than valid JSON. Be strict and conservative with your scores."""


def parse_scores(text: str) -> dict[str, int]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON in judge output")
    data = json.loads(m.group(0))
    out = {}
    for d in DIMENSIONS:
        v = int(data[d])
        if not 1 <= v <= 20:
            raise ValueError(f"{d}={v} out of range")
        out[d] = v
    out["total"] = sum(out[d] for d in DIMENSIONS)
    return out


async def score(question: str, reference: str, candidate: str, client: AnthropicClient, retries: int = 3) -> dict[str, int] | None:
    msg = Message(role="user", content=PROMPT.format(question=question, reference=reference, candidate=candidate or "(no answer)"))
    for attempt in range(retries):
        try:
            reply = await asyncio.wait_for(client.chat([msg], max_tokens=200), timeout=60)
            return parse_scores(reply.content)
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  judge failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                return None
            await asyncio.sleep(2 ** attempt)
    return None


async def run(profile: str, set_name: str, model: str, concurrency: int, max_tasks: int | None) -> dict[str, Any]:
    ev_dir = paths.EVALS / profile / set_name
    rows = [json.loads(l) for l in (ev_dir / "per_task.jsonl").read_text().splitlines() if l.strip()]
    results = json.loads((ev_dir / "results.json").read_text())
    tasks = {t.task_id: t for t in read_all(Path(results["tasks_file"]), Task)}
    rows = [r for r in rows if tasks.get(r["task_id"]) and tasks[r["task_id"]].grading.reference_answer][:max_tasks]
    client = AnthropicClient(EndpointProfile(name="sweqa-judge", kind="anthropic", model=model, max_generation_tokens=200, thinking=False))
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        t = tasks[r["task_id"]]
        async with sem:
            s = await score(t.question, t.grading.reference_answer or "", r.get("answer", ""), client)
        print(f"  {r['task_id']:26s} {'total=%d' % s['total'] if s else 'FAILED'}", flush=True)
        return {"task_id": r["task_id"], "scores": s}

    print(f"sweqa judge: profile={profile} set={set_name} model={model} n={len(rows)}", flush=True)
    scored = await asyncio.gather(*(one(r) for r in rows))
    ok = [s["scores"] for s in scored if s["scores"]]
    summary = {"n": len(rows), "scored": len(ok), "model": model}
    for d in (*DIMENSIONS, "total"):
        summary[d] = sum(s[d] for s in ok) / len(ok) if ok else float("nan")
    (ev_dir / "sweqa_judge.json").write_text(json.dumps({"summary": summary, "per_task": scored}, indent=2))
    results.setdefault("summary", {})["sweqa_total"] = summary["total"]
    results["summary"]["sweqa_correctness"] = summary["correctness"]
    (ev_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"done: total={summary['total']:.1f}/100 " + " ".join(f"{d}={summary[d]:.1f}" for d in DIMENSIONS), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=None)
    a = ap.parse_args(argv)
    asyncio.run(run(a.profile, a.set, a.model, a.concurrency, a.max_tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
