"""Turn a prose reference answer into atomic rubric items, once, at import time (Sonnet, cached).

Why here and not in the judge: a rubric that exists only inside a judge call is non-deterministic across evaluations,
cannot be reviewed or corrected, and is paid for on every eval of every checkpoint. Storing it in `grading.rubric`
makes every judged task use the same rubric-mode prompt with a fixed denominator.
"""
from __future__ import annotations

import asyncio
import re
import sys
from typing import Any

from codeqa.clients.anthropic import AnthropicClient
from codeqa.datagen import llm
from codeqa.shared.contracts import Task

MAX_ITEMS = 6
SYSTEM = """You turn a reference answer about a code repository into a short grading rubric.
Write the atomic facts a correct answer to the question MUST state, as a list of at most {n} items:
- each item is one independently checkable fact (one function, one file, one behaviour); no compound items, no "and";
- keep exact identifiers, file names, and literal values from the reference; drop narrative, motivation and restatements of the question;
- order by importance; if the reference contains fewer than {n} essential facts, write fewer.
Reply with JSON only: {{"facts": ["...", "..."]}}"""


def user_prompt(question: str, reference: str) -> str:
    ref = reference.strip()
    if len(ref) > 6000:
        ref = ref[:6000] + "\n[...]"
    return f"Question:\n{question}\n\nReference answer:\n{ref}"


def clean_items(facts: Any) -> list[str]:
    if not isinstance(facts, list):
        return []
    out: list[str] = []
    for f in facts:
        s = " ".join(str(f).split()).strip(" -*")
        if len(s) >= 8 and s not in out:
            out.append(s)
    return out[:MAX_ITEMS]


async def derive_one(client: AnthropicClient, cache: llm.JsonlCache, task: Task) -> list[str]:
    key = task.task_id
    hit = cache.get(key)
    if hit is not None:
        return hit["facts"]
    reply = await llm.ask(client, SYSTEM.format(n=MAX_ITEMS), user_prompt(task.question, task.grading.reference_answer or ""), max_tokens=700)
    obj = llm.parse_json_object(reply) or {}
    facts = clean_items(obj.get("facts"))
    if len(facts) < 2:                                   # one retry with a nudge; otherwise leave the rubric empty (judge falls back to reference mode)
        reply = await llm.ask(client, SYSTEM.format(n=MAX_ITEMS), user_prompt(task.question, task.grading.reference_answer or "") + "\n\nReply with only the JSON object.", max_tokens=700)
        facts = clean_items((llm.parse_json_object(reply) or {}).get("facts"))
    cache.put(key, {"facts": facts})
    return facts


async def derive_all(tasks: list[Task], cache_name: str, concurrency: int = 8) -> dict[str, list[str]]:
    todo = [t for t in tasks if t.grading.reference_answer and not t.grading.rubric]
    client = AnthropicClient(llm.SONNET.model_copy(update={"name": "rubric-writer", "max_generation_tokens": 700}))
    cache = llm.JsonlCache(cache_name)
    results = await llm.run_bounded(todo, lambda t: derive_one(client, cache, t), concurrency=concurrency, label="rubrics")
    out = {t.task_id: (r or []) for t, r in zip(todo, results)}
    n_ok = sum(1 for v in out.values() if len(v) >= 2)
    print(f"  rubrics: {n_ok}/{len(todo)} tasks got >= 2 items (cache {cache.path.name})", flush=True, file=sys.stderr)
    return out


def apply(tasks: list[Task], rubrics: dict[str, list[str]]) -> list[Task]:
    out = []
    for t in tasks:
        facts = rubrics.get(t.task_id)
        if facts and len(facts) >= 2:
            t = t.model_copy(update={"grading": t.grading.model_copy(update={"rubric": facts})})
        out.append(t)
    return out
