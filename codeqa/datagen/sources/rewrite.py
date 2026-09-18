"""Haiku turns a CodeScout issue into a where/which question that does not leak the answer location.

Cached in data/cache/rewrites/codescout.jsonl by instance_id. A rewrite is rejected when it mentions any gold path,
basename, module stem, or symbol name; rejected rows are dropped from the derived set.
"""
from __future__ import annotations

import re
from typing import Any

from codeqa.clients.anthropic import AnthropicClient
from codeqa.datagen import llm

SYSTEM = """You write questions that a developer new to a codebase would ask a code-search assistant.
You are given a GitHub issue. Write ONE question asking where in the repository the behavior the issue is about is implemented,
or which function or class is responsible for it. Rules:
- Describe the behavior in plain words. Do not mention file names, paths, module names, or function/class names from the issue.
- Do not mention that this is a bug or an issue, and do not ask how to fix it. Ask where the relevant code lives.
- One sentence, under 40 words, starting with "Where" or "Which".
Reply with JSON only: {"question": "..."}"""


def forbidden_terms(paths: list[str], symbols: list[str]) -> set[str]:
    """Things a rewrite must not mention: gold paths, their stems, and every gold identifier (Class, method, function)."""
    terms: set[str] = set()
    for p in paths:
        stem = p.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        terms.add(p); terms.add(stem)
    for s in symbols:
        tail = s.split(":", 1)[-1]
        for part in tail.split("."):
            if len(part) > 2:
                terms.add(part)
    terms -= {"init", "main", "utils", "util", "core", "base", "cli", "api", "app", "test", "tests", "self", "config", "run", "get", "set", "__init__"}
    return terms


def _is_multiword(ident: str) -> bool:
    return "_" in ident.strip("_") or bool(re.search(r"[a-z][A-Z]", ident))


def leaks(question: str, terms: set[str]) -> str | None:
    """A leak is: a gold path (or path suffix), a multi-word identifier written as such (`snake_case`, `CamelCase`),
    or a single-word identifier written as code (`name`, name(), .name). A plain domain word ('snowflake', 'image')
    that happens to be a module stem is allowed: real users say it, and the model still has to find the symbol."""
    q = question
    ql = q.lower()
    for t in sorted(terms, key=len, reverse=True):
        if "/" in t:
            if t.lower() in ql or t.rsplit("/", 1)[-1].lower() in ql:
                return t
        elif _is_multiword(t):
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(t) + r"(?![A-Za-z0-9_])", q, re.I):
                return t
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(t.replace("_", " ")) + r"(?![A-Za-z0-9_])", q, re.I) and t.count("_") >= 2:
                return t
        else:
            if re.search(r"`" + re.escape(t) + r"`|(?<![A-Za-z0-9_])" + re.escape(t) + r"\(\)|\." + re.escape(t) + r"(?![A-Za-z0-9_])", q):
                return t
    return None


def user_prompt(problem_statement: str) -> str:
    text = problem_statement.strip()
    if len(text) > 2500:
        text = text[:2500] + "\n[...]"
    return f"GitHub issue:\n\n{text}"


async def rewrite_one(client: AnthropicClient, cache: llm.JsonlCache, row: dict[str, Any], paths: list[str], symbols: list[str]) -> dict[str, Any]:
    """-> {"question": str | None, "reason": str}. Cached by instance_id."""
    key = row["instance_id"]
    hit = cache.get(key)
    if hit is not None:
        if hit["reason"].startswith("leak") and hit.get("raw"):      # leak rules changed: re-judge the cached text
            leak = leaks(hit["raw"], forbidden_terms(paths, symbols))
            if not leak:
                hit = {"question": hit["raw"], "reason": "ok"}
                cache.put(key, hit)
        return hit
    reply = await llm.ask(client, SYSTEM, user_prompt(row["problem_statement"]))
    obj = llm.parse_json_object(reply)
    q = (obj or {}).get("question")
    if not isinstance(q, str) or not q.strip():
        result = {"question": None, "reason": "unparseable", "raw": reply[:300]}
    else:
        q = " ".join(q.split())
        leak = leaks(q, forbidden_terms(paths, symbols))
        if leak:
            result = {"question": None, "reason": f"leak:{leak}", "raw": q}
        elif not re.match(r"^(where|which|what|how|in which)\b", q, re.I):
            result = {"question": None, "reason": "not_a_question", "raw": q}
        else:
            result = {"question": q, "reason": "ok"}
    cache.put(key, result)
    return result
