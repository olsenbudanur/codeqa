"""B4 runner: structural tasks for every training repo -> data/tasks/raw/structural.jsonl (+ report).

Training repos = data/repo_list_dcb.txt + data/repo_list_codescout.txt (written by B3). For each: snapshot + index
(exists), use `<repo_id>__nodoc` when lane A has built it, generate, paraphrase locate docstrings with Haiku (cached,
leak-checked), cap 60 per repo balanced across the four types.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter
from typing import Any

from codeqa.clients.anthropic import AnthropicClient
from codeqa.datagen import llm
from codeqa.datagen.repos import ensure_repo
from codeqa.datagen.sources import structural
from codeqa.shared import paths
from codeqa.shared.contracts import Task
from codeqa.shared.jsonl import write

REPORTS = paths.TASKS / "reports"
PARAPHRASE_SYSTEM = """You turn the docstring of a Python function, method, or class into the question a developer would type into a
code-search assistant to find that code, without knowing its name.
Rules:
- Describe what the code does or is for, in plain words, from the docstring only.
- Never use the name of the function/method/class, its module, or any word derived from its name. A list of banned words is given.
- One sentence, under 35 words, starting with "Where" or "Which".
Reply with JSON only: {"question": "..."}"""


def _log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def read_repo_list(name: str) -> list[tuple[str, str, str]]:
    p = paths.DATA / name
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        repo, _, sha = line.partition("@")
        owner, _, rname = repo.partition("/")
        out.append((owner, rname, sha))
    return out


def training_repos() -> list[tuple[str, str, str]]:
    seen: set[str] = set()
    out = []
    for owner, repo, sha in read_repo_list("repo_list_dcb.txt") + read_repo_list("repo_list_codescout.txt"):
        k = f"{owner}/{repo}".lower()
        if k not in seen:
            seen.add(k); out.append((owner, repo, sha))
    return out


async def paraphrase_locate(client: AnthropicClient, cache: llm.JsonlCache, task: Task) -> dict[str, Any]:
    key = task.task_id
    hit = cache.get(key)
    if hit is not None:
        return hit
    path, qual = task.grading.expected_symbols[0].split(":", 1)
    kind = "method" if "." in qual else "function or class"
    stem = path.rsplit("/", 1)[-1].removesuffix(".py")
    banned = structural.name_tokens(qual, stem)
    user = (f"Kind: {kind}\nDocstring:\n{task.grading.reference_answer}\n\n"
            f"Banned words (do not use any of these or their variants): {', '.join(sorted(banned))}")
    reply = await llm.ask(client, PARAPHRASE_SYSTEM, user)
    obj = llm.parse_json_object(reply)
    q = (obj or {}).get("question")
    if not isinstance(q, str) or not q.strip():
        res = {"question": None, "reason": "unparseable"}
    else:
        q = " ".join(q.split())
        leak = structural.contains_leak(q, banned)
        res = {"question": None, "reason": f"leak:{leak}", "raw": q} if leak else {"question": q, "reason": "ok"}
    cache.put(key, res)
    return res


async def generate_all(per_type: int = 15, cap: int = 60, concurrency: int = 8, only: list[str] | None = None) -> dict[str, Any]:
    t0 = time.time()
    repos = training_repos()
    if only:
        repos = [r for r in repos if r[1] in only or f"{r[0]}/{r[1]}" in only]
    client = AnthropicClient(llm.HAIKU)
    cache = llm.JsonlCache("structural_locate")
    all_tasks: list[Task] = []
    per_repo: dict[str, Any] = {}
    for owner, repo, sha in repos:
        m, syms = ensure_repo(owner, repo, sha)
        nodoc_id = m.repo_id + "__nodoc"
        has_nodoc = (paths.index_dir(nodoc_id) / "symbols.json").exists()
        task_repo = nodoc_id if has_nodoc else m.repo_id
        tasks, stats = structural.generate(m, syms, task_repo, per_type=per_type, cap=cap)
        locate = [t for t in tasks if t.task_type == "locate"]
        others = [t for t in tasks if t.task_type != "locate"]
        results = await llm.run_bounded(locate, lambda t: paraphrase_locate(client, cache, t), concurrency=concurrency, label=f"paraphrase {repo}")
        kept_locate: list[Task] = []
        reasons: Counter = Counter()
        for t, r in zip(locate, results):
            if r is None:
                reasons["call_failed"] += 1; continue
            reasons[r["reason"].split(":")[0]] += 1
            if r.get("question"):
                kept_locate.append(t.model_copy(update={"question": r["question"]}))
        # balance: per_type locate, then fill the remaining cap from any surplus
        chosen = kept_locate[:per_type] + others
        surplus = kept_locate[per_type:]
        while len(chosen) < cap and surplus:
            chosen.append(surplus.pop(0))
        all_tasks.extend(chosen)
        per_repo[m.repo_id] = {"task_repo_id": task_repo, "nodoc": has_nodoc, "candidates": stats["candidates"],
                               "kept": dict(Counter(t.task_type for t in chosen)), "paraphrase": dict(reasons)}
        _log(f"  structural {m.repo_id}{' (nodoc)' if has_nodoc else ' (NO nodoc yet)'}: {len(chosen)} tasks {per_repo[m.repo_id]['kept']}; paraphrase {dict(reasons)}")
    out = paths.TASKS_RAW / "structural.jsonl"
    n = write(out, all_tasks)
    report = {"source": "structural", "records": n, "file": str(out.relative_to(paths.ROOT)),
              "types": dict(Counter(t.task_type for t in all_tasks)), "repos": per_repo,
              "repos_without_nodoc": [k for k, v in per_repo.items() if not v["nodoc"]], "seconds": round(time.time() - t0, 1)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "structural.json").write_text(json.dumps(report, indent=2))
    _log(f"  structural: {n} tasks over {len(per_repo)} repos; types {report['types']}; {report['seconds']}s")
    return report


def main(per_type: int = 15, cap: int = 60, concurrency: int = 8, only: list[str] | None = None) -> dict[str, Any]:
    return asyncio.run(generate_all(per_type, cap, concurrency, only))
