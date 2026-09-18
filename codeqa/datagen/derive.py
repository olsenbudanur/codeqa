"""B3: CodeScout (OpenHands/SWE-rebench-code-search) -> data/tasks/raw/codescout.jsonl.

Steps: pick the top-N repos by row count (excluding every SWE-QA-Bench repo and the 8 DeepCodeBench repos), snapshot
each at one commit (the base_commit of its newest instance), keep rows whose gold files exist and whose gold entities
resolve in that snapshot's index, rewrite the issue into a where/which question with Haiku, drop rewrites that leak.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

from codeqa.clients.anthropic import AnthropicClient
from codeqa.datagen import cache, llm
from codeqa.datagen.repos import ensure_repo
from codeqa.datagen.resolve import RepoIndex, symbol_span
from codeqa.datagen.sources import derive_codescout as cs
from codeqa.datagen.sources import import_deepcodebench as dcb
from codeqa.datagen.sources import import_sweqa as sq
from codeqa.datagen.sources import rewrite
from codeqa.shared import paths
from codeqa.shared.contracts import Span, Task, make_repo_id
from codeqa.shared.jsonl import write

REPORTS = paths.TASKS / "reports"


def _log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def excluded_repos() -> set[str]:
    """owner/repo, lower-cased: all 15 SWE-QA-Bench repos (never train on them) + the 8 DeepCodeBench repos (already used)."""
    ex = {f"{o}/{r}".lower() for o, r in sq.REPOS.values()}
    for split in ("train", "test"):
        for row in cache.load_cached(dcb.DATASET, "default", split):
            o, r, _ = dcb.repo_of(row)
            ex.add(f"{o}/{r}".lower())
    return ex


def instance_number(instance_id: str) -> int:
    m = re.search(r"-(\d+)$", instance_id)
    return int(m.group(1)) if m else -1


def select_repos(rows: list[dict[str, Any]], n: int, exclude: set[str]) -> list[tuple[str, int, str]]:
    """-> [(owner/repo, row_count, chosen_sha)] top-n by rows with entity gold; sha = base_commit of the newest instance."""
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["repo"].lower() in exclude:
            continue
        paths_, symbols = cs.gold(r)
        if paths_ and symbols and len(paths_) <= 3:
            by_repo[r["repo"]].append(r)
    ranked = sorted(by_repo.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:n]
    out = []
    for repo, rs in ranked:
        newest = max(rs, key=lambda r: instance_number(r["instance_id"]))
        out.append((repo, len(rs), newest["base_commit"]))
    return out


def resolve_gold(idx: RepoIndex, paths_: list[str], symbols: list[str]) -> tuple[list[str], list[str], list[Span]] | None:
    """Gold at the chosen commit: every gold file must exist; at least one entity per file must resolve to an index symbol.
    Returns (paths, symbols 'path:Qualified', evidence spans) or None."""
    kept_paths: list[str] = []
    kept_syms: list[str] = []
    spans: list[Span] = []
    for p in paths_:
        rp = idx.resolve_path(p)
        if rp is None:
            return None
        ents = [s for s in symbols if s.split(":", 1)[0] == p]
        # 'Parser' next to 'Parser._parse_x' means the method was edited; keep the specific entity, drop the bare class
        quals = {e.split(":", 1)[1] for e in ents}
        ents = [e for e in ents if not any(q.startswith(e.split(":", 1)[1] + ".") for q in quals)]
        found_here = False
        for ent in ents:
            qual = ent.split(":", 1)[1]
            parts = qual.split(".")
            # strict: 'Cls.meth' needs parent+name in this file; 'fn' needs a top-level name in this file
            hits = [h for h in idx.by_name.get(parts[-1], [])
                    if h.path == rp and (len(parts) == 1 or h.parent == parts[-2])]
            if not hits:
                continue
            h = hits[0]
            found_here = True
            kept_syms.append(f"{rp}:{qual}")
            spans.append(symbol_span(h))
        if not found_here:
            return None
        kept_paths.append(rp)
    seen: set[tuple[str, int, int]] = set()
    spans = [s for s in spans if not ((s.path, s.start, s.end) in seen or seen.add((s.path, s.start, s.end)))]
    return sorted(set(kept_paths)), sorted(set(kept_syms)), spans[:5]


async def derive_codescout(n_repos: int = 15, concurrency: int = 8, limit_per_repo: int | None = None) -> dict[str, Any]:
    t0 = time.time()
    rows = cache.load_cached(cs.DATASET, "default", "train")
    exclude = excluded_repos()
    chosen = select_repos(rows, n_repos, exclude)
    _log(f"  codescout: {len(rows)} rows; {len(chosen)} repos chosen (excluded {len(exclude)} owner/repo names)")
    for repo, cnt, sha in chosen:
        _log(f"    {cnt:4d} rows  {repo}@{sha[:7]}")

    # snapshot + index each chosen repo at its one commit
    indexes: dict[str, RepoIndex] = {}
    repo_ids: dict[str, str] = {}
    for repo, _, sha in chosen:
        owner, name = repo.split("/")
        m, syms = ensure_repo(owner, name, sha)
        indexes[repo] = RepoIndex(m, syms)
        repo_ids[repo] = m.repo_id
    (paths.DATA / "repo_list_codescout.txt").write_text(
        "# CodeScout training repos (newest instance's base_commit)\n" + "\n".join(f"{repo}@{sha}" for repo, _, sha in chosen) + "\n")

    # gold resolution
    per_repo: dict[str, Counter] = defaultdict(Counter)
    candidates: list[tuple[dict[str, Any], str, list[str], list[str], list[Span]]] = []
    for r in rows:
        repo = r["repo"]
        if repo not in indexes:
            continue
        gp, gs = cs.gold(r)
        per_repo[repo]["rows"] += 1
        if not gp or not gs or len(gp) > 3:
            per_repo[repo]["no_gold"] += 1
            continue
        res = resolve_gold(indexes[repo], gp, gs)
        if res is None:
            per_repo[repo]["gold_unresolved"] += 1
            continue
        per_repo[repo]["gold_ok"] += 1
        if limit_per_repo and per_repo[repo]["gold_ok"] > limit_per_repo:
            continue
        candidates.append((r, repo, *res))
    _log(f"  codescout: {len(candidates)} rows with gold resolved at the chosen commit; rewriting with Haiku ...")

    # rewrite
    client = AnthropicClient(llm.HAIKU)
    rw_cache = llm.JsonlCache("codescout")

    async def one(c):
        r, repo, p, s, spans = c
        return await rewrite.rewrite_one(client, rw_cache, r, p, s)

    results = await llm.run_bounded(candidates, one, concurrency=concurrency, label="rewrite")

    tasks: list[Task] = []
    reasons: Counter = Counter()
    for (r, repo, p, s, spans), res in zip(candidates, results):
        if res is None:
            reasons["call_failed"] += 1
            continue
        reasons[res["reason"].split(":")[0]] += 1
        if not res.get("question"):
            continue
        t = cs.to_task(r, question=res["question"])
        assert t is not None
        t = t.model_copy(update={
            "repo_id": repo_ids[repo],
            "grading": t.grading.model_copy(update={"expected_paths": p, "expected_symbols": s, "required_citations": spans}),
        })
        tasks.append(t)
        per_repo[repo]["kept"] += 1

    out = paths.TASKS_RAW / "codescout.jsonl"
    n = write(out, tasks)
    report = {
        "source": "codescout", "records": n, "file": str(out.relative_to(paths.ROOT)),
        "repos": {repo: {"repo_id": repo_ids[repo], **per_repo[repo]} for repo, _, _ in chosen},
        "rewrite_outcomes": dict(reasons),
        "types": dict(Counter(t.task_type for t in tasks)),
        "yield": f"{n}/{sum(c['rows'] for c in per_repo.values())}",
        "seconds": round(time.time() - t0, 1),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "codescout.json").write_text(json.dumps(report, indent=2))
    _log(f"  codescout: {n} tasks kept of {report['yield'].split('/')[1]} rows in chosen repos; rewrites {dict(reasons)}; {report['seconds']}s")
    return report


def main(n_repos: int = 15, concurrency: int = 8, limit_per_repo: int | None = None) -> dict[str, Any]:
    return asyncio.run(derive_codescout(n_repos, concurrency, limit_per_repo))
