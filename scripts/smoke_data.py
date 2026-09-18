"""Can we actually get the data we want? Pull real rows, build validated Tasks, snapshot + index one repo, cross-check.

Run: uv run python -m scripts.smoke_data
Needs no API keys (GitHub via `gh auth token`).
"""
from __future__ import annotations

import json
import sys
import time

from codeqa.clients import github, hf
from codeqa.datagen.sources import derive_codescout, import_deepcodebench, import_sweqa
from codeqa.agent.indexing.index import build_index
from codeqa.agent.indexing.snapshot import snapshot
from codeqa.shared import paths
from codeqa.shared.contracts import Task
from codeqa.shared.jsonl import write

paths.ensure_dirs()
ok = True


def section(title: str):
    print(f"\n=== {title}")


# 1. DeepCodeBench ----------------------------------------------------------
section("DeepCodeBench rows -> Task")
t0 = time.time()
rows = hf.rows(import_deepcodebench.DATASET, "default", "train", 0, 5)
dcb = [import_deepcodebench.to_task(r) for r in rows]
print(f"  {len(dcb)} tasks in {time.time()-t0:.1f}s; repo_ids: {sorted({t.repo_id for t in dcb})}")
print(f"  types: {[t.task_type for t in dcb]}; rubric sizes: {[len(t.grading.rubric) for t in dcb]}; paths: {[t.grading.expected_paths for t in dcb][:2]}")
print("  sample:", json.dumps(dcb[0].model_dump(), indent=None)[:600])
write(paths.TASKS_RAW / "smoke_deepcodebench.jsonl", dcb)

# 2. CodeScout --------------------------------------------------------------
section("CodeScout rows -> Task (gold only; question is a placeholder until rewrite)")
rows = hf.rows(derive_codescout.DATASET, "default", "train", 0, 20)
cs = [t for t in (derive_codescout.to_task(r) for r in rows) if t]
print(f"  {len(cs)}/{len(rows)} rows kept (<=3 non-test files with entity gold)")
if cs:
    print(f"  sample gold: paths={cs[0].grading.expected_paths} symbols={cs[0].grading.expected_symbols[:3]} repo={cs[0].repo_id}")
write(paths.TASKS_RAW / "smoke_codescout.jsonl", cs)

# 3. SWE-QA-Bench + commits -------------------------------------------------
section("SWE-QA-Bench rows + repo_commit.txt -> eval Task with citations")
tree = github.fetch_text(*import_sweqa.GITHUB_REPO, "repo_commit.txt") if False else None
# locate repo_commit.txt anywhere in the repo tree
import httpx
r = httpx.get(f"https://api.github.com/repos/{import_sweqa.GITHUB_REPO[0]}/{import_sweqa.GITHUB_REPO[1]}/git/trees/HEAD",
              params={"recursive": "1"}, headers=github._headers(), timeout=60)
r.raise_for_status()
rc_paths = [t["path"] for t in r.json()["tree"] if t["path"].endswith("repo_commit.txt")]
print("  repo_commit.txt at:", rc_paths)
commits = import_sweqa.parse_repo_commits(github.fetch_text(*import_sweqa.GITHUB_REPO, rc_paths[0]))
print(f"  parsed {len(commits)} repo->sha entries; flask={commits.get('flask')}")
flask_sha = commits["flask"]
rows = hf.rows(import_sweqa.DATASET, "default", "flask", 0, 10)
sq = [import_sweqa.to_task(r, "flask", flask_sha, i) for i, r in enumerate(rows)]
n_cit = sum(1 for t in sq if t.grading.required_citations)
print(f"  {len(sq)} tasks; {n_cit} with extracted citations; example: {sq[0].grading.required_citations[:2]}")
write(paths.TASKS_EVAL / "smoke_sweqa_flask.jsonl", sq)

# 4. Snapshot flask at the pinned commit -------------------------------------
section(f"Snapshot pallets/flask@{flask_sha} via GitHub tarball")
t0 = time.time()
m = snapshot("pallets", "flask", flask_sha)
py = [f for f in m.files if f.lang == "python"]
print(f"  repo_id={m.repo_id}; {len(m.files)} files kept ({len(py)} python); dropped={m.dropped}; {time.time()-t0:.1f}s")

# 5. Index it with tree-sitter ------------------------------------------------
section("tree-sitter index")
t0 = time.time()
syms = build_index(m)
kinds = {k: sum(1 for s in syms if s.kind == k) for k in ("function", "class", "method")}
print(f"  {len(syms)} symbols {kinds} in {time.time()-t0:.1f}s")
probe = next(s for s in syms if s.name == "Flask" and s.kind == "class")
line = (paths.repo_dir(m.repo_id) / probe.path).read_text().splitlines()[probe.start - 1]
print(f"  probe: {probe.qualified} L{probe.start}-L{probe.end}; line {probe.start} reads: {line.strip()[:60]!r}")
ok &= line.strip().startswith("class Flask")

# 6. Cross-check SWE-QA citations against the snapshot ------------------------
section("SWE-QA flask citations vs snapshot")
root = paths.repo_dir(m.repo_id)
checked = exists = inrange = 0
for t in sq:
    for c in t.grading.required_citations:
        checked += 1
        p = root / c.path
        if p.exists():
            exists += 1
            if c.end <= len(p.read_text().splitlines()):
                inrange += 1
print(f"  {checked} citations: {exists} paths exist, {inrange} line ranges within file length")
ok &= exists > 0

# 7. Validate every written record round-trips through the contract ------------
section("contract round-trip")
for f in (paths.TASKS_RAW / "smoke_deepcodebench.jsonl", paths.TASKS_RAW / "smoke_codescout.jsonl", paths.TASKS_EVAL / "smoke_sweqa_flask.jsonl"):
    n = sum(1 for line in f.read_text().splitlines() if Task.model_validate_json(line))
    print(f"  {f.relative_to(paths.ROOT)}: {n} valid")

print("\nRESULT:", "OK" if ok else "PROBLEMS (see above)")
sys.exit(0 if ok else 1)
