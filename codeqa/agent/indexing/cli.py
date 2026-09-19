"""Indexing CLI: snapshot -> index -> summarize -> map (-> nodoc), per repo or for a list.

  uv run python -m codeqa.agent.indexing.cli snapshot pallets/flask@85c5d93
  uv run python -m codeqa.agent.indexing.cli index pallets__flask__85c5d93
  uv run python -m codeqa.agent.indexing.cli summarize pallets__flask__85c5d93
  uv run python -m codeqa.agent.indexing.cli map pallets__flask__85c5d93
  uv run python -m codeqa.agent.indexing.cli nodoc pallets__flask__85c5d93
  uv run python -m codeqa.agent.indexing.cli all --repos data/repo_list.txt [--summaries] [--nodoc]

Repo list lines: `owner/repo@sha` (sha may be short); `#` comments allowed. Haiku summaries only with `--summaries`.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback

from codeqa.agent.indexing.index import build_index, load_symbols
from codeqa.agent.indexing.repomap import build_map, count_tokens
from codeqa.agent.indexing.snapshot import load_manifest, snapshot
from codeqa.agent.indexing.strip_docstrings import make_nodoc
from codeqa.agent.indexing.summaries import load_summaries, summarize
from codeqa.shared import paths


def parse_spec(spec: str) -> tuple[str, str, str]:
    repo, _, sha = spec.strip().partition("@")
    owner, _, name = repo.partition("/")
    if not (owner and name and sha):
        raise SystemExit(f"bad repo spec {spec!r}; want owner/repo@sha")
    return owner, name, sha


def cmd_snapshot(spec: str, force: bool = False) -> str:
    owner, name, sha = parse_spec(spec)
    t0 = time.time()
    m = snapshot(owner, name, sha, force=force)
    print(f"[snapshot] {m.repo_id}: {len(m.files)} files, dropped {m.dropped} ({time.time()-t0:.1f}s)")
    return m.repo_id


def cmd_index(repo_id: str) -> None:
    t0 = time.time()
    syms = build_index(load_manifest(repo_id))
    print(f"[index] {repo_id}: {len(syms)} symbols ({time.time()-t0:.1f}s)")


def cmd_summarize(repo_id: str) -> None:
    t0 = time.time()
    m = load_manifest(repo_id)
    asyncio.run(summarize(m, load_symbols(repo_id)))
    print(f"[summarize] {repo_id}: done ({time.time()-t0:.1f}s)")


def cmd_map(repo_id: str) -> None:
    m = load_manifest(repo_id)
    text = build_map(m, load_symbols(repo_id), load_summaries(repo_id))
    print(f"[map] {repo_id}: {len(text.splitlines())} lines, {count_tokens(text)} tokens -> {paths.index_dir(repo_id) / 'map.txt'}")


def cmd_nodoc(repo_id: str) -> None:
    t0 = time.time()
    m, syms = make_nodoc(load_manifest(repo_id))
    print(f"[nodoc] {m.repo_id}: {m.dropped.get('docstrings_blanked', 0)} docstrings blanked, {len(syms)} symbols ({time.time()-t0:.1f}s)")
    build_map(m, syms, {})  # structural map without summaries (structural tasks only)


def cmd_all(repos_file: str, fast: bool, nodoc: bool) -> int:
    specs = [l.split("#", 1)[0].strip() for l in open(repos_file)]
    specs = [s for s in specs if s]
    failures: list[str] = []
    for spec in specs:
        t0 = time.time()
        try:
            rid = cmd_snapshot(spec)
            cmd_index(rid)
            if not fast:
                cmd_summarize(rid)
            cmd_map(rid)
            if nodoc:
                cmd_nodoc(rid)
            print(f"[all] {rid} ok in {time.time()-t0:.0f}s\n")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{spec}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"[all] {len(specs) - len(failures)}/{len(specs)} repos indexed")
    for f in failures:
        print("  FAILED", f)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="codeqa.agent.indexing.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("snapshot"); p.add_argument("spec"); p.add_argument("--force", action="store_true")
    for name in ("index", "summarize", "map", "nodoc"):
        sub.add_parser(name).add_argument("repo_id")
    p = sub.add_parser("all"); p.add_argument("--repos", required=True); p.add_argument("--summaries", action="store_true", help="also run Haiku summaries + the full map (off by default; only the `full`/`tree_overview` agent variants need them)"); p.add_argument("--fast", action="store_true", help="(now the default; kept for old scripts)"); p.add_argument("--nodoc", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "snapshot":
        cmd_snapshot(a.spec, a.force)
    elif a.cmd == "index":
        cmd_index(a.repo_id)
    elif a.cmd == "summarize":
        cmd_summarize(a.repo_id)
    elif a.cmd == "map":
        cmd_map(a.repo_id)
    elif a.cmd == "nodoc":
        cmd_nodoc(a.repo_id)
    elif a.cmd == "all":
        return cmd_all(a.repos, fast=not a.summaries, nodoc=a.nodoc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
