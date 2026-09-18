"""datagen CLI.

  uv run python -m codeqa.datagen.cli import --source deepcodebench|sweqa|all
  uv run python -m codeqa.datagen.cli derive --source codescout --repos 15
  uv run python -m codeqa.datagen.cli generate --source structural
  uv run python -m codeqa.datagen.cli filter --samples 4
"""
from __future__ import annotations

import argparse
import json
import sys

from codeqa.shared import paths


def cmd_import(args: argparse.Namespace) -> int:
    from codeqa.datagen import importers
    sources = ["deepcodebench", "sweqa"] if args.source == "all" else [args.source]
    for s in sources:
        print(f"== import {s}", flush=True)
        rep = importers.import_deepcodebench() if s == "deepcodebench" else importers.import_sweqa()
        print(json.dumps({k: v for k, v in rep.items() if k not in ("per_repo",)}, indent=1), flush=True)
    return 0


def cmd_derive(args: argparse.Namespace) -> int:
    from codeqa.datagen import derive
    rep = derive.main(n_repos=args.repos, concurrency=args.concurrency, limit_per_repo=args.limit_per_repo)
    print(json.dumps({k: v for k, v in rep.items() if k != "repos"}, indent=1), flush=True)
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from codeqa.datagen import generate
    rep = generate.main(per_type=args.per_type, cap=args.cap, concurrency=args.concurrency, only=args.only)
    print(json.dumps({k: v for k, v in rep.items() if k != "repos"}, indent=1), flush=True)
    return 0


def cmd_teach(args: argparse.Namespace) -> int:
    from codeqa.datagen import teach
    rep = teach.main(seeds_per_repo=args.seeds_per_repo, concurrency=args.concurrency, only=args.only, limit=args.limit, max_cost_usd=args.max_cost)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_repo"}, indent=1), flush=True)
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    from codeqa.datagen import filter as flt
    rep = flt.main(profile_name=args.profile, samples=args.samples, concurrency=args.concurrency, limit=args.limit,
                   sources=tuple(args.sources) if args.sources else flt.RAW_SOURCES, refine=args.refine, target_n=args.target_n,
                   shard=tuple(int(x) for x in args.shard.split('/')) if args.shard else None)
    print(json.dumps(rep, indent=1), flush=True)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    from codeqa.datagen import split
    rep = split.build(lo=args.lo, hi=args.hi, per_repo_cap=args.per_repo_cap, keep_unmeasured=args.keep_unmeasured)
    print(json.dumps(rep, indent=1), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    paths.ensure_dirs()
    ap = argparse.ArgumentParser(prog="codeqa.datagen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="import DeepCodeBench / SWE-QA into C5 files")
    p.add_argument("--source", choices=["deepcodebench", "sweqa", "all"], default="all")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("derive", help="derive CodeScout locate tasks (snapshot repos, resolve gold, Haiku rewrite)")
    p.add_argument("--source", choices=["codescout"], default="codescout")
    p.add_argument("--repos", type=int, default=15)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit-per-repo", type=int, default=None, help="cap rewrites per repo (smoke runs)")
    p.set_defaults(fn=cmd_derive)
    p = sub.add_parser("generate", help="structural tasks from the index for every training repo")
    p.add_argument("--source", choices=["structural"], default="structural")
    p.add_argument("--per-type", type=int, default=15)
    p.add_argument("--cap", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--only", nargs="*", default=None, help="repo names to restrict to (smoke runs)")
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("teach", help="teacher tasks: Sonnet authors with the tools, Haiku answers blind, judge agreement keeps")
    p.add_argument("--seeds-per-repo", type=int, default=30)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="max attempts this run (smoke)")
    p.add_argument("--max-cost", type=float, default=None, help="stop starting attempts once this run has spent this many USD")
    p.set_defaults(fn=cmd_teach)
    p = sub.add_parser("filter", help="base pass-rate measurement: N samples per task through the student, per-component rates")
    p.add_argument("--profile", default="qwen4b-base")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--refine", action="store_true", help="re-sample only tasks currently at score 0 or 1 with fewer than --target-n samples")
    p.add_argument("--target-n", type=int, default=4)
    p.add_argument("--shard", default=None, help="i/n: run only every n-th task starting at i (parallel processes)")
    p.set_defaults(fn=cmd_filter)
    p = sub.add_parser("split", help="apply the pass-rate window, per-repo cap, write train/ and eval/fast.jsonl")
    p.add_argument("--lo", type=float, default=0.1)
    p.add_argument("--hi", type=float, default=0.9)
    p.add_argument("--per-repo-cap", type=int, default=120)
    p.add_argument("--keep-unmeasured", action="store_true")
    p.set_defaults(fn=cmd_split)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
