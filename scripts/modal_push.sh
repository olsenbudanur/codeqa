#!/usr/bin/env bash
# Push inputs from the laptop to the Modal volume: snapshots, index, tasks, profiles. The volume is the source of
# truth for jobs (decision #8). Safe to re-run; only changed files are uploaded (--force overwrites).
# Usage:
#   scripts/modal_push.sh                       # index + tasks + profiles (fast; run after adding tasks or repos)
#   scripts/modal_push.sh repos                 # all snapshots too (~1 GB the first time)
#   scripts/modal_push.sh repo <repo_id>        # one snapshot + its index (after `indexing.cli all` locally)
set -euo pipefail
case "${1:-}" in
  repos)
    # one put per snapshot: a single ~1 GB put of data/repos died with "stream timeout" (2026-09-19)
    for d in data/repos/*/; do rid=$(basename "$d"); modal volume put codeqa-data "$d" "/repos/$rid" --force || echo "FAILED $rid"; done
    modal volume put codeqa-data data/index /index --force ;;
  repo)
    rid="$2"
    modal volume put codeqa-data "data/repos/$rid" "/repos/$rid" --force
    modal volume put codeqa-data "data/index/$rid" "/index/$rid" --force
    [ -d "data/repos/${rid}__nodoc" ] && modal volume put codeqa-data "data/repos/${rid}__nodoc" "/repos/${rid}__nodoc" --force && modal volume put codeqa-data "data/index/${rid}__nodoc" "/index/${rid}__nodoc" --force ;;
  *)
    modal volume put codeqa-data data/index /index --force
    modal volume put codeqa-data data/tasks /tasks --force
    modal volume put codeqa-data profiles.yaml /profiles.yaml --force ;;
esac
echo "pushed. Jobs see the new files on their next start (a running job does not)."
