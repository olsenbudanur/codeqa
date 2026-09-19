#!/usr/bin/env bash
# Pull job outputs from the Modal volume into the local data/ (or an EC2 box's data/): logs, evals, traces, models, profiles.
# Usage: scripts/modal_sync.sh [data_dir]      (default ./data). Safe to re-run; --force overwrites changed files.
# Note: `modal volume get VOL /dir DEST` places the directory at DEST/dir, so DEST is the data root, not data/dir.
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
for d in logs evals traces models; do
  modal volume get codeqa-data "/$d" "$DEST" --force >/dev/null 2>&1 && echo "pulled /$d" || echo "(no /$d on the volume yet)"
done
modal volume get codeqa-data /profiles.yaml "$DEST/profiles.modal.yaml" --force >/dev/null 2>&1 || true
echo "synced to $DEST (volume profiles.yaml at $DEST/profiles.modal.yaml; merge new checkpoints into ./profiles.yaml by hand)"
