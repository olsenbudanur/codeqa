#!/usr/bin/env bash
# Pull job outputs from the Modal volume into data/ WITHOUT touching local outputs the volume does not have.
#   modal volume get … --force replaces any local directory that also exists on the volume (it deleted local
#   eval sets on 2026-09-19). So: stage under data/.modal_pull/, then merge file-by-file with rsync (no --delete),
#   and merge data/models/manifest.json by record name instead of copying it.
# Usage: scripts/modal_sync.sh [data_dir]   (default ./data)
set -euo pipefail
DEST="${1:-data}"
STAGE="$DEST/.modal_pull"
rm -rf "$STAGE"; mkdir -p "$STAGE"
for d in logs evals traces models; do
  if modal volume get codeqa-data "/$d" "$STAGE" --force >/dev/null 2>&1; then
    if [ "$d" = "models" ]; then
      mkdir -p "$DEST/models"
      python3 - "$STAGE/models/manifest.json" "$DEST/models/manifest.json" <<'PY'
import json, sys, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
a = json.loads(src.read_text()) if src.exists() else []
b = json.loads(dst.read_text()) if dst.exists() else []
seen = {r["name"] for r in b}
merged = b + [r for r in a if r["name"] not in seen]      # local wins on conflict; volume adds new records
dst.write_text(json.dumps(merged, indent=1)); print(f"pulled /models: manifest merged ({len(b)} local + {len(merged)-len(b)} new)")
PY
      rsync -a --exclude manifest.json "$STAGE/models/" "$DEST/models/"
    else
      mkdir -p "$DEST/$d"; rsync -a "$STAGE/$d/" "$DEST/$d/" && echo "pulled /$d (merged, nothing deleted)"
    fi
  else
    echo "(no /$d on the volume yet)"
  fi
done
modal volume get codeqa-data /profiles.yaml "$DEST/profiles.modal.yaml" --force >/dev/null 2>&1 || true
rm -rf "$STAGE"
echo "synced into $DEST; volume profiles at $DEST/profiles.modal.yaml (merge new checkpoints into ./profiles.yaml by hand)"
