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
# CODEQA_SYNC_RUNS="p4_lean p4_full": pull only those run folders under /logs (a full /logs pull is >500 MB and no longer
# fits a one-minute loop); /evals and /models are always pulled, /traces only on a full sync.
if [ -n "${CODEQA_SYNC_RUNS:-}" ]; then
  mkdir -p "$STAGE/logs" "$DEST/logs"
  for run in $CODEQA_SYNC_RUNS; do
    if modal volume get codeqa-data "/logs/$run" "$STAGE/logs" --force >/dev/null 2>&1; then
      mkdir -p "$DEST/logs/$run"; rsync -a "$STAGE/logs/$run/" "$DEST/logs/$run/" && echo "pulled /logs/$run"
    else
      echo "(no /logs/$run on the volume yet)"
    fi
  done
  DIRS="evals models"
else
  DIRS="logs evals traces models"
fi
for d in $DIRS; do
  # prune staged run/eval folders that no longer exist on the volume (renamed or deleted runs must not come back)
  if [ "$d" = logs ] || [ "$d" = evals ]; then
    live=$(modal volume ls codeqa-data "/$d" 2>/dev/null | grep -oE "$d/[A-Za-z0-9_.-]+" | sed "s#^$d/##" | sort -u)
    for x in $(ls "$STAGE/$d" 2>/dev/null); do echo "$live" | grep -qx "$x" || rm -rf "$STAGE/$d/$x"; done
  fi
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
# `modal volume get` stamps every file with the download time, which breaks the workshop's "updated" ordering (it reads
# the mtime of metrics.jsonl / config.json). Restore those two mtimes per run from the volume listing (minute precision).
python3 - "$DEST" <<'PY'
import json, os, subprocess, sys, pathlib
from datetime import datetime, timedelta, timezone
dest = pathlib.Path(sys.argv[1]); TZ = {"PDT": -7, "PST": -8, "UTC": 0, "EDT": -4, "EST": -5}
def parse(s):
    d, t, z = s.split(); dt = datetime.strptime(d + " " + t, "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=timezone(timedelta(hours=TZ.get(z, 0)))).timestamp()
fixed = 0
for run in sorted(p for p in (dest / "logs").iterdir() if p.is_dir()) if (dest / "logs").exists() else []:
    try: rows = json.loads(subprocess.run(["modal", "volume", "ls", "--json", "codeqa-data", f"logs/{run.name}"], capture_output=True, text=True, timeout=60).stdout)
    except Exception: continue
    for r in rows:
        name = r["filename"].rsplit("/", 1)[-1]
        if name in ("metrics.jsonl", "config.json") and (run / name).exists():
            try: ts = parse(r["created_modified"]); os.utime(run / name, (ts, ts)); fixed += 1
            except Exception: pass
print(f"restored {fixed} run mtimes from the volume listing")
PY
rm -rf "$STAGE"
echo "synced into $DEST; volume profiles at $DEST/profiles.modal.yaml (merge new checkpoints into ./profiles.yaml by hand)"
