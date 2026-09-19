#!/usr/bin/env bash
# Run the ablation arms back to back on the laptop (one variable each, 10 steps from base, same data/seed/eval),
# and register each step-10 checkpoint as a profile carrying its variant. Stops at the first failure.
#   scripts/queue_arms.sh            # bash, lean, nogates (in that order)
#   scripts/queue_arms.sh lean       # a subset
set -uo pipefail
cd "$(dirname "$0")/.."
COMMON="--tasks data/tasks/train/run1.jsonl --profile qwen4b-base --steps 10 --group-size 8 --groups-per-batch 16 --lr 1e-4 --variant none --eval-tasks data/tasks/eval/fast.jsonl --eval-every 10 --save-every 10 --seed 0 --if-exists delete"
ARMS="${*:-bash lean nogates}"
for arm in $ARMS; do
  case "$arm" in
    bash)    ENVS="CODEQA_AGENT_VARIANT=bash CODEQA_BASH_EXECUTOR=modal CODEQA_MODAL_SANDBOXES=8"; VARIANT=bash ;;
    lean)    ENVS="CODEQA_AGENT_VARIANT=lean"; VARIANT=lean ;;
    nogates) ENVS="CODEQA_AGENT_VARIANT=full CODEQA_HONESTY_GATES=off"; VARIANT=full ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  run="run1_$arm"
  echo "=== $(date '+%H:%M:%S') start $run ($ENVS)"
  env CODEQA_RUNTIME=local $ENVS uv run python -u -m codeqa.trainer.run $COMMON --run-name "$run" > "data/$run.log" 2>&1
  rc=$?
  echo "=== $(date '+%H:%M:%S') $run exit $rc"
  [ $rc -ne 0 ] && { echo "stopping queue"; exit $rc; }
  uv run python - "$run" "$VARIANT" <<'PY'
import json, sys, pathlib
from datetime import datetime, timezone
from codeqa.shared import paths
from codeqa.shared.contracts import CheckpointRecord, EndpointProfile
from codeqa.shared.profiles import add_profile, load_profiles
run, variant = sys.argv[1], sys.argv[2]
recs = [json.loads(l) for l in open(f"data/logs/{run}/checkpoints.jsonl") if l.strip()]
ck = next((r for r in recs if r.get("batch") == 10 and r.get("sampler_path")), recs[-1])
name = f"qwen4b-{run}-step10"
if name not in load_profiles():
    add_profile(EndpointProfile(name=name, kind="tinker", model=ck["sampler_path"], base_model="Qwen/Qwen3.5-4B", renderer="qwen3_5",
                                max_context=32768, max_generation_tokens=2048, variant=variant))
rows = [json.loads(l) for l in open(f"data/logs/{run}/metrics.jsonl") if l.strip()]
ev = {k.split("/")[-1]: rows[-1][k] for k in rows[-1] if k.startswith("eval/fast/env/all/")} if rows else {}
man = paths.MODELS_MANIFEST; records = json.loads(man.read_text()) if man.exists() else []
if not any(r["name"] == name for r in records):
    records.append(CheckpointRecord(name=name, run=run, step=10, created_at=datetime.now(timezone.utc), tinker_path=ck["sampler_path"], profile=name,
                                    evals={"fast_inloop": {k: float(v) for k, v in ev.items() if k in ("reward","correctness","format_ok","citations_grounded","tool_calls","turns")}},
                                    notes=f"ablation arm {run}, variant {variant}, 10 steps from base").model_dump(mode="json"))
    man.write_text(json.dumps(records, indent=1))
print(f"registered {name} -> {ck['sampler_path'][-30:]} | held-out at step 10: " + ", ".join(f"{k}={float(v):.3f}" for k, v in ev.items() if k in ("reward","correctness","tool_calls")))
PY
done
echo "=== queue done"
