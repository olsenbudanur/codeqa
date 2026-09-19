"""One phase-1 ablation arm as a single job (runs unchanged on the laptop or inside the Modal runner):
train N steps from base -> register the last checkpoint as a profile carrying the variant -> held-out eval at T=0.2.

  uv run python -m scripts.arm full|lean|bash|nogates [--steps 30]
  modal run --detach apps/trainer/modal_runner.py --module scripts.arm --args "lean --steps 30"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ARMS = {  # arm -> (variant, extra env, default steps)
    "full": ("full", {}, 30),
    "lean": ("lean", {}, 30),
    "bash": ("bash", {"CODEQA_BASH_EXECUTOR": "modal", "CODEQA_MODAL_SANDBOXES": "8"}, 30),
    "nogates": ("full", {"CODEQA_HONESTY_GATES": "off"}, 10),
}
def common() -> list[str]:
    from codeqa.shared import paths   # resolves under CODEQA_DATA_DIR (/data inside the Modal runner)
    return ["--tasks", str(paths.TASKS_TRAIN / "all.jsonl"),     # B6 output: 1,489 tasks in the pass-rate window, 62 % gold-graded "--profile", "qwen4b-base", "--group-size", "8", "--groups-per-batch", "16",
            "--lr", "1e-4", "--variant", "none", "--eval-tasks", str(paths.TASKS_EVAL / "fast.jsonl"), "--eval-every", "10", "--save-every", "10",
            "--seed", "0", "--if-exists", "delete"]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("arm", choices=ARMS); ap.add_argument("--steps", type=int); ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    variant, env, default_steps = ARMS[a.arm]
    steps = a.steps or default_steps
    os.environ["CODEQA_RUNTIME"] = "local"          # we are the job; never redirect
    os.environ["CODEQA_AGENT_VARIANT"] = variant
    os.environ.update(env)
    run = f"p1_{a.arm}"
    from codeqa.trainer import run as trainer
    rc = trainer.main([*common(), "--run-name", run, "--steps", str(steps)])
    if rc:
        print(f"[arm] training failed rc={rc}", flush=True); return rc

    from codeqa.shared import paths
    from codeqa.shared.contracts import CheckpointRecord, EndpointProfile
    from codeqa.shared.profiles import add_profile, load_profiles
    recs = [json.loads(l) for l in open(paths.LOGS / run / "checkpoints.jsonl") if l.strip()]
    ck = next((r for r in reversed(recs) if r.get("sampler_path") and r.get("name") != "final"), recs[-1])
    step = int(ck.get("batch", steps)); name = f"qwen4b-{run}-step{step}"
    if name not in load_profiles():
        add_profile(EndpointProfile(name=name, kind="tinker", model=ck["sampler_path"], base_model="Qwen/Qwen3.5-4B", renderer="qwen3_5",
                                    max_context=32768, max_generation_tokens=2048, variant=variant))
    rows = [json.loads(l) for l in open(paths.LOGS / run / "metrics.jsonl") if l.strip()]
    ev = {k.split("/")[-1]: float(rows[-1][k]) for k in rows[-1] if k.startswith("eval/fast/env/all/")} if rows else {}
    man = paths.MODELS_MANIFEST; records = json.loads(man.read_text()) if man.exists() else []
    if not any(r["name"] == name for r in records):
        records.append(CheckpointRecord(name=name, run=run, step=step, created_at=datetime.now(timezone.utc), tinker_path=ck["sampler_path"], profile=name,
                                        evals={"fast_inloop_t1": {k: v for k, v in ev.items() if k in ("reward", "correctness", "format_ok", "citations_grounded", "tool_calls", "turns")}},
                                        notes=f"phase-1 arm {a.arm}: variant {variant}, {steps} steps from base, env {env}").model_dump(mode="json"))
        man.parent.mkdir(parents=True, exist_ok=True); man.write_text(json.dumps(records, indent=1))
    print(f"[arm] registered {name}; in-loop held-out at step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in ev.items() if k in ("reward", "correctness", "tool_calls")), flush=True)
    if a.skip_eval:
        return 0
    from codeqa.evals import run as evals
    for k in list(env):                      # the grader switch must NOT apply to the common-grader eval
        if k == "CODEQA_HONESTY_GATES": os.environ.pop(k, None)
    return evals.main(["--profile", name, "--tasks", str(paths.TASKS_EVAL / "fast.jsonl"), "--set", "fast_t02", "--temperature", "0.2", "--concurrency", "8"])


if __name__ == "__main__":
    sys.exit(main())
