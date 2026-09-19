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
    "full": ("full", {}, 20),
    "lean": ("lean", {}, 20),
    "bash": ("bash", {"CODEQA_BASH_EXECUTOR": "modal", "CODEQA_MODAL_SANDBOXES": "16"}, 20),
    "nogates": ("full", {"CODEQA_HONESTY_GATES": "off"}, 10),
    "stage2": ("lean", {}, 30),       # cheap config: lean prefix (1k tree map, 4 tools), merged task set, 8 groups (env overrides below)
}
BASE_PROFILE = os.environ.get("CODEQA_ARM_PROFILE", "qwen4b-base")
BASE_MODEL = {"qwen4b-base": "Qwen/Qwen3.5-4B", "qwen9b-base": "Qwen/Qwen3.5-9B"}[BASE_PROFILE]
SIZE_TAG = "qwen9b" if "9B" in BASE_MODEL else "qwen4b"


def common() -> list[str]:
    from codeqa.shared import paths   # resolves under CODEQA_DATA_DIR (/data inside the Modal runner)
    return ["--tasks", str(paths.TASKS_TRAIN / os.environ.get("CODEQA_ARM_TASKS", "run1.jsonl")),     # CODEQA_ARM_TASKS: file under data/tasks/train
            "--profile", BASE_PROFILE,                             # CODEQA_ARM_PROFILE: qwen4b-base (default) | qwen9b-base
            "--group-size", "8", "--groups-per-batch", os.environ.get("CODEQA_ARM_GROUPS", "16"),
            "--lr", "1e-4", "--variant", "none", "--eval-tasks", str(paths.TASKS_EVAL / "fast.jsonl"), "--eval-every", os.environ.get("CODEQA_ARM_EVAL_EVERY", "10"), "--eval-max-tasks", os.environ.get("CODEQA_ARM_EVAL_TASKS", "60"), "--save-every", "5",
            "--seed", "0", "--if-exists", "resume",
            *(["--judge-model", os.environ["CODEQA_ARM_JUDGE"]] if os.environ.get("CODEQA_ARM_JUDGE") else [])]   # e.g. claude-sonnet-5: cleaner explain rewards      # resume: Modal restarts a preempted function with the same input (seen 2026-09-19 20:06)


BUDGET_USD = float(os.environ.get("CODEQA_TINKER_BUDGET_USD", "100"))
USD_PER_M_TOKENS = float(os.environ.get("CODEQA_USD_PER_M_TOKENS", "0.40"))   # conservative all-in 4B rate (9B measured ≈ 0.57 $/M observation tokens incl. decode + training)


def _estimated_tinker_spend_usd() -> float:
    """Sum observation + action tokens over every run under this prefix (train rows and held-out rows) x rate."""
    import json
    from codeqa.shared import paths
    prefix = os.environ.get("CODEQA_ARM_PREFIX", "p1_")
    tokens = 0.0
    for mf in paths.LOGS.glob(f"{prefix}*/metrics.jsonl"):
        for line in open(mf):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            for k, v in r.items():                                    # only the all-episodes totals: per-source/per-type keys would triple-count
                if k in ("env/all/total_ob_tokens", "env/all/total_ac_tokens") or (k.startswith("eval/") and k.endswith(("env/all/total_ob_tokens", "env/all/total_ac_tokens"))):
                    tokens += float(v or 0)
    return tokens / 1e6 * USD_PER_M_TOKENS


def _start_watchdog(run: str, minutes: int) -> None:
    """If no metrics row lands for `minutes`, finish our Tinker sessions and exit 3. With `--if-exists resume` and a
    per-step checkpoint, a relaunch (or Modal's retry) continues from the last step instead of leaving a zombie."""
    import threading, time
    from codeqa.shared import paths
    from codeqa.clients.tinker import close_all_sessions
    m = paths.LOGS / run / "metrics.jsonl"

    def loop() -> None:
        last_rows, last_change = -1, time.time()
        while True:
            time.sleep(30)
            rows = sum(1 for _ in open(m)) if m.exists() else 0
            spent = _estimated_tinker_spend_usd()
            if spent > BUDGET_USD and rows != last_rows:                 # hard cap on Tinker spend across this prefix's runs (est.)
                print(f"[arm] BUDGET: estimated Tinker spend ${spent:.0f} > cap ${BUDGET_USD:.0f}; stopping at this step boundary", flush=True)
                (paths.DATA / "STOP").touch()                              # the queue starts no further arm
                close_all_sessions("success", "budget cap reached")
                os._exit(0)
            if (paths.DATA / "STOP").exists() and rows != last_rows:      # a new row = optimizer step just ran; nothing is in flight yet
                print(f"[arm] STOP file present at step boundary (rows={rows}); closing Tinker sessions and exiting 0", flush=True)
                close_all_sessions("success", "stopped by STOP file at a step boundary")
                os._exit(0)
            if rows != last_rows:
                last_rows, last_change = rows, time.time()
            elif time.time() - last_change > minutes * 60:
                print(f"[arm] WATCHDOG: no new metrics row for {minutes} min (rows={rows}); closing Tinker sessions and exiting 3", flush=True)
                close_all_sessions("interrupted", "watchdog: no progress")
                os._exit(3)

    threading.Thread(target=loop, daemon=True, name="stall-watchdog").start()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("arm", choices=ARMS); ap.add_argument("--steps", type=int); ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    variant, env, default_steps = ARMS[a.arm]
    default_steps = int(os.environ.get("CODEQA_ARM_STEPS_NOGATES", default_steps)) if a.arm == "nogates" else int(os.environ.get("CODEQA_ARM_STEPS", default_steps))
    steps = a.steps or default_steps
    os.environ["CODEQA_RUNTIME"] = "local"          # we are the job; never redirect
    os.environ["CODEQA_AGENT_VARIANT"] = variant
    os.environ.update(env)
    run = f"{os.environ.get('CODEQA_ARM_PREFIX', 'p1_')}{a.arm}"
    from codeqa.shared import paths as _paths
    os.environ["CODEQA_GROUPS_DIR"] = str(_paths.LOGS / run)          # groups.json sidecars for the Workshop
    from codeqa.clients.tinker import _install_session_hygiene, close_all_sessions
    _install_session_hygiene()                     # the cookbook's own ServiceClient gets tracked and closed on exit/SIGTERM
    from codeqa.shared import paths
    if (paths.LOGS / run / "checkpoints.jsonl").exists():          # resuming: the "first" evaluator call is not step 0 -> do not skip it
        os.environ["CODEQA_SKIP_FIRST_EVAL"] = "0"
    _start_watchdog(run, minutes=int(os.environ.get("CODEQA_STALL_MINUTES", "40")))
    from codeqa.trainer import run as trainer
    rc = trainer.main([*common(), "--run-name", run, "--steps", str(steps)])
    if rc:
        print(f"[arm] training failed rc={rc}", flush=True); return rc

    from codeqa.shared import paths
    from codeqa.shared.contracts import CheckpointRecord, EndpointProfile
    from codeqa.shared.profiles import add_profile, load_profiles
    recs = [json.loads(l) for l in open(paths.LOGS / run / "checkpoints.jsonl") if l.strip()]
    ck = next((r for r in recs if r.get("name") == "final" and r.get("sampler_path")), None) or next((r for r in reversed(recs) if r.get("sampler_path")), recs[-1])
    step = int(ck.get("batch", steps)); name = f"{SIZE_TAG}-{run}-step{step}"
    if name not in load_profiles():
        add_profile(EndpointProfile(name=name, kind="tinker", model=ck["sampler_path"], base_model=BASE_MODEL, renderer="qwen3_5",
                                    max_context=65536, max_generation_tokens=2048, variant=variant,
                                    api_key_env=os.environ.get("CODEQA_TINKER_KEY_ENV") or None))   # phase 1 trains in the second org: TINKER_API_KEY_NEW
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
    return evals.main(["--profile", name, "--tasks", str(paths.TASKS_EVAL / "fast.jsonl"), "--set", "fast_t02", "--temperature", "0.2", "--concurrency", "8",
                       "--max-tasks", os.environ.get("CODEQA_ARM_EVAL_TASKS", "60"),
                       *(["--judge-model", os.environ["CODEQA_ARM_JUDGE"]] if os.environ.get("CODEQA_ARM_JUDGE") else [])])


if __name__ == "__main__":
    sys.exit(main())
