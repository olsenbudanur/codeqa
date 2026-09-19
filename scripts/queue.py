"""Phase-1 queue: run the arms ONE AT A TIME in a single detached job, waiting for Tinker's LoRA sampling to respond
before each arm (the hot-sampler-weights cap pauses requests silently; a one-token canary is the only readable signal).

  modal run --detach apps/trainer/modal_runner.py --module scripts.queue --args "full lean bash nogates"

Skips arms whose final T=0.2 eval exists; a retried/relaunched job resumes a half-done arm from its last checkpoint
(`scripts.arm` uses --if-exists resume). Every arm runs as a subprocess so its Tinker sessions are closed at exit.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

def _known_arms() -> tuple[str, ...]:
    """Every arm `scripts.arm` defines (bash16v1, bash24eff, lean16, ...), so a new arm needs no edit here.
    2026-09-20: a hard-coded tuple silently dropped the phase-6 arm names and ran the default four instead."""
    try:
        from scripts.arm import ARMS as arm_table
        return tuple(arm_table)
    except Exception:  # noqa: BLE001
        return ("full", "lean", "bash", "nogates")


ARMS = _known_arms()
PROBE_TIMEOUT = 240          # a new LoRA sampler cold-starts in ~40 s; 60 s was too tight
PROBE_INTERVAL = 300
MAX_WAIT = 6 * 3600
COOLDOWN_BETWEEN_ARMS = 120


def log(msg: str) -> None:
    print(f"[queue {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def arm_done(arm: str) -> bool:
    from codeqa.shared import paths
    return any((p / "fast_t02" / "results.json").exists() for p in paths.EVALS.glob(f"qwen*-{os.environ.get('CODEQA_ARM_PREFIX', 'p1_')}{arm}-step*"))


async def _probe_once() -> float | None:
    """Create a fresh LoRA sampler (there may be no checkpoint in this org yet) and sample one token.
    A brand-new sampler cold-starts in ~40 s, so the wait is generous."""
    import tinker
    from codeqa.clients import tinker as tk
    sc = tinker.ServiceClient()
    try:
        path = os.environ.get("CODEQA_CANARY_PATH")            # an existing sampler checkpoint in THIS org: probes without creating a new sampler
        if path:
            samp = sc.create_sampling_client(model_path=path)
        else:
            tc = await sc.create_lora_training_client_async(base_model=os.environ.get("CODEQA_CANARY_MODEL", "Qwen/Qwen3.5-4B"), rank=32)
            samp = await asyncio.wait_for(tc.save_weights_and_get_sampling_client_async(name="queue-canary"), timeout=120)
        prompt = tk.renderer("Qwen/Qwen3.5-4B").build_generation_prompt([{"role": "user", "content": "hi"}])
        t0 = time.time()
        await asyncio.wait_for(samp.sample_async(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=1, temperature=1.0)),
                               timeout=PROBE_TIMEOUT)
        return time.time() - t0
    except asyncio.TimeoutError:
        return None
    finally:
        try:
            sc.close("success", "queue canary").result(timeout=20)
        except Exception:  # noqa: BLE001
            pass


def wait_for_lora_sampling() -> None:
    t0 = time.time()
    while True:
        secs = asyncio.run(_probe_once())
        if secs is not None:
            log(f"LoRA sampling responds ({secs:.1f}s); proceeding")
            return
        waited = time.time() - t0
        if waited > MAX_WAIT:
            log(f"LoRA sampling still paused after {waited/3600:.1f} h; giving up")
            sys.exit(4)
        log(f"LoRA sampling paused (canary timed out); retry in {PROBE_INTERVAL//60} min (waited {waited/60:.0f} min)")
        time.sleep(PROBE_INTERVAL)


def stop_requested() -> bool:
    """A stop asks the queue not to start another arm and the arm to exit at its next step boundary. Request it from the
    laptop with `uv run python -m codeqa.shared.control stop` (a Modal Dict the container reads live; a file put on the
    volume is NOT seen by a running container, 2026-09-20 14:44). `... clear` before the next launch."""
    from codeqa.shared.control import stop_requested as _stop
    return _stop()


def _run_arm(arm: str) -> int:
    """The arm is a child process; SIGTERM (from `modal app stop` via the runner) is forwarded so the arm's handler
    closes its Tinker sessions before the container goes away."""
    import signal
    child = subprocess.Popen([sys.executable, "-u", "-m", "scripts.arm", arm], env=os.environ.copy())

    def forward(signum, frame):
        try:
            child.send_signal(signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    return child.wait()


def main(argv: list[str]) -> int:
    canary = "--no-canary" not in argv                     # --no-canary: skip the probe (results sooner; use when the org is known to sample)
    requested = [a for a in argv if not a.startswith("--")]
    unknown = [a for a in requested if a not in ARMS]
    if unknown:                                            # never fall back to the default four on a typo: fail loudly
        log(f"unknown arm(s) {unknown}; known: {list(ARMS)}"); return 2
    arms = requested or ["full", "lean", "bash", "nogates"]
    from codeqa.clients.tinker import _install_session_hygiene
    _install_session_hygiene()
    for i, arm in enumerate(arms):
        if arm_done(arm):
            log(f"{arm}: already finished (fast_t02 exists); skipping")
            continue
        if stop_requested():
            log("STOP file present; not starting another arm"); return 0
        if canary:
            wait_for_lora_sampling()
        log(f"=== {arm}: starting (arm {i+1}/{len(arms)})")
        t0 = time.time()
        rc = _run_arm(arm)
        log(f"=== {arm}: exit {rc} after {(time.time()-t0)/60:.0f} min")
        if rc != 0:
            log(f"{arm} failed; stopping the queue so nothing runs unattended in a bad state")
            return rc
        if i + 1 < len(arms):
            log(f"cooldown {COOLDOWN_BETWEEN_ARMS}s before the next arm")
            time.sleep(COOLDOWN_BETWEEN_ARMS)
    log("queue done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
