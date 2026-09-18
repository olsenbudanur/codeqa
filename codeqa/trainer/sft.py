"""SFT warm start (optional, decisions.md): teacher traces -> cookbook supervised conversations -> a checkpoint RL resumes from.

  1. Generate teacher traces over TRAINING tasks (Claude, bracket citations), e.g.
       uv run python -u -m codeqa.evals.run --profile claude --tasks data/tasks/train/teacher.jsonl --set sft_seed --concurrency 3
  2. Keep the ones that pass every gate and render them exactly as RepoEnv would (tool prefix + same prompt):
       uv run python -m codeqa.trainer.sft build --traces data/evals/claude/sft_seed/traces --tasks data/tasks/train/teacher.jsonl \
           --out data/tasks/train/sft_seed.jsonl
  3. Train (the lead launches):  uv run python -u -m codeqa.trainer.sft train --file data/tasks/train/sft_seed.jsonl --run-name sft1 --epochs 2
     then RL:  codeqa.trainer.run ... --load-checkpoint <data/logs/sft1/checkpoints.jsonl state_path>

The rendered conversation is `tk.with_tool_prefix(renderer, env.initial_messages(), env.specs())` + the trace's assistant/tool
turns, so the supervised tokens match the RL observation token-for-token (the one rule).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from codeqa.agent.env import RepoEnv
from codeqa.clients import tinker as tk
from codeqa.grader.grade import grade
from codeqa.grader.judge import KeywordJudge, default_client
from codeqa.shared import paths
from codeqa.shared.contracts import Message, Task, Trace
from codeqa.shared.jsonl import read_all
from codeqa.shared.profiles import get_profile

logger = logging.getLogger(__name__)


def render_conversation(task: Task, trace: Trace, profile_name: str = "qwen4b-base") -> list[dict[str, Any]]:
    """Cookbook-format messages: RepoEnv's prefixed system + user message, then the trace's assistant/tool turns."""
    profile = get_profile(profile_name)
    env = RepoEnv(task, profile)
    renderer = tk.renderer(tk.base_model_of(profile), profile.renderer)
    prefix = tk.with_tool_prefix(renderer, env.initial_messages(), env.specs())
    body = [m for m in trace.messages if m.role in ("assistant", "tool")]
    # the driver's budget notice is a user message mid-trace; keep it so the model sees what it saw
    body_full: list[Message] = []
    started = False
    for m in trace.messages:
        if m.role in ("assistant", "tool"):
            started = True
        if started and m.role != "system":
            body_full.append(m)
    conv = tk.to_cookbook(list(prefix) + body_full)
    for m in conv:                                   # JSON-serializable; rehydrated by CodeQASFTDatasetBuilder
        if m.get("tool_calls"):
            m["tool_calls"] = [tc.model_dump(mode="json") for tc in m["tool_calls"]]
    return conv


def rehydrate(conv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON messages -> cookbook messages (tool_calls back to renderer ToolCall objects)."""
    from tinker_cookbook.renderers.base import ToolCall as CbToolCall
    out = []
    for m in conv:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [tc if not isinstance(tc, dict) else CbToolCall.model_validate(tc) for tc in m["tool_calls"]]
        out.append(m)
    return out


class _ListDataset:
    """Minimal cookbook SupervisedDataset over a list of datums."""

    def __init__(self, datums: list, batch_size: int):
        self.datums, self.batch_size = datums, batch_size

    def get_batch(self, index: int):
        return self.datums[index * self.batch_size:(index + 1) * self.batch_size]

    def __len__(self) -> int:
        return max((len(self.datums) + self.batch_size - 1) // self.batch_size, 0)

    def set_epoch(self, seed: int = 0) -> None:
        import random
        random.Random(seed).shuffle(self.datums)


def load_datums(file: Path, profile_name: str, max_length: int) -> list:
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import conversation_to_datum
    profile = get_profile(profile_name)
    renderer = tk.renderer(tk.base_model_of(profile), profile.renderer)
    datums = []
    with file.open() as fh:
        for line in fh:
            if line.strip():
                conv = rehydrate(json.loads(line)["messages"])
                datums.append(conversation_to_datum(conv, renderer, max_length, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES))
    return datums


async def build(traces_dir: Path, tasks_path: Path, out: Path, profile_name: str, min_reward: float, judge: str,
                max_len_tokens: int | None) -> dict[str, int]:
    tasks = {t.task_id: t for t in read_all(tasks_path, Task)}
    client = KeywordJudge() if judge == "keyword" else default_client()
    stats = {"traces": 0, "no_task": 0, "gate_failed": 0, "below_min": 0, "kept": 0}
    rows: list[dict[str, Any]] = []
    for f in sorted(traces_dir.glob("*.json")):
        trace = Trace.model_validate_json(f.read_text())
        stats["traces"] += 1
        task = tasks.get(trace.task_id)
        if task is None:
            stats["no_task"] += 1
            continue
        r = await grade(task, trace, judge_client=client)
        if r.gate_failed is not None:
            stats["gate_failed"] += 1
            continue
        if not (r.reward >= min_reward):
            stats["below_min"] += 1
            continue
        conv = render_conversation(task, trace, profile_name)
        rows.append({"messages": conv, "task_id": task.task_id, "source": task.source, "task_type": task.task_type, "reward": r.reward})
        stats["kept"] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    logger.info("sft build: %s -> %s", stats, out)
    return stats


def datum_stats(file: Path, profile_name: str = "qwen4b-base", max_length: int = 32768, n: int = 3) -> list[int]:
    """Token lengths of the first n rendered conversations (proof the renderer accepts them)."""
    return [d.model_input.length for d in load_datums(file, profile_name, max_length)[:n]]


def build_sft_config(file: Path, run_name: str, profile_name: str = "qwen4b-base", epochs: int = 2, batch_size: int = 8,
                     lr: float = 1e-4, lora_rank: int = 32, max_length: int = 32768, test_size: int = 8, save_every: int = 10):
    import chz
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.types import SupervisedDatasetBuilder

    @chz.chz
    class CodeQASFTDatasetBuilder(SupervisedDatasetBuilder):
        file_path: str
        profile_name: str
        batch_size: int
        test_size: int
        max_length: int

        def __call__(self):
            datums = load_datums(Path(self.file_path), self.profile_name, self.max_length)
            test = datums[:self.test_size] if self.test_size else []
            trainset = datums[self.test_size:] if self.test_size else datums
            return _ListDataset(trainset, self.batch_size), (_ListDataset(test, self.batch_size) if test else None)

    profile = get_profile(profile_name)
    base = tk.base_model_of(profile)
    builder = CodeQASFTDatasetBuilder(file_path=str(file), profile_name=profile_name, batch_size=batch_size, test_size=test_size, max_length=max_length)
    return train.Config(log_path=str(paths.LOGS / run_name), model_name=base, recipe_name="codeqa_sft", renderer_name=profile.renderer,
                        dataset_builder=builder, learning_rate=lr, num_epochs=epochs, lora_rank=lora_rank, save_every=save_every, eval_every=0)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout, force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="traces + tasks -> supervised conversations JSONL (gate-passing only)")
    b.add_argument("--traces", required=True); b.add_argument("--tasks", required=True); b.add_argument("--out", required=True)
    b.add_argument("--profile", default="qwen4b-base", help="tinker profile whose renderer/tool prefix to render with")
    b.add_argument("--min-reward", type=float, default=0.5); b.add_argument("--judge", default="haiku", choices=["haiku", "keyword"])
    t = sub.add_parser("train", help="run the cookbook supervised loop (the lead launches)")
    t.add_argument("--file", required=True); t.add_argument("--run-name", required=True); t.add_argument("--profile", default="qwen4b-base")
    t.add_argument("--epochs", type=int, default=2); t.add_argument("--batch-size", type=int, default=8); t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--if-exists", default="delete", choices=["delete", "resume", "raise", "ask"])
    a = ap.parse_args(argv)
    if a.cmd == "build":
        stats = asyncio.run(build(Path(a.traces), Path(a.tasks), Path(a.out), a.profile, a.min_reward, a.judge, None))
        print(stats, flush=True)
        if stats["kept"]:
            print("datum token lengths (first 3):", datum_stats(Path(a.out), a.profile), flush=True)
        return 0
    from tinker_cookbook import cli_utils
    from tinker_cookbook.supervised import train
    cfg = build_sft_config(Path(a.file), a.run_name, a.profile, a.epochs, a.batch_size, a.lr)
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=a.if_exists)
    print(f"sft {a.run_name}: model={cfg.model_name} epochs={cfg.num_epochs} lr={cfg.learning_rate} log={cfg.log_path}", flush=True)
    asyncio.run(train.main(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
