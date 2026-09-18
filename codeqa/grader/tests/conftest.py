from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ["CODEQA_GRADER_TOKENIZER"] = "proxy"     # deterministic counts in tests; the real tokenizer is covered by test_tokenizer_live

from codeqa.grader import judge as judge_mod
from codeqa.grader.repo import FIXTURE_REPO_ID, load_repo
from codeqa.shared import paths
from codeqa.shared.contracts import Task, Trace
from codeqa.shared.jsonl import read_all

FIX = paths.FIXTURES


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(judge_mod, "BACKOFF_BASE", 0.0)


@pytest.fixture
def repo():
    return load_repo(FIXTURE_REPO_ID)


@pytest.fixture
def tasks() -> dict[str, Task]:
    return {t.task_id: t for t in read_all(FIX / "tasks.jsonl", Task)}


def load_trace(name: str) -> Trace:
    return Trace.model_validate(json.loads((FIX / "traces" / f"{name}.json").read_text()))


@pytest.fixture
def trace():
    return load_trace
