"""The data/ layout, defined once."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = Path(os.environ.get("CODEQA_DATA_DIR", ROOT / "data"))

REPOS = DATA / "repos"
INDEX = DATA / "index"
TASKS = DATA / "tasks"
TASKS_RAW = TASKS / "raw"
TASKS_TRAIN = TASKS / "train"
TASKS_EVAL = TASKS / "eval"
TRACES = DATA / "traces"
LOGS = DATA / "logs"
EVALS = DATA / "evals"
MODELS = DATA / "models"
MODELS_MANIFEST = MODELS / "manifest.json"

FIXTURES = ROOT / "tests" / "fixtures"
PROFILES = Path(os.environ.get("CODEQA_PROFILES", ROOT / "profiles.yaml"))   # on Modal: /data/profiles.yaml (the volume)


def repo_dir(repo_id: str) -> Path:
    return REPOS / repo_id


def index_dir(repo_id: str) -> Path:
    return INDEX / repo_id


def ensure_dirs() -> None:
    for p in (REPOS, INDEX, TASKS_RAW, TASKS_TRAIN, TASKS_EVAL, TRACES, LOGS, EVALS, MODELS):
        p.mkdir(parents=True, exist_ok=True)
