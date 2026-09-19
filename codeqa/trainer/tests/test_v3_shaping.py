"""v3 trainer pieces: the forced-answer factor constant and the training-step publication used by the reward ramp."""
from __future__ import annotations

import os

from codeqa.trainer.dataset_builder import CodeQADataset
from codeqa.trainer.group_rewards import FORCED_ANSWER_FACTOR


def test_forced_factor():
    assert FORCED_ANSWER_FACTOR == 0.9


def test_dataset_publishes_step_unless_told_not_to(monkeypatch):
    monkeypatch.delenv("CODEQA_GROUPS_DIR", raising=False)
    monkeypatch.setenv("CODEQA_TRAIN_STEP", "5")
    CodeQADataset([object()] * 4, batch_size=2, publish_step=False).get_batch(1)
    assert os.environ["CODEQA_TRAIN_STEP"] == "5"
    CodeQADataset([object()] * 4, batch_size=2).get_batch(3)
    assert os.environ["CODEQA_TRAIN_STEP"] == "3"
