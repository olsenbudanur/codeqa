import pytest

from codeqa.trainer.group_rewards import advantages, fill_judge_errors, group_metrics, no_answer_penalty, unique_tool_sequences


def test_one_nan_in_group_of_8_gets_zero_advantage_others_unchanged():
    healthy = [1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0]
    rewards = healthy + [0.0]                     # the env reported 0.0 for the errored sample
    errored = [False] * 7 + [True]
    adds, totals = fill_judge_errors(rewards, errored)
    adv = advantages(totals)
    assert adv[7] == pytest.approx(0.0)
    assert adv[:7] == pytest.approx(advantages(healthy))
    assert adds[:7] == [0.0] * 7 and adds[7] == pytest.approx(sum(healthy) / 7)


def test_all_errored_group_stays_constant():
    adds, totals = fill_judge_errors([0.0, 0.0, 0.0], [True, True, True])
    assert adds == [0.0, 0.0, 0.0] and totals == [0.0, 0.0, 0.0]
    m = group_metrics(totals, [True, True, True], [["grep"], ["grep"], ["read_file"]])
    assert m["group_all_judge_errors"] == 1.0 and m["judge_error_rate"] == 1.0 and m["group_reward_std"] == 0.0


def test_diversity_metric():
    assert unique_tool_sequences([["a", "b"], ["a", "b"], ["b"], []]) == 3
    m = group_metrics([1.0, 0.0], [False, False], [["a"], ["a"]])
    assert m["unique_tool_sequences_per_group"] == 1.0 and m["group_reward_std"] == 0.5


def test_no_answer_penalty_only_for_stalled_episodes():
    assert no_answer_penalty("budget", "format") == -0.1
    assert no_answer_penalty("max_turns", "format") == -0.1
    assert no_answer_penalty("answer", "format") == 0.0          # answered but over the cap / malformed: 0, not -0.1
    assert no_answer_penalty("answer", "citations") == 0.0
    assert no_answer_penalty("answer", None) == 0.0
    assert no_answer_penalty("parse_error", "format") == 0.0     # cookbook handles parse errors / overflow with its own terms
