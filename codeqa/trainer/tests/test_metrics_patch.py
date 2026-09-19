import math

import torch

from codeqa.trainer.metrics_patch import extra_metrics, install


class _T:
    def __init__(self, v):
        self.v = torch.tensor(v, dtype=torch.float32)

    def to_torch(self):
        return self.v


class _Datum:
    def __init__(self, logprobs, advantages, mask):
        self.loss_fn_inputs = {"logprobs": _T(logprobs), "advantages": _T(advantages), "mask": _T(mask)}


def test_extra_metrics_on_policy_group_centered():
    # two datums from one group: advantages +0.5 / -0.5 on action tokens, on-policy (train == sample logprobs)
    d = [_Datum([-1.0, -2.0, -0.5], [0.5, 0.5, 0.5], [0, 1, 1]), _Datum([-1.5, -0.7, -0.9], [-0.5, -0.5, -0.5], [0, 1, 1])]
    tl = [torch.tensor([-1.0, -2.0, -0.5]), torch.tensor([-1.5, -0.7, -0.9])]
    m = extra_metrics(d, tl)
    assert math.isclose(m["optim/loss"], 0.0, abs_tol=1e-6)            # centered advantages, ratio 1
    assert math.isclose(m["optim/loss_abs"], 0.5, abs_tol=1e-6)
    assert m["optim/importance_ratio_mean"] == 1.0 and m["optim/clip_fraction"] == 0.0
    assert m["optim/action_tokens"] == 4.0 and m["optim/frac_tokens_with_advantage"] == 1.0
    assert math.isclose(m["optim/nll"], (2.0 + 0.5 + 0.7 + 0.9) / 4, abs_tol=1e-6)


def test_extra_metrics_off_policy_ratio_and_clip():
    d = [_Datum([-1.0, -1.0], [1.0, 1.0], [1, 1])]
    tl = [torch.tensor([-1.0, -0.5])]                                  # second token: ratio e^0.5 = 1.65 > clip band
    m = extra_metrics(d, tl)
    assert m["optim/clip_fraction"] == 0.5 and m["optim/importance_ratio_max"] > 1.6
    assert m["optim/loss"] < -1.0                                        # positive advantages, ratio >= 1


def test_install_wraps_cookbook_hook_once():
    from tinker_cookbook.rl import train as cb
    assert install() and install()
    assert getattr(cb.compute_kl_sample_train, "_codeqa_patched", False)
