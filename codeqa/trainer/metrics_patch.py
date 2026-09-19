"""Extra optimizer-side metrics per step: surrogate loss, importance ratios, advantage stats.

Tinker runs the loss on its servers and returns only per-token training logprobs, so the cookbook never logs a loss.
We wrap `tinker_cookbook.rl.train.compute_kl_sample_train`, the one place that sees sampling logprobs, training
logprobs and advantages together, and add:

  optim/loss                 -mean_t(ratio_t * A_t) over action tokens: the importance-sampling policy-gradient
                             surrogate (what the optimizer minimises). Advantages are group-centered, so it hovers near 0
                             by construction; its drift and sign flips are informative, its level is not.
  optim/loss_abs             mean_t |ratio_t * A_t|: magnitude of the learning signal (0 = nothing to learn from).
  optim/advantage_std        std of per-token advantages; collapses when groups stop disagreeing.
  optim/frac_tokens_with_advantage  share of action tokens with A != 0.
  optim/importance_ratio_mean|max   exp(train_logp - sample_logp); ~1 in sync training, drifts when off-policy.
  optim/clip_fraction        share of action tokens with |ratio - 1| > 0.2 (PPO-style clip band).
  optim/nll                  -mean sample logprob of action tokens (the policy's own confidence; sharpening = falling).
  optim/action_tokens        count.

Installed by `codeqa.trainer.run` before `train.main`; a no-op if the cookbook's function signature changes.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)
CLIP_EPS = 0.2


def extra_metrics(data_D: list[Any], training_logprobs_D: list[Any]) -> dict[str, float]:
    import torch
    ratios, advs, objs, nll = [], [], [], []
    for datum, train_lp in zip(data_D, training_logprobs_D):
        inp = datum.loss_fn_inputs
        samp = inp["logprobs"].to_torch()
        adv = inp["advantages"].to_torch()
        mask = inp["mask"].to_torch() > 0
        tl = train_lp if isinstance(train_lp, torch.Tensor) else torch.as_tensor(train_lp)
        n = min(len(samp), len(adv), len(mask), len(tl))
        samp, adv, mask, tl = samp[:n], adv[:n], mask[:n], tl[:n]
        if not mask.any():
            continue
        r = torch.exp((tl - samp)[mask]).float()
        a = adv[mask].float()
        ratios.append(r); advs.append(a); objs.append(r * a); nll.append(-samp[mask].float())
    if not ratios:
        return {}
    r = torch.cat(ratios); a = torch.cat(advs); o = torch.cat(objs); s = torch.cat(nll)
    return {
        "optim/loss": float(-o.mean()),
        "optim/loss_abs": float(o.abs().mean()),
        "optim/advantage_std": float(a.std()) if len(a) > 1 else 0.0,
        "optim/frac_tokens_with_advantage": float((a != 0).float().mean()),
        "optim/importance_ratio_mean": float(r.mean()),
        "optim/importance_ratio_max": float(r.max()),
        "optim/clip_fraction": float(((r - 1).abs() > CLIP_EPS).float().mean()),
        "optim/nll": float(s.mean()),
        "optim/action_tokens": float(len(a)),
    }


def install() -> bool:
    """Wrap the cookbook hook in place. Returns True when installed."""
    try:
        from tinker_cookbook.rl import train as cb_train
    except Exception as e:  # noqa: BLE001
        logger.warning("metrics_patch: cookbook not importable: %s", e)
        return False
    original = getattr(cb_train, "compute_kl_sample_train", None)
    if original is None or getattr(original, "_codeqa_patched", False):
        return original is not None

    def wrapped(data_D, training_logprobs_D):
        out = original(data_D, training_logprobs_D)
        try:
            out.update(extra_metrics(data_D, training_logprobs_D))
        except Exception as e:  # noqa: BLE001 - never break training over a metric
            logger.warning("metrics_patch: extra metrics failed: %s", e)
        return out

    wrapped._codeqa_patched = True  # type: ignore[attr-defined]
    cb_train.compute_kl_sample_train = wrapped
    return True
