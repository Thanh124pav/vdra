"""Numeric parity: ``compute_policy_loss_treetune`` must reproduce treetune's
``PPOTrainer._compute_actor_loss`` PPO-clip core bit-for-bit.

We inline the reference math directly from ppo_trainer.py:1069-1166 (the parts
that depend only on the input tensors, i.e. excluding the model forward pass,
forward-KL and ref-KL which are disabled by default for tree configs). Both the
reference and the ported loss consume the same ``(old_log_prob, log_prob,
advantages, response_mask)`` tensors.

Run:
    PYTHONPATH=verl python -m pytest verl/recipe/gear_tree/tests/test_policy_loss_parity.py -q
"""

import torch

from recipe.gear_tree.policy_loss import compute_policy_loss_treetune


class _Cfg:
    """Minimal stand-in for verl ActorConfig (only what the loss reads)."""

    def __init__(self, clip_ratio=0.2, use_prob_mask=True, ratio_threshold=10.0):
        self.clip_ratio = clip_ratio
        self._d = {"use_prob_mask": use_prob_mask, "ratio_threshold": ratio_threshold}

    def get(self, k, default=None):
        return self._d.get(k, default)


def _masked_mean(values, mask):
    return (values * mask).sum() / mask.sum()


def _reference_loss(old_logprobs, logprobs, advantages, response_mask,
                    cliprange=0.2, use_prob_mask=True, ratio_threshold=10.0):
    """Verbatim reference from treetune ppo_trainer.py:1070-1160."""
    action_mask = response_mask
    if use_prob_mask:
        prob_mask = torch.exp(old_logprobs) < 0.9
        action_mask = action_mask.bool() & prob_mask
    action_mask = action_mask.to(advantages.dtype)

    log_ratio = (logprobs - old_logprobs) * action_mask
    log_ratio_clamped = torch.clamp(log_ratio, -10.0, 10.0)
    ratio = torch.exp(log_ratio_clamped)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_losses = torch.max(pg_losses1, pg_losses2)
    pg_loss = _masked_mean(pg_losses, action_mask)

    avg_ratio = _masked_mean(ratio, action_mask)
    if avg_ratio.item() > ratio_threshold:
        pg_loss = pg_loss * 0.0
    return pg_loss


def _rand_batch(seed, b=4, t=7):
    g = torch.Generator().manual_seed(seed)
    old = torch.randn(b, t, generator=g) * 0.5 - 0.3   # log-probs (<=0-ish)
    old = torch.clamp(old, max=-0.01)
    new = old + torch.randn(b, t, generator=g) * 0.2
    adv = torch.randn(b, t, generator=g)
    mask = (torch.rand(b, t, generator=g) > 0.2).float()
    mask[:, 0] = 1.0  # ensure non-empty
    return old, new, adv, mask


def test_ppo_clip_parity_default():
    for seed in range(6):
        old, new, adv, mask = _rand_batch(seed)
        ref = _reference_loss(old, new, adv, mask)
        got, pg_clipfrac, approx_kl, pg_clipfrac_lower = compute_policy_loss_treetune(
            old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
            config=_Cfg(),
        )
        assert torch.allclose(got, ref, atol=0, rtol=0), (seed, got.item(), ref.item())


def test_prob_mask_off_parity():
    old, new, adv, mask = _rand_batch(11)
    ref = _reference_loss(old, new, adv, mask, use_prob_mask=False)
    got, pg_clipfrac, approx_kl, pg_clipfrac_lower = compute_policy_loss_treetune(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
        config=_Cfg(use_prob_mask=False),
    )
    assert torch.allclose(got, ref, atol=0, rtol=0)


def test_ratio_threshold_skip_zeros_loss():
    # Force a huge ratio so the batch is skipped -> loss exactly 0.
    b, t = 2, 4
    old = torch.full((b, t), -5.0)
    new = torch.full((b, t), 5.0)          # log_ratio ~ +10 -> ratio ~ e^10
    adv = torch.ones(b, t)
    mask = torch.ones(b, t)
    got, pg_clipfrac, approx_kl, pg_clipfrac_lower = compute_policy_loss_treetune(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask,
        config=_Cfg(ratio_threshold=10.0),
    )
    assert got.item() == 0.0
    assert pg_clipfrac_lower.item() == 0.0


def test_differs_from_no_prob_mask_when_high_prob_tokens_exist():
    # A token with old prob >= 0.9 (old_logprob ~ 0) should be excluded by prob mask.
    old = torch.tensor([[-0.01, -2.0, -2.0, -2.0]])  # first token prob ~0.99
    new = old + 0.1
    adv = torch.tensor([[5.0, 0.1, 0.1, 0.1]])       # big adv on the masked token
    mask = torch.ones(1, 4)
    with_mask, pg_clipfrac, approx_kl, pg_clipfrac_lower = compute_policy_loss_treetune(
        old_log_prob=old, log_prob=new, advantages=adv, response_mask=mask, config=_Cfg(),
    )
    without_mask = _reference_loss(old, new, adv, mask, use_prob_mask=False)
    assert not torch.allclose(with_mask, without_mask)
