"""Treetune-faithful PPO policy loss, registered into verl's policy-loss registry.

Ports ``PPOTrainer._compute_actor_loss`` from
``treetune/trainers/ppo_trainer.py`` (lines 1069-1166) **exactly**, so that the
tree-family algorithms (SPO / TreeRL / TreePO / GEAR-*) keep identical PPO
numerics after moving off DeepSpeed.

Differences vs verl's built-in ``vanilla`` loss that this preserves:
  * ``use_prob_mask``: tokens whose *old* prob >= 0.9 are dropped from the loss
    and from the mean (treetune ppo_trainer.py:1072-1074).
  * log-ratio is masked then clamped to **±10** before ``exp`` (not ±20), and
    there is **no dual-clip** (ppo_trainer.py:1113-1125).
  * ``ratio_threshold`` skip: if mean(ratio) over the action mask exceeds the
    threshold (default 10), the whole batch loss is zeroed (ppo_trainer.py:1155-1160).
  * mean is ``masked_mean`` over the *prob-masked* action mask, i.e. token-mean
    over kept tokens (treetune utils.py:239-245).

KL handling: for the tree-family configs treetune runs the precomputed-advantage
path, which uses **neither** KL-in-reward (GAE branch only) **nor** KL-in-loss by
default (``kl_penalty_loss_type=None``, ``forward_kl_penalty_coef=0``). So this
loss implements only the PPO-clip core. To reproduce the optional KL-in-loss
ablation, enable verl's actor ``use_kl_loss`` with ``kl_loss_type`` and
``kl_loss_coef`` (dp_actor adds it separately) and set
``loss_agg_mode='seq-mean-token-sum'`` to match treetune's ``.sum(1).mean()``.

Register: importing this module runs the decorator. Select with
``actor_rollout_ref.actor.policy_loss.loss_mode=treetune_ppo``.
"""

from typing import TYPE_CHECKING, Optional

import torch

from verl.trainer.ppo.core_algos import register_policy_loss

if TYPE_CHECKING:  # ActorConfig is only used as a type hint; avoid a hard import.
    from verl.workers.config import ActorConfig


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean with a finite differentiable zero for empty masks."""

    mask = mask.to(dtype=values.dtype)
    numerator = (values * mask).sum()
    denominator = mask.sum()
    if denominator.item() == 0:
        return numerator * 0.0
    return numerator / denominator


def _weighted_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over valid tokens, optionally normalized by positive edge weights."""

    mask = mask.to(dtype=values.dtype)
    if edge_weights is None:
        return _masked_mean(values, mask)
    weights = edge_weights.to(dtype=values.dtype, device=values.device)
    if weights.shape != values.shape:
        raise ValueError(
            f"edge_weights shape {tuple(weights.shape)} must match values shape {tuple(values.shape)}"
        )
    valid_weights = weights[mask.bool()]
    if valid_weights.numel() and (
        not torch.isfinite(valid_weights).all().item()
        or torch.any(valid_weights <= 0).item()
    ):
        raise ValueError("edge_weights must be finite and strictly positive on valid tokens")
    weighted_mask = mask * weights
    numerator = (values * weighted_mask).sum()
    denominator = weighted_mask.sum()
    if denominator.item() == 0:
        return numerator * 0.0
    return numerator / denominator


@register_policy_loss("treetune_ppo")
def compute_policy_loss_treetune(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: "Optional[ActorConfig]" = None,
    rollout_is_weights: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO-clip loss identical to treetune ``_compute_actor_loss`` (PPO-clip core).

    Shapes: all of ``old_log_prob``, ``log_prob``, ``advantages``,
    ``response_mask`` are ``(batch_size, response_length)``.
    """
    assert config is not None

    # treetune hyper-params (defaults match PPOHParams in ppo_trainer.py).
    cliprange = float(config.clip_ratio)              # PPOHParams.cliprange = 0.2
    use_prob_mask = bool(config.get("use_prob_mask", True))
    ratio_threshold = float(config.get("ratio_threshold", 10.0))

    # --- action mask (optionally prob-masked) : ppo_trainer.py:1070-1074 ---
    action_mask = response_mask
    if use_prob_mask:
        prob_mask = torch.exp(old_log_prob) < 0.9
        action_mask = action_mask.bool() & prob_mask
    action_mask = action_mask.to(dtype=advantages.dtype)

    # --- PPO-clip loss : ppo_trainer.py:1113-1126 ---
    log_ratio = (log_prob - old_log_prob) * action_mask
    log_ratio_clamped = torch.clamp(log_ratio, -10.0, 10.0)
    ratio = torch.exp(log_ratio_clamped)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_losses = torch.max(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = _weighted_masked_mean(pg_losses, action_mask, edge_weights=edge_weights)

    # --- ratio-threshold skip : ppo_trainer.py:1153-1160 ---
    avg_ratio = _masked_mean(ratio, action_mask)
    is_skipped = False
    if avg_ratio.item() > ratio_threshold:
        pg_loss = pg_loss * 0.0
        is_skipped = True

    # --- metrics : ppo_trainer.py:1162-1172 ---
    pg_clipfrac = _masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)
    approx_kl = 0.5 * _masked_mean((log_prob - old_log_prob) ** 2, action_mask)
    policy_kl = _masked_mean(old_log_prob - log_prob, action_mask)

    _ = policy_kl
    _ = is_skipped
    pg_clipfrac_lower = torch.zeros((), dtype=pg_loss.dtype, device=pg_loss.device)
    return pg_loss, pg_clipfrac, approx_kl, pg_clipfrac_lower
