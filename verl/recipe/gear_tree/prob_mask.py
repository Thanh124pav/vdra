"""Shared probability-mask predicate (PLAN.md §1).

SINGLE source of truth for "is this response token active?", imported by
BOTH sides so they can never drift apart:

* the actor-side effective action mask (``policy_loss``);
* extraction-time active-token counting (``tree_advantage``).

Kept in its own dependency-light module (``math`` + ``torch`` only) so the
extraction path does not pull in the heavy ``verl.trainer.ppo.core_algos``
import chain that ``policy_loss`` needs for loss registration.
"""

from __future__ import annotations

import math

import torch

# The historical treetune default.
DEFAULT_PROBABILITY_MASK_THRESHOLD = 0.9


def probability_mask_active(
    old_log_prob, threshold: float = DEFAULT_PROBABILITY_MASK_THRESHOLD
):
    """Active iff ``exp(old_log_prob) < threshold`` — the STRICT comparison.

    Evaluated in LOG SPACE for numerical stability
    (``old_log_prob < log(threshold)``), which is exactly equivalent because
    ``exp``/``log`` are strictly increasing and ``threshold > 0`` is enforced
    by ``PolicyLossConfig``.

    Accepts a torch tensor (returns a bool tensor) or a plain float/sequence
    (returns a bool / list of bool).
    """
    log_threshold = math.log(float(threshold))
    if torch.is_tensor(old_log_prob):
        return old_log_prob < log_threshold
    if isinstance(old_log_prob, (list, tuple)):
        return [float(lp) < log_threshold for lp in old_log_prob]
    return float(old_log_prob) < log_threshold


def count_prob_mask_active_tokens(
    old_log_probs, threshold: float = DEFAULT_PROBABILITY_MASK_THRESHOLD
) -> int:
    """Extraction-time active-token count, sharing the predicate above."""
    return int(
        sum(1 for active in probability_mask_active(old_log_probs, threshold) if active)
    )


def effective_action_mask(
    response_mask,
    old_log_prob,
    *,
    use_prob_mask: bool,
    probability_mask_threshold: float = DEFAULT_PROBABILITY_MASK_THRESHOLD,
    is_dummy=None,
):
    """Dummy-safe effective policy-gradient action mask (PLAN.md §1/§9).

        use_prob_mask=false: response_mask AND NOT dummy
        use_prob_mask=true:  response_mask AND NOT dummy
                             AND exp(old_log_prob) < threshold

    ``is_dummy`` is a per-row [batch] (or broadcastable) indicator for
    collective-safety padding rows; they are masked out EXPLICITLY here and
    never rely on the probability mask happening to drop them.
    """
    mask = response_mask.bool()
    if is_dummy is not None:
        keep = ~(is_dummy.bool())
        if keep.dim() == 1:
            keep = keep.unsqueeze(-1)
        mask = mask & keep
    if use_prob_mask:
        mask = mask & probability_mask_active(old_log_prob, probability_mask_threshold)
    return mask
