"""PLAN.md P0.4 — batch-slot mean N_B normalization tests.

The canonical VDRA main loss for one optimizer batch of ``N_B`` selected
segment slots is

    L_B = (1/N_B) * sum_{s in retained(B)} L_s^r,

where ``r ∈ {mean, sum}``. Microbatch splits must NOT change the denominator:
each microbatch contributes ``sum(rows_in_mb) / N_B`` and the accumulated
gradient across the full mini-batch equals the direct 128-row reference.

These tests exercise ``compute_policy_loss_vdra_segment_mean`` in the
batch-slot mean path (``original_optimizer_batch_slot_count`` set), including:

* full-vs-split gradient parity for mean AND sum;
* row permutation invariance;
* microbatch fragmentation into 2, 4, 8 pieces gives the same gradient;
* mean vs sum are numerically distinct on non-uniform active lengths;
* duplicating identical active tokens leaves a mean row loss unchanged and
  doubles a sum row loss.
"""

from __future__ import annotations

from typing import List

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from recipe.gear_tree.policy_loss import compute_policy_loss_vdra_segment_mean


def _actor_cfg(reduction: str = "mean") -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=32,
        policy_loss=PolicyLossConfig(
            loss_mode="vdra_segment_mean_ppo",
            segment_token_reduction=reduction,
        ),
    )


def _fake_rows(n_rows: int, max_len: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    response_mask = torch.zeros((n_rows, max_len))
    active_lens = torch.randint(1, max_len + 1, (n_rows,))
    for i, k in enumerate(active_lens.tolist()):
        response_mask[i, :k] = 1.0
    # exp(-0.2) ≈ 0.82 < 0.9 so treetune's use_prob_mask keeps active tokens.
    old_log_prob = torch.full((n_rows, max_len), -0.2)
    log_prob = torch.full((n_rows, max_len), -0.2)
    advantages = torch.randn(n_rows, max_len) * 0.5
    return old_log_prob, log_prob, advantages, response_mask


def _loss_from_batch(
    rows_slice, cfg, *, n_b: int
):
    old_log_prob, log_prob, advantages, response_mask = rows_slice
    return compute_policy_loss_vdra_segment_mean(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=cfg,
        original_optimizer_batch_slot_count=n_b,
    )[0]


class TestFullVsSplit:
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("splits", [1, 2, 4, 8])
    def test_gradient_parity(self, reduction, splits):
        n_rows = 128
        cfg = _actor_cfg(reduction)

        # Direct 128-row reference.
        theta_ref = nn.Parameter(torch.zeros(1))
        old_log_prob, log_prob, advantages, response_mask = _fake_rows(n_rows)
        # log_prob depends on theta so we can differentiate loss w.r.t. it.
        lp = log_prob + theta_ref
        loss_ref = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=lp,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg,
            original_optimizer_batch_slot_count=n_rows,
        )[0]
        loss_ref.backward()
        grad_ref = theta_ref.grad.detach().clone()

        # Split path with the same denominator.
        theta = nn.Parameter(torch.zeros(1))
        mb_size = n_rows // splits
        for k in range(splits):
            s = slice(k * mb_size, (k + 1) * mb_size)
            lp_k = log_prob[s] + theta
            loss_k = compute_policy_loss_vdra_segment_mean(
                old_log_prob=old_log_prob[s],
                log_prob=lp_k,
                advantages=advantages[s],
                response_mask=response_mask[s],
                config=cfg,
                original_optimizer_batch_slot_count=n_rows,
            )[0]
            loss_k.backward()
        grad_split = theta.grad.detach().clone()
        assert torch.allclose(grad_ref, grad_split, atol=1e-6), (
            f"{reduction}/{splits}: grad_ref={grad_ref} vs {grad_split}"
        )

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_row_permutation_invariance(self, reduction):
        n_rows = 128
        cfg = _actor_cfg(reduction)
        old_log_prob, log_prob, advantages, response_mask = _fake_rows(n_rows)

        theta_a = nn.Parameter(torch.zeros(1))
        loss_a = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob + theta_a,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg,
            original_optimizer_batch_slot_count=n_rows,
        )[0]
        loss_a.backward()

        perm = torch.randperm(n_rows)
        theta_b = nn.Parameter(torch.zeros(1))
        loss_b = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob[perm],
            log_prob=log_prob[perm] + theta_b,
            advantages=advantages[perm],
            response_mask=response_mask[perm],
            config=cfg,
            original_optimizer_batch_slot_count=n_rows,
        )[0]
        loss_b.backward()
        assert torch.allclose(theta_a.grad, theta_b.grad, atol=1e-6)


class TestModeDistinction:
    """PLAN.md P0.4 mode-specific checks."""

    def _one_row(self, active_len: int, max_len: int = 8):
        response_mask = torch.zeros((1, max_len))
        response_mask[0, :active_len] = 1.0
        old_log_prob = torch.full((1, max_len), -0.2)
        log_prob = torch.full((1, max_len), -0.2)
        advantages = torch.full((1, max_len), 0.5)
        return old_log_prob, log_prob, advantages, response_mask

    def test_mean_row_loss_unchanged_by_active_token_duplication(self):
        cfg = _actor_cfg("mean")
        loss_short = compute_policy_loss_vdra_segment_mean(
            *self._one_row(2), config=cfg, original_optimizer_batch_slot_count=1
        )[0]
        loss_long = compute_policy_loss_vdra_segment_mean(
            *self._one_row(4), config=cfg, original_optimizer_batch_slot_count=1
        )[0]
        assert torch.allclose(loss_short, loss_long, atol=1e-6), (
            f"mean row loss should be independent of active-token count: "
            f"short={loss_short.item()} long={loss_long.item()}"
        )

    def test_sum_row_loss_scales_with_active_token_count(self):
        cfg = _actor_cfg("sum")
        loss_short = compute_policy_loss_vdra_segment_mean(
            *self._one_row(2), config=cfg, original_optimizer_batch_slot_count=1
        )[0]
        loss_long = compute_policy_loss_vdra_segment_mean(
            *self._one_row(4), config=cfg, original_optimizer_batch_slot_count=1
        )[0]
        # 4 identical active tokens should give 2× the loss of 2 active tokens.
        assert torch.allclose(loss_long, 2.0 * loss_short, atol=1e-6), (
            f"sum row loss should double when active tokens double: "
            f"short={loss_short.item()} long={loss_long.item()}"
        )


class TestDenominatorContract:
    """PLAN.md P0.4: N_B is the ORIGINAL slot count, not the local row count.

    Emulate the trainer's split: sub-batches of 32 rows each share the same
    N_B = 128. Sum of partial losses must equal the direct 128-row loss.
    """

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_partial_loss_sums_equal_direct_loss(self, reduction):
        n_rows = 128
        cfg = _actor_cfg(reduction)
        rows = _fake_rows(n_rows)
        direct = _loss_from_batch(rows, cfg, n_b=n_rows)

        parts = []
        for s in range(0, n_rows, 32):
            slc = tuple(t[s : s + 32] for t in rows)
            parts.append(_loss_from_batch(slc, cfg, n_b=n_rows))
        assert torch.allclose(sum(parts), direct, atol=1e-6)

    def test_zero_slot_count_raises(self):
        cfg = _actor_cfg("mean")
        rows = _fake_rows(4)
        with pytest.raises(ValueError, match="original_optimizer_batch_slot_count"):
            _loss_from_batch(rows, cfg, n_b=0)

    def test_missing_all_denominator_sources_raises(self):
        """Neither ``original_optimizer_batch_slot_count`` nor
        ``segment_objective_weights`` nor ``(tree_group_ids,
        tree_total_segment_count)`` → hard error, no silent fallback.
        """
        cfg = _actor_cfg("mean")
        rows = _fake_rows(4)
        with pytest.raises(ValueError, match="requires either"):
            compute_policy_loss_vdra_segment_mean(
                old_log_prob=rows[0],
                log_prob=rows[1],
                advantages=rows[2],
                response_mask=rows[3],
                config=cfg,
            )
