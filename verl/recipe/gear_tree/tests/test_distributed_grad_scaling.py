"""Distributed gradient-scaling parity for the batch-slot mean (ablation).

This algebra check covers the labeled ``batch_slot_mean_ablation`` loss whose
distributed scaling is a preserved contract; canonical tree-segment-mean
distributed parity is a Hard-stage (H1) verification item.

``ppo_mini_batch_size=128`` is the GLOBAL optimizer batch size. Multi-rank
sharding must reproduce the single-rank 128-row reference gradient after the
DDP/FSDP reducer averages gradients across ranks.

The batch-slot mean loss is

    L_B^{local} = sum(rows_on_this_rank) / N_B_global.

If the reducer averages gradients across ``W`` ranks, then

    grad = (1/W) * sum_r grad_r
         = (1/W) * sum_r ( d/dtheta sum(rows_r) / N_B_global )
         = ( sum_all_rows / (W * N_B_global) )
         = grad_single_rank / W.

To match the single-rank reference we therefore multiply the local loss by
``W`` OR use gradient sum instead of average. This test proves the
equivalence numerically without spawning ``torch.distributed`` processes:

  * baseline: one rank sees all 128 rows, N_B = 128 → grad_ref.
  * sharded:  two "ranks" each see 64 rows with N_B = 128 (global), local
    losses summed (== gradient sum reduction) → should equal grad_ref.
  * sharded avg: two "ranks" each see 64 rows with N_B = 128 and local
    losses averaged → gives grad_ref/2 UNLESS we compensate by multiplying
    the local loss by W. This test documents that trap.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from recipe.gear_tree.policy_loss import compute_policy_loss_vdra_segment_mean
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def _actor_cfg(reduction: str = "mean") -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=32,
        policy_loss=PolicyLossConfig(
            loss_mode="vdra_segment_mean_ppo",
            segment_token_reduction=reduction,
            batch_slot_mean_ablation=True,
        ),
    )


def _fake_rows(n_rows: int, max_len: int = 6, seed: int = 0):
    torch.manual_seed(seed)
    response_mask = torch.zeros((n_rows, max_len))
    active_lens = torch.randint(1, max_len + 1, (n_rows,))
    for i, k in enumerate(active_lens.tolist()):
        response_mask[i, :k] = 1.0
    old_log_prob = torch.full((n_rows, max_len), -0.2)
    log_prob = torch.full((n_rows, max_len), -0.2)
    advantages = torch.randn(n_rows, max_len) * 0.5
    return old_log_prob, log_prob, advantages, response_mask


def _single_rank_grad(cfg, rows):
    old, lp, adv, mask = rows
    n_rows = old.shape[0]
    theta = nn.Parameter(torch.zeros(1))
    loss = compute_policy_loss_vdra_segment_mean(
        old_log_prob=old,
        log_prob=lp + theta,
        advantages=adv,
        response_mask=mask,
        config=cfg,
        original_optimizer_batch_slot_count=n_rows,
    )[0]
    loss.backward()
    return theta.grad.detach().clone()


def _sharded_grad_summed(cfg, rows, world_size: int):
    old, lp, adv, mask = rows
    n_rows = old.shape[0]
    per_rank = n_rows // world_size
    theta = nn.Parameter(torch.zeros(1))
    for r in range(world_size):
        s = slice(r * per_rank, (r + 1) * per_rank)
        local_loss = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old[s],
            log_prob=lp[s] + theta,
            advantages=adv[s],
            response_mask=mask[s],
            config=cfg,
            original_optimizer_batch_slot_count=n_rows,  # GLOBAL N_B
        )[0]
        # Emulate a gradient-sum reduction across ranks (each rank calls
        # backward, gradients accumulate on theta).
        local_loss.backward()
    return theta.grad.detach().clone()


def _sharded_grad_averaged(cfg, rows, world_size: int):
    """Emulate DDP/FSDP average reduction: divide local loss by W before
    backward. Without world-size compensation this gives grad_ref / W.
    """
    old, lp, adv, mask = rows
    n_rows = old.shape[0]
    per_rank = n_rows // world_size
    theta = nn.Parameter(torch.zeros(1))
    for r in range(world_size):
        s = slice(r * per_rank, (r + 1) * per_rank)
        local_loss = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old[s],
            log_prob=lp[s] + theta,
            advantages=adv[s],
            response_mask=mask[s],
            config=cfg,
            original_optimizer_batch_slot_count=n_rows,
        )[0]
        (local_loss / world_size).backward()
    return theta.grad.detach().clone()


class TestSingleRankVsShardedSum:
    """PLAN.md P0.6 canonical equivalence:
    sharded (gradient sum) == single rank.
    """

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test_gradient_sum_matches_single_rank(self, reduction, world_size):
        cfg = _actor_cfg(reduction)
        rows = _fake_rows(128)
        grad_ref = _single_rank_grad(cfg, rows)
        grad_shard = _sharded_grad_summed(cfg, rows, world_size=world_size)
        assert torch.allclose(grad_ref, grad_shard, atol=1e-6)


class TestAverageReductionRegressionGuard:
    """PLAN.md P0.6: uncompensated averaging divides by W — the trainer must
    know this and either use gradient sum or multiply local loss by W.
    """

    def test_uncompensated_average_underscales_by_world_size(self):
        cfg = _actor_cfg("mean")
        rows = _fake_rows(128)
        grad_ref = _single_rank_grad(cfg, rows)
        grad_avg = _sharded_grad_averaged(cfg, rows, world_size=2)
        # Confirm the trap: averaged reducer without compensation gives half.
        assert torch.allclose(grad_ref / 2, grad_avg, atol=1e-6)

    def test_average_reduction_needs_world_size_compensation(self):
        """PLAN.md P0.6: multiply local loss by W restores parity when the
        reducer averages gradients. Prove the identity holds numerically.
        """
        cfg = _actor_cfg("mean")
        rows = _fake_rows(128)
        grad_ref = _single_rank_grad(cfg, rows)
        # Multiply each local loss by W before dividing by W in the reducer.
        old, lp, adv, mask = rows
        theta = nn.Parameter(torch.zeros(1))
        world = 4
        per_rank = 128 // world
        for r in range(world):
            s = slice(r * per_rank, (r + 1) * per_rank)
            local_loss = compute_policy_loss_vdra_segment_mean(
                old_log_prob=old[s],
                log_prob=lp[s] + theta,
                advantages=adv[s],
                response_mask=mask[s],
                config=cfg,
                original_optimizer_batch_slot_count=128,
            )[0]
            # Compensate: * world, then reducer / world → net factor 1.
            ((local_loss * world) / world).backward()
        assert torch.allclose(grad_ref, theta.grad, atol=1e-6)


class TestScorerTopologyContract:
    """PLAN.md P0.6: canonical smoke config must resolve one valid topology."""

    def test_same_server_mode_accepted(self):
        from recipe.gear_tree.scorer_verification import resolve_endpoints

        uses, rollout, scorer = resolve_endpoints(
            {
                "strict_vdra": True,
                "scorer_uses_rollout_server": True,
                "rollout_api_base": None,
                "scorer_api_base": "http://127.0.0.1:8000/v1",
            }
        )
        assert uses is True
        assert scorer == "http://127.0.0.1:8000/v1"

    def test_same_server_with_conflicting_rollout_api_rejected(self):
        from recipe.gear_tree.scorer_verification import resolve_endpoints

        with pytest.raises(ValueError, match="scorer_uses_rollout_server"):
            resolve_endpoints(
                {
                    "strict_vdra": True,
                    "scorer_uses_rollout_server": True,
                    "rollout_api_base": "http://another:9000/v1",
                    "scorer_api_base": "http://127.0.0.1:8000/v1",
                }
            )

    def test_two_endpoint_mode_requires_both_in_strict(self):
        from recipe.gear_tree.scorer_verification import resolve_endpoints

        with pytest.raises(ValueError, match="rollout_api_base"):
            resolve_endpoints(
                {
                    "strict_vdra": True,
                    "scorer_uses_rollout_server": False,
                    "rollout_api_base": None,
                    "scorer_api_base": "http://127.0.0.1:8000/v1",
                }
            )
        with pytest.raises(ValueError, match="scorer_api_base"):
            resolve_endpoints(
                {
                    "strict_vdra": True,
                    "scorer_uses_rollout_server": False,
                    "rollout_api_base": "http://rollout:9000/v1",
                    "scorer_api_base": None,
                }
            )

    def test_non_strict_allows_missing_endpoints(self):
        from recipe.gear_tree.scorer_verification import resolve_endpoints

        uses, rollout, scorer = resolve_endpoints(
            {
                "strict_vdra": False,
                "scorer_uses_rollout_server": False,
                "rollout_api_base": None,
                "scorer_api_base": None,
            }
        )
        assert uses is False
        assert rollout is None and scorer is None

    def test_strict_none_version_raises(self):
        from recipe.gear_tree.scorer_verification import fetch_rollout_weight_version

        def _returns_none(*a, **kw):
            return None

        with pytest.raises(RuntimeError, match="server-reported weight"):
            fetch_rollout_weight_version(
                {
                    "strict_vdra": True,
                    "scorer_uses_rollout_server": True,
                    "scorer_api_base": "http://x/v1",
                },
                fetch_fn=_returns_none,
            )

    def test_non_strict_none_version_returns_none(self):
        from recipe.gear_tree.scorer_verification import fetch_rollout_weight_version

        def _returns_none(*a, **kw):
            return None

        got = fetch_rollout_weight_version(
            {
                "strict_vdra": False,
                "scorer_uses_rollout_server": True,
                "scorer_api_base": "http://x/v1",
            },
            fetch_fn=_returns_none,
        )
        assert got is None
