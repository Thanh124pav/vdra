"""PLAN.md P0.C: canonical DataProto carries no float objective weights.

* ``edges_to_dataproto(loss_mode="vdra_segment_mean_ppo")`` attaches neither
  ``objective_weights`` nor ``segment_objective_weights``.
* The explicit ``vdra_node_balanced_ppo`` ablation still receives both.
* The canonical batch-slot loss/gradient is identical with or without the
  legacy weight tensors present, so deleting them cannot change training.
"""

from __future__ import annotations

import pytest
import torch

from recipe.gear_tree.policy_loss import compute_policy_loss_vdra_segment_mean
from recipe.gear_tree.tree_data import (
    compute_segment_objective_weights,
    edges_to_dataproto,
)


class _Tok:
    pad_token_id = 0
    eos_token_id = 0


def _edges(n: int = 4) -> list[dict]:
    return [
        {
            "edge_id": f"t0/e{i}",
            "tree_id": "t0",
            "parent_group_id": "t0/pg0",
            "child_segment_id": f"t0/e{i}",
            "question_id": "q0",
            "allocated_k": n,
            "sample_multiplicity": 1,
            "tree_total_segment_count": n,
            "queue_flush_id": "0",
            "queue_released_segment_count": n,
            "query_token_ids": [1, 2],
            "response_token_ids": [3, 4, 5],
            "actor_shifted_log_probs": [-0.1, -0.2, -0.3],
            "advantage": 0.5,
            "value": 0.4,
            "reward": 1.0,
        }
        for i in range(n)
    ]


def _build(loss_mode: str):
    return edges_to_dataproto(
        _edges(),
        _Tok(),
        max_prompt_length=8,
        max_response_length=4,
        loss_mode=loss_mode,
    )


class TestCanonicalBatchHasNoObjectiveWeights:
    def test_segment_mean_batch_has_neither_weight_tensor(self):
        batch = _build("vdra_segment_mean_ppo")
        assert "objective_weights" not in batch.batch
        assert "segment_objective_weights" not in batch.batch

    def test_default_loss_mode_is_canonical(self):
        batch = edges_to_dataproto(
            _edges(), _Tok(), max_prompt_length=8, max_response_length=4
        )
        assert "objective_weights" not in batch.batch
        assert "segment_objective_weights" not in batch.batch

    def test_canonical_batch_keeps_integer_identity_metadata(self):
        batch = _build("vdra_segment_mean_ppo")
        for key in (
            "tree_group_ids",
            "parent_group_ids",
            "allocated_k",
            "sample_multiplicity",
            "tree_total_segment_count",
        ):
            assert key in batch.batch, key

    def test_node_balanced_ablation_still_receives_weights(self):
        batch = _build("vdra_node_balanced_ppo")
        assert "objective_weights" in batch.batch
        assert "segment_objective_weights" in batch.batch
        assert torch.isclose(
            batch.batch["objective_weights"].sum(), torch.tensor(1.0)
        )


class TestCanonicalLossUnchangedWithoutWeights:
    def _loss_inputs(self, n: int = 4, t: int = 3):
        torch.manual_seed(0)
        old = torch.randn(n, t) * 0.1
        new = (old + torch.randn(n, t) * 0.05).requires_grad_(True)
        adv = torch.randn(n, t)
        mask = torch.ones(n, t)
        mask[1, 2] = 0.0  # uneven token lengths
        return old, new, adv, mask

    class _Cfg:
        clip_ratio = 0.2

        class policy_loss:
            segment_token_reduction = "mean"

        def get(self, key, default=None):
            return {"use_prob_mask": False}.get(key, default)

    def test_loss_and_grad_identical_with_and_without_weight_tensor(self):
        old, new, adv, mask = self._loss_inputs()
        weights = torch.tensor(
            compute_segment_objective_weights(_edges()), dtype=torch.float32
        )

        loss_without, *_ = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old,
            log_prob=new,
            advantages=adv,
            response_mask=mask,
            config=self._Cfg(),
            original_optimizer_batch_slot_count=4,
        )
        (grad_without,) = torch.autograd.grad(loss_without, new, retain_graph=True)

        loss_with, *_ = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old,
            log_prob=new,
            advantages=adv,
            response_mask=mask,
            config=self._Cfg(),
            segment_objective_weights=weights,
            original_optimizer_batch_slot_count=4,
        )
        (grad_with,) = torch.autograd.grad(loss_with, new)

        assert torch.allclose(loss_without, loss_with)
        assert torch.allclose(grad_without, grad_with)
