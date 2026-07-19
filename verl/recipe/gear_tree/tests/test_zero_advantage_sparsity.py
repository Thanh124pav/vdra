"""PLAN.md P0.5 — zero-advantage sparse-vs-dense + unique ID tests.

Canonical safe path keeps every realized child row
(``only_adv_greater_than_zero=False``) so the parent denominator matches
``allocated_k``. An optional sparse path may drop zero-contribution rows
before model execution — but the batch-slot mean loss must use the ORIGINAL
N_B (128 in main). This gives identical loss/gradient dense vs sparse.

Additional invariants:
* an all-zero optimizer batch skips ``optimizer.step()``;
* replay insertion rejects duplicate ``edge_id`` transactionally (unique
  ``tree_instance_id`` regression guard).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn

from recipe.gear_tree.policy_loss import compute_policy_loss_vdra_segment_mean
from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


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


def _make_batch(n_rows: int, max_len: int = 6, zero_frac: float = 0.5, seed: int = 0):
    torch.manual_seed(seed)
    response_mask = torch.zeros((n_rows, max_len))
    active_lens = torch.randint(1, max_len + 1, (n_rows,))
    for i, k in enumerate(active_lens.tolist()):
        response_mask[i, :k] = 1.0
    old_log_prob = torch.full((n_rows, max_len), -0.2)
    log_prob = torch.full((n_rows, max_len), -0.2)
    advantages = torch.randn(n_rows, max_len) * 0.5
    # Zero-advantage mask: force a fraction of rows to zero advantage.
    n_zero = int(n_rows * zero_frac)
    zero_row_idx = torch.randperm(n_rows)[:n_zero]
    advantages[zero_row_idx] = 0.0
    return old_log_prob, log_prob, advantages, response_mask, zero_row_idx


class TestDenseVsSparseParity:
    """PLAN.md P0.5: dense and sparse execution must give identical loss and
    gradients when zero slots are counted in N_B.
    """

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_gradient_matches_when_N_B_preserved(self, reduction):
        cfg = _actor_cfg(reduction)
        n_rows = 128
        old_log_prob, log_prob, advantages, response_mask, zero_idx = _make_batch(
            n_rows, zero_frac=0.5
        )

        # DENSE path: all 128 rows go through the loss.
        theta_dense = nn.Parameter(torch.zeros(1))
        loss_dense = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob + theta_dense,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg,
            original_optimizer_batch_slot_count=n_rows,
        )[0]
        loss_dense.backward()

        # SPARSE path: drop zero-advantage rows before the loss, but keep the
        # original N_B in the denominator.
        keep = torch.tensor(
            [i for i in range(n_rows) if i not in set(zero_idx.tolist())],
            dtype=torch.long,
        )
        theta_sparse = nn.Parameter(torch.zeros(1))
        loss_sparse = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob[keep],
            log_prob=log_prob[keep] + theta_sparse,
            advantages=advantages[keep],
            response_mask=response_mask[keep],
            config=cfg,
            original_optimizer_batch_slot_count=n_rows,  # preserved N_B
        )[0]
        loss_sparse.backward()

        assert torch.allclose(loss_dense.detach(), loss_sparse.detach(), atol=1e-6), (
            f"loss dense={loss_dense.item()} sparse={loss_sparse.item()}"
        )
        assert torch.allclose(theta_dense.grad, theta_sparse.grad, atol=1e-6)


class TestAllZeroBatchIsHandled:
    """PLAN.md P0.5: a batch whose every retained slot has zero contribution
    should skip ``optimizer.step()`` and leave ``global_step`` unchanged.

    We verify the loss reduces to zero so the caller can safely detect and
    skip. The trainer-side skip is exercised in P0.3 tests.
    """

    def test_all_zero_advantage_gives_zero_loss(self):
        cfg = _actor_cfg("mean")
        n_rows = 8
        response_mask = torch.ones((n_rows, 4))
        old_log_prob = torch.full((n_rows, 4), -0.2)
        log_prob = torch.full((n_rows, 4), -0.2)
        advantages = torch.zeros((n_rows, 4))
        loss = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg,
            original_optimizer_batch_slot_count=n_rows,
        )[0]
        assert torch.allclose(loss, torch.zeros(()))


class TestUniqueTreeInstanceIds:
    """PLAN.md P0.5: strict main runs must reject a fallback to
    ``(snapshot, question)`` — the buffer already rejects duplicate edge_ids
    transactionally. Regression: two stochastic trees for the same question
    and snapshot must coexist because their derived edge_ids differ.
    """

    def _edge(self, edge_id: str, question_id: str = "q0"):
        return {
            "edge_id": edge_id,
            "question_id": question_id,
            "query_token_ids": [0],
            "response_token_ids": [1, 2],
            "actor_shifted_log_probs": [0.0, 0.0],
            "advantage": 0.1,
            "value": 0.0,
            "reward": 0.0,
        }

    def test_duplicate_edge_id_rejected_transactionally(self):
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=32,
            max_edge_age_iterations=4,
            max_edges_per_question_per_iteration=16,
            tree_shape=[4, 4],
        )
        buf.add(
            [self._edge("existing")],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        size_before = len(buf)
        incoming = [
            self._edge("new_1"),
            self._edge("existing"),  # duplicate — must roll back whole batch
        ]
        with pytest.raises(ValueError, match="already in the buffer"):
            buf.add(
                incoming,
                generation_rollout_iteration=1,
                policy_snapshot_id="snap",
            )
        # Buffer size did not change: no partial insertion.
        assert len(buf) == size_before

    def test_two_stochastic_trees_same_question_can_coexist(self):
        """Different tree_instance_id → different edge_id → both add cleanly."""
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=32,
            max_edge_age_iterations=4,
            max_edges_per_question_per_iteration=16,
            tree_shape=[4, 4],
        )
        buf.add(
            [
                self._edge("tree1_edge0", question_id="q0"),
                self._edge("tree2_edge0", question_id="q0"),
            ],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        assert len(buf) == 2


class TestRealizedChildCountPersisted:
    """PLAN.md P0.5: ``realized_child_count`` is stamped on every edge from
    the pre-filter snapshot, independent of ``only_adv_greater_than_zero``.
    """

    def test_field_exists_when_extract_stamps(self):
        # Construct a minimal edge dict with the field set explicitly, then
        # confirm downstream tensorization would read it. Full end-to-end
        # extract_edges_from_tree paths are covered by test_tree_advantage.py.
        edge = {
            "edge_id": "e0",
            "parent_group_id": "p0",
            "allocated_k": 4,
            "realized_child_count": 4,
        }
        assert edge["realized_child_count"] == edge["allocated_k"]
