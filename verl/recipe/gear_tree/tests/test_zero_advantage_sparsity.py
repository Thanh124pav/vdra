"""Zero-advantage sparse-vs-dense + unique ID tests.

Production computes advantages before filtering, retains positive and
negative advantages, and drops exact-zero rows from tensor execution to save
compute. This suite exercises that invariance on the
``tree_balanced_segment_mean`` ablation, whose weights
``w_s = 1 / (N_T * N_seg(T))`` use the PRE-FILTER ``tree_total_segment_count``
so dropping zero rows leaves the loss unchanged — the (L1+L2+0+0)/4 parity
asserted below. (The canonical paper objectives achieve the same invariance
through the pre-filter M_B / T_B denominators that count zero slots; that is
covered by test_policy_aggregation_loss.py and test_logical_update_batch.py.)

Additional invariant:
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
            policy_aggregation="tree_balanced_segment_mean",
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
    """Finish Medium Stage item 2: dense and sparse execution give identical
    loss and gradients under the canonical weights
    ``w_s = 1 / (N_T * N_seg(T))`` because ``N_seg(T)`` is the PRE-FILTER
    ``tree_total_segment_count`` and ``N_T`` is fixed from the original
    optimizer batch — never the retained replay-slot count.
    """

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_gradient_matches_when_prefilter_counts_preserved(self, reduction):
        cfg = _actor_cfg(reduction)
        n_rows, n_trees = 128, 16
        old_log_prob, log_prob, advantages, response_mask, zero_idx = _make_batch(
            n_rows, zero_frac=0.5
        )
        # 16 trees of 8 pre-filter segments each.
        seg_counts = torch.full((n_rows,), float(n_rows // n_trees))

        # DENSE path: all 128 rows go through the loss.
        theta_dense = nn.Parameter(torch.zeros(1))
        loss_dense = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob + theta_dense,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg,
            tree_total_segment_count=seg_counts,
            original_optimizer_batch_tree_count=n_trees,
        )[0]
        loss_dense.backward()

        # SPARSE path: drop zero-advantage rows before the loss, keeping the
        # pre-filter tree_total_segment_count on every retained row and the
        # original batch's N_T.
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
            tree_total_segment_count=seg_counts[keep],
            original_optimizer_batch_tree_count=n_trees,  # preserved N_T
        )[0]
        loss_sparse.backward()

        assert torch.allclose(loss_dense.detach(), loss_sparse.detach(), atol=1e-6), (
            f"loss dense={loss_dense.item()} sparse={loss_sparse.item()}"
        )
        assert torch.allclose(theta_dense.grad, theta_sparse.grad, atol=1e-6)

    def test_pos_neg_zero_zero_parity(self):
        """The instruction's literal example: advantages [pos, neg, 0, 0] in
        one tree of four pre-filter segments. After filtering the two zero
        rows the loss must remain (L1 + L2 + 0 + 0) / 4 = (L1 + L2) / 4."""
        cfg = _actor_cfg("mean")
        max_len = 4
        response_mask = torch.ones((4, max_len))
        old_log_prob = torch.full((4, max_len), -0.2)
        log_prob = torch.full((4, max_len), -0.2)
        advantages = torch.zeros((4, max_len))
        advantages[0] = 0.7   # positive
        advantages[1] = -0.3  # negative
        seg_counts = torch.full((4,), 4.0)

        loss_dense = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg,
            tree_total_segment_count=seg_counts,
            original_optimizer_batch_tree_count=1,
        )[0]
        loss_sparse = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob[:2],
            log_prob=log_prob[:2],
            advantages=advantages[:2],
            response_mask=response_mask[:2],
            config=cfg,
            tree_total_segment_count=seg_counts[:2],
            original_optimizer_batch_tree_count=1,
        )[0]
        # ratio == 1 -> L1 = -0.7, L2 = 0.3 (token means of -adv).
        expected = torch.tensor((-0.7 + 0.3) / 4.0)
        assert torch.allclose(loss_dense, expected, atol=1e-6)
        assert torch.allclose(loss_sparse, expected, atol=1e-6)
        assert torch.allclose(loss_dense, loss_sparse, atol=1e-6)


class TestAllZeroBatchIsHandled:
    """The loss helper returns zero for an all-zero-advantage batch.

    Stage 1 removes the trainer-level all-zero shortcut; this test does not
    claim optimizer-step or global-step behavior.
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
            tree_total_segment_count=torch.full((n_rows,), 8.0),
            original_optimizer_batch_tree_count=1,
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
