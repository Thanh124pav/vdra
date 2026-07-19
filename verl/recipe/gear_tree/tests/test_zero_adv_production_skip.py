"""PLAN.md E1: production zero-advantage behavior stays dense.

* The diagnostic predicate keys on the EXACT training advantage that is
  broadcast into the policy ``advantages`` tensor, never on ``pav_advantage``.
* A valid all-zero sampled batch must still follow the normal actor path.
* Missing advantages fail explicitly instead of being interpreted as zero.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe.gear_tree.replay_buffer import (
    GearTreeReplayBuffer,
    batch_has_zero_learning_signal,
    reserve_replay_edges,
)


def _edge(edge_id: str, advantage: float) -> dict:
    return {
        "edge_id": edge_id,
        "question_id": "q0",
        "query_token_ids": [1],
        "response_token_ids": [2, 3],
        "actor_shifted_log_probs": [-0.1, -0.2],
        "advantage": advantage,
        "value": 0.5,
        "reward": 0.5,
    }


class TestZeroSignalPredicate:
    def test_all_zero_batch_is_flagged(self):
        edges = [_edge(f"e{i}", 0.0) for i in range(4)]
        assert batch_has_zero_learning_signal(edges) is True

    def test_mixed_batch_is_not_flagged(self):
        edges = [_edge("e0", 0.0), _edge("e1", 0.3)]
        assert batch_has_zero_learning_signal(edges) is False

    def test_all_nonzero_batch_is_not_flagged(self):
        edges = [_edge(f"e{i}", 1.0) for i in range(4)]
        assert batch_has_zero_learning_signal(edges) is False

    def test_empty_batch_is_not_flagged(self):
        assert batch_has_zero_learning_signal([]) is False

    def test_uses_training_advantage_not_pav(self):
        # Diagnostic pav_advantage != 0 must not hide that the actual
        # training advantage is 0.
        edge = _edge("e0", 0.0)
        edge["prover_advantage"] = 1.0
        edge["pav_advantage"] = 1.0
        assert batch_has_zero_learning_signal([edge]) is True

    def test_missing_advantage_raises(self):
        edge = _edge("e0", 0.0)
        del edge["advantage"]
        with pytest.raises(ValueError, match="missing training advantage"):
            batch_has_zero_learning_signal([edge])


class TestDenseFlowWithReplayBuffer:
    def test_all_zero_reservation_is_valid_data_not_auto_consumed(self):
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=512,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=1000,
            sampling_seed=0,
        )
        buf.add(
            [_edge(f"e{i}", 0.0) for i in range(8)],
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert batch_has_zero_learning_signal(reservation.edges)
        assert len(reservation.edges) == 8
        assert len(buf) == 8
        buf.rollback(reservation)
        again = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert len(again.edges) == 8


class TestExtractionFilterUsesTrainingAdvantage:
    def _tree(self, child_rewards):
        # Minimal 1-parent tree for extract_edges_from_tree. With
        # spo/rloo, advantage = child reward - parent reward, so a child
        # reward equal to the root reward (0.0) gives training advantage 0.
        children = []
        for i, reward in enumerate(child_rewards):
            children.append(
                {
                    "text": f"c{i}",
                    "token_ids": [10 + i],
                    "response_token_ids": [10 + i],
                    "actor_shifted_log_probs": [-0.1],
                    "reward": reward,
                    "children": [],
                    "gear_segment_id": f"seg-c{i}",
                }
            )
        return {
            "text": "root",
            "full_text": "root",
            "full_token_ids": [1, 2],
            "token_ids": [1, 2],
            "reward": 0.0,
            "children": children,
            "gear_segment_id": "seg-root",
            "_request_object": {"_treetune__idx": 0},
        }

    def test_sparse_mode_drops_only_zero_training_advantage_rows(self):
        from recipe.gear_tree.tree_advantage import extract_edges_from_tree

        tree = self._tree([0.0, 1.0, 1.0])
        dense = extract_edges_from_tree(
            dict(tree), only_adv_greater_than_zero=False
        )
        sparse = extract_edges_from_tree(
            dict(self._tree([0.0, 1.0, 1.0])), only_adv_greater_than_zero=True
        )
        assert len(dense) == 3
        # Sparse mode keeps exactly the rows whose TRAINING advantage is
        # non-zero (the same scalar tensorized into `advantages`).
        assert len(sparse) == len(
            [e for e in dense if float(e["advantage"]) != 0.0]
        )
        for e in sparse:
            assert float(e["advantage"]) != 0.0

    def test_denominator_counts_survive_sparse_filtering(self):
        from recipe.gear_tree.tree_advantage import extract_edges_from_tree

        sparse = extract_edges_from_tree(
            dict(self._tree([0.0, 1.0, 1.0])), only_adv_greater_than_zero=True
        )
        # tree_total_segment_count still counts every realized segment.
        assert sparse and all(
            int(e["tree_total_segment_count"]) == 3 for e in sparse
        )


class TestProductionWiring:
    def test_trainer_does_not_shortcut_all_zero_batches(self):
        source = (
            Path(__file__).resolve().parents[1] / "gear_ray_trainer.py"
        ).read_text()
        assert "batch_has_zero_learning_signal" not in source
        assert "training/all_zero_batch_skipped" not in source

    def test_replay_validation_runs_before_tensorization(self):
        source = (
            Path(__file__).resolve().parents[1] / "gear_ray_trainer.py"
        ).read_text()
        validate_idx = source.index("replay_batch_metrics = self._update_manifest_from_replay_batch(")
        tensorize_idx = source.index("edge_batch = self._edges_to_update_batch(sampled_edges")
        actor_idx = source.index("update_actor(edge_batch)")
        assert validate_idx < tensorize_idx < actor_idx

    def test_worker_default_is_dense(self):
        source = (
            Path(__file__).resolve().parents[1] / "gear_tree_worker.py"
        ).read_text()
        assert 'gt.get("only_adv_greater_than_zero", False)' in source

    def test_extraction_filter_no_longer_uses_pav(self):
        source = (
            Path(__file__).resolve().parents[1] / "tree_advantage.py"
        ).read_text()
        assert "only_adv_greater_than_zero or pav_advantage" not in source
        assert "only_adv_greater_than_zero or advantage != 0" in source
