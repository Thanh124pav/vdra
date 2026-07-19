"""PLAN.md P0.B: construction validation vs replay-batch validation split.

* A sampled batch containing only some siblings of a parent group must PASS
  replay validation and must not increment ``group_integrity_failures``.
* The full generated tree must still FAIL construction validation when it
  realized fewer children than ``allocated_k``.
"""

from __future__ import annotations

import pytest

from recipe.gear_tree.manifest_lifecycle import (
    build_run_manifest,
    update_manifest_from_generated_edges,
    update_manifest_from_replay_batch,
)
from recipe.gear_tree.tree_data import (
    validate_replay_batch,
    validate_tree_construction,
)


def _edge(
    *,
    edge_id: str,
    tree_id: str = "t0",
    parent_group_id: str = "t0/pg0",
    question_id: str = "q0",
    allocated_k: int = 6,
    generation_rollout_iteration: int = 3,
    advantage: float = 1.0,
    pruned: bool = False,
    tree_total_segment_count: int = 6,
    queue_released_segment_count: int = 6,
    queue_flush_id: str = "0",
) -> dict:
    return {
        "edge_id": edge_id,
        "tree_id": tree_id,
        "parent_group_id": parent_group_id,
        "child_segment_id": edge_id,
        "question_id": question_id,
        "allocated_k": allocated_k,
        "sample_multiplicity": 1,
        "generation_rollout_iteration": generation_rollout_iteration,
        "advantage": advantage,
        "pruned": pruned,
        "tree_total_segment_count": tree_total_segment_count,
        "queue_released_segment_count": queue_released_segment_count,
        "queue_flush_id": queue_flush_id,
        "query_token_ids": [1],
        "response_token_ids": [2, 3],
        "actor_shifted_log_probs": [-0.1, -0.2],
        "value": 0.5,
        "reward": 0.5,
    }


def _full_tree(k: int = 6, realized: int | None = None) -> list[dict]:
    realized = k if realized is None else realized
    return [
        _edge(
            edge_id=f"t0/e{i}",
            allocated_k=k,
            tree_total_segment_count=realized,
            queue_released_segment_count=realized,
        )
        for i in range(realized)
    ]


def _manifest():
    return build_run_manifest(
        tree_policy={
            "policy_aggregation": "global_segment_mean",
            "segment_token_reduction": "mean",
            "advantage_mode": "spo_local",
        },
        gear_tree_cfg={},
        actor_loss_mode="vdra_segment_mean_ppo",
    )


class TestConstructionValidation:
    def test_full_tree_passes(self):
        metrics = validate_tree_construction(_full_tree(k=6), strict_fresh_iid=True)
        assert metrics["vdra/construction_failures"] == 0
        assert metrics["vdra/group_integrity_failures"] == 0

    def test_five_of_six_realized_children_fail(self):
        with pytest.raises(ValueError, match="Tree-construction"):
            validate_tree_construction(
                _full_tree(k=6, realized=5), strict_fresh_iid=True
            )

    def test_duplicate_edge_ids_fail(self):
        tree = _full_tree(k=2)
        tree[1]["edge_id"] = tree[0]["edge_id"]
        with pytest.raises(ValueError, match="duplicate edge_id"):
            validate_tree_construction(tree, strict_fresh_iid=True)

    def test_pruned_placeholder_fails(self):
        tree = _full_tree(k=2)
        tree[0]["pruned"] = True
        with pytest.raises(ValueError, match="pruned placeholder"):
            validate_tree_construction(tree, strict_fresh_iid=True)

    def test_queue_identity_mismatch_fails(self):
        tree = _full_tree(k=2)
        for e in tree:
            e["queue_released_segment_count"] = 1  # sum 1 != total 2
        with pytest.raises(ValueError, match="queue_released_segment_count"):
            validate_tree_construction(tree, strict_fresh_iid=True)

    def test_misaligned_log_probs_fail(self):
        tree = _full_tree(k=2)
        tree[0]["actor_shifted_log_probs"] = [-0.1]
        with pytest.raises(ValueError, match="misaligned"):
            validate_tree_construction(tree, strict_fresh_iid=True)


class TestReplayBatchValidation:
    def test_two_of_six_siblings_pass(self):
        sampled = _full_tree(k=6)[:2]
        metrics = validate_replay_batch(
            sampled,
            target_edges_per_iteration=512,
            max_edges_per_question_per_iteration=33,
            max_edge_age_iterations=8,
            current_rollout_iteration=4,
            strict=True,
        )
        assert metrics["vdra/replay_batch_failures"] == 0
        assert metrics["vdra/replay_selected_edges"] == 2.0

    def test_missing_generation_iteration_fails(self):
        sampled = _full_tree(k=6)[:2]
        del sampled[0]["generation_rollout_iteration"]
        with pytest.raises(ValueError, match="generation_rollout_iteration"):
            validate_replay_batch(sampled, strict=True)

    def test_duplicate_edge_ids_fail(self):
        sampled = _full_tree(k=6)[:2]
        sampled[1]["edge_id"] = sampled[0]["edge_id"]
        with pytest.raises(ValueError, match="duplicate sampled edge_id"):
            validate_replay_batch(sampled, strict=True)

    def test_negative_age_fails(self):
        sampled = _full_tree(k=6)[:2]
        with pytest.raises(ValueError, match="age outside"):
            validate_replay_batch(
                sampled,
                max_edge_age_iterations=8,
                current_rollout_iteration=0,  # generation iteration is 3
                strict=True,
            )

    def test_expired_age_fails(self):
        sampled = _full_tree(k=6)[:2]
        with pytest.raises(ValueError, match="age outside"):
            validate_replay_batch(
                sampled,
                max_edge_age_iterations=8,
                current_rollout_iteration=11,  # age 8 >= 8
                strict=True,
            )

    def test_over_cap_fails(self):
        sampled = _full_tree(k=6)
        with pytest.raises(ValueError, match="exceeds resolved"):
            validate_replay_batch(
                sampled,
                max_edges_per_question_per_iteration=4,
                strict=True,
            )

    def test_over_target_fails(self):
        sampled = _full_tree(k=6)
        with pytest.raises(ValueError, match="exceeds"):
            validate_replay_batch(
                sampled, target_edges_per_iteration=4, strict=True
            )

    def test_missing_advantage_fails(self):
        sampled = _full_tree(k=6)[:2]
        sampled[0]["advantage"] = None
        with pytest.raises(ValueError, match="advantage"):
            validate_replay_batch(sampled, strict=True)


class TestManifestStageSplit:
    def test_partial_sample_does_not_increment_group_integrity_failures(self):
        manifest = _manifest()
        sampled = _full_tree(k=6)[:2]
        metrics = update_manifest_from_replay_batch(
            manifest,
            sampled,
            strict=True,
            target_edges_per_iteration=512,
            max_edges_per_question_per_iteration=33,
            max_edge_age_iterations=8,
            current_rollout_iteration=4,
        )
        assert manifest.group_integrity_failures == 0
        assert manifest.replay_batch_failures == 0
        assert metrics["vdra/replay_batch_failures"] == 0
        assert manifest.replay_age_uses_rollout_iteration is True

    def test_generated_partial_tree_still_fails_construction_stage(self):
        manifest = _manifest()
        with pytest.raises(ValueError, match="Tree-construction"):
            update_manifest_from_generated_edges(
                manifest, _full_tree(k=6, realized=5), strict=True
            )
        assert manifest.group_integrity_failures >= 1

    def test_construction_pass_flips_segment_bits(self):
        manifest = _manifest()
        update_manifest_from_generated_edges(
            manifest, _full_tree(k=6), strict=True
        )
        assert manifest.group_integrity_failures == 0
        assert manifest.segment_count_invariants_passed is True
        assert manifest.fresh_iid_row_count_matches_allocated_k is True

    def test_replay_failure_increments_replay_counter_only(self):
        manifest = _manifest()
        sampled = _full_tree(k=6)[:2]
        sampled[1]["edge_id"] = sampled[0]["edge_id"]
        with pytest.raises(ValueError):
            update_manifest_from_replay_batch(manifest, sampled, strict=True)
        assert manifest.replay_batch_failures == 1
        assert manifest.group_integrity_failures == 0
