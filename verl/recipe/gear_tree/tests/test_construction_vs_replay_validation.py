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

    def test_missing_advantage_key_fails(self):
        sampled = _full_tree(k=6)[:2]
        del sampled[0]["advantage"]
        with pytest.raises(ValueError, match="advantage"):
            validate_replay_batch(sampled, strict=True)

    def test_none_advantage_fails(self):
        sampled = _full_tree(k=6)[:2]
        sampled[0]["advantage"] = None
        with pytest.raises(ValueError, match="advantage"):
            validate_replay_batch(sampled, strict=True)

    def test_zero_advantage_is_valid(self):
        sampled = _full_tree(k=6)[:2]
        sampled[0]["advantage"] = 0.0
        metrics = validate_replay_batch(sampled, strict=True)
        assert metrics["vdra/replay_batch_failures"] == 0


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


def _zero_filter_tree(num_children=3, zero_mask=(False, False, True)):
    """One-parent tree; children with zero_mask=True get exactly zero
    advantage (reward equal to the root baseline)."""
    children = []
    for i in range(num_children):
        children.append(
            {
                "reward": 0.0 if zero_mask[i] else 1.0,
                "gear_segment_id": f"c{i}",
                "response_token_ids": [i + 1],
                "actor_shifted_log_probs": [-0.1],
                "leaf": True,
                "text": "x",
            }
        )
    return {
        "_request_object": {
            "_treetune__idx": "q1",
            "policy_snapshot_id": "snap",
            "rollout_iteration": 3,
            # Canonical make_tree_instance_id shape so an all-zero tree passes
            # strict manifest identity verification.
            "tree_instance_id": "snap|iter:3|q:q1|t:zf1",
        },
        "reward": 0.0,
        "full_text": "root",
        "full_token_ids": [1],
        "vdra_allocated_k": num_children,
        "vdra_queue_flush_id": 0,
        "children": children,
    }


class TestZeroFilterConstructionContract:
    """Finish Medium Stage item 1: fresh-IID construction validation must
    require ``realized_child_count == allocated_k`` and
    ``retained_row_count <= allocated_k`` — never
    ``retained_row_count == allocated_k`` — because exact-zero-advantage
    edges are intentionally removed."""

    def _extract(self, tree, summaries=None):
        from recipe.gear_tree.tree_advantage import extract_edges_from_tree

        edges = extract_edges_from_tree(
            tree,
            only_adv_greater_than_zero=True,
            strict_fresh_iid=True,
            collect_construction_summaries=summaries,
        )
        for i, e in enumerate(edges):
            e["edge_id"] = f"zf/e{i}"
        return edges

    def test_strict_three_realized_one_zero_removed_passes(self):
        summaries: list = []
        edges = self._extract(_zero_filter_tree(), summaries)
        assert len(edges) == 2  # one zero edge removed
        assert all(e["realized_child_count"] == 3 for e in edges)
        assert all(e["allocated_k"] == 3 for e in edges)
        metrics = validate_tree_construction(
            edges, strict_fresh_iid=True, construction_summaries=summaries
        )
        assert metrics["vdra/construction_failures"] == 0
        manifest = _manifest()
        update_manifest_from_generated_edges(
            manifest, edges, strict=True, construction_summaries=summaries
        )
        assert manifest.group_integrity_failures == 0
        assert manifest.fresh_iid_row_count_matches_allocated_k is True
        assert manifest.segment_count_invariants_passed is True

    def test_all_zero_parent_preserves_construction_summary(self):
        # Every child has exactly zero advantage: zero retained edges, but
        # the construction facts survive in the separate summary and pass.
        summaries: list = []
        edges = self._extract(
            _zero_filter_tree(zero_mask=(True, True, True)), summaries
        )
        assert edges == []  # zero edges never re-enter replay
        assert len(summaries) == 1
        facts = summaries[0]["parent_construction"]
        assert len(facts) == 1
        (parent_facts,) = facts.values()
        assert parent_facts["realized"] == 3
        assert parent_facts["allocated_k"] == 3
        assert parent_facts["retained"] == 0
        metrics = validate_tree_construction(
            [], strict_fresh_iid=True, construction_summaries=summaries
        )
        assert metrics["vdra/construction_failures"] == 0
        manifest = _manifest()
        update_manifest_from_generated_edges(
            manifest, [], strict=True, construction_summaries=summaries
        )
        assert manifest.group_integrity_failures == 0
        assert manifest.fresh_iid_row_count_matches_allocated_k is True

    def test_summary_construction_shortfall_still_fails(self):
        # A summary showing realized != allocated_k is a real construction
        # defect and must fail even with zero retained edges.
        summaries: list = []
        self._extract(_zero_filter_tree(zero_mask=(True, True, True)), summaries)
        (facts,) = summaries[0]["parent_construction"].values()
        facts["realized"] = 2
        with pytest.raises(ValueError, match="allocated_k"):
            validate_tree_construction(
                [], strict_fresh_iid=True, construction_summaries=summaries
            )
        manifest = _manifest()
        with pytest.raises(ValueError):
            update_manifest_from_generated_edges(
                manifest, [], strict=True, construction_summaries=summaries
            )
        assert manifest.group_integrity_failures >= 1
        assert manifest.fresh_iid_row_count_matches_allocated_k is False

    def test_whole_queue_zero_filtered_does_not_fail_queue_identity(self):
        # All edges of one queue removed by the zero filter: the retained
        # rows carry the full pre-filter queue map in tree_summary, so the
        # queue identity must not report a false mismatch.
        from recipe.gear_tree.tree_advantage import extract_edges_from_tree

        tree = _zero_filter_tree(zero_mask=(False, False, True))
        tree["children"][2]["vdra_queue_flush_id"] = 1
        edges = extract_edges_from_tree(
            tree, only_adv_greater_than_zero=True, strict_fresh_iid=True
        )
        for i, e in enumerate(edges):
            e["edge_id"] = f"zfq/e{i}"
        assert {e["queue_flush_id"] for e in edges} == {0}
        metrics = validate_tree_construction(edges, strict_fresh_iid=True)
        assert metrics["vdra/queue_segment_identity_failures"] == 0.0


def _all_zero_summary(tree_id="snap0|iter:1|q:q0|t:abcd", *, k=3, tid_pg="pg0"):
    """A construction summary for a tree whose every child had zero advantage:
    zero retained edges, but the pre-filter facts are intact."""
    return {
        "tree_id": tree_id,
        "policy_snapshot_id": "snap0",
        "rollout_iteration": 1,
        "question_id": "q0",
        "tree_total_segment_count": k,
        "retained_edge_count": 0,
        "queue_released_segment_count": {"0": k},
        "parent_construction": {
            tid_pg: {"realized": k, "allocated_k": k, "retained": 0}
        },
    }


class TestAllZeroTreeManifestIdentity:
    """Finish Medium Stage / M4 follow-up: a tree whose every child had zero
    advantage retains no edges, so its identity lives only in the construction
    summary. Manifest identity verification must count it — otherwise a valid
    all-zero tree falsely leaves unique_tree_ids_verified=False."""

    def test_all_zero_only_batch_verifies_tree_identity(self):
        manifest = _manifest()
        update_manifest_from_generated_edges(
            manifest,
            [],  # every tree fully zero-filtered
            strict=True,
            construction_summaries=[_all_zero_summary()],
        )
        assert manifest.unique_tree_ids_verified is True
        assert manifest.group_integrity_failures == 0
        assert manifest.segment_count_invariants_passed is True

    def test_generic_all_zero_tree_id_is_rejected_strict(self):
        manifest = _manifest()
        with pytest.raises(ValueError, match="canonical"):
            update_manifest_from_generated_edges(
                manifest,
                [],
                strict=True,
                construction_summaries=[_all_zero_summary(tree_id="t0")],
            )
        assert manifest.unique_tree_ids_verified is False

    def test_all_zero_summary_rejects_wrong_snapshot_in_manifest(self):
        manifest = _manifest()
        with pytest.raises(ValueError, match="snapshot"):
            update_manifest_from_generated_edges(
                manifest,
                [],
                strict=True,
                construction_summaries=[
                    _all_zero_summary(
                        tree_id="wrong_snapshot|iter:1|q:q0|t:abcd"
                    )
                ],
            )
        assert manifest.unique_tree_ids_verified is False

    def test_all_zero_summary_rejects_wrong_iteration_in_manifest(self):
        manifest = _manifest()
        with pytest.raises(ValueError, match="rollout iteration"):
            update_manifest_from_generated_edges(
                manifest,
                [],
                strict=True,
                construction_summaries=[
                    _all_zero_summary(tree_id="snap0|iter:999|q:q0|t:abcd")
                ],
            )
        assert manifest.unique_tree_ids_verified is False

    def test_all_zero_summary_rejects_wrong_question_in_manifest(self):
        manifest = _manifest()
        with pytest.raises(ValueError, match="question"):
            update_manifest_from_generated_edges(
                manifest,
                [],
                strict=True,
                construction_summaries=[
                    _all_zero_summary(
                        tree_id="snap0|iter:1|q:wrong_question|t:abcd"
                    )
                ],
            )
        assert manifest.unique_tree_ids_verified is False

    def test_two_all_zero_trees_sharing_id_is_a_collision(self):
        manifest = _manifest()
        dup = _all_zero_summary()
        with pytest.raises(ValueError, match="collides"):
            update_manifest_from_generated_edges(
                manifest,
                [],
                strict=True,
                construction_summaries=[dup, dict(dup)],
            )
        assert manifest.unique_tree_ids_verified is False

    def test_all_zero_tree_colliding_with_edge_tree_is_a_collision(self):
        manifest = _manifest()
        shared = "snap0|iter:1|q:q0|t:abcd"
        edges = _full_tree(k=2)
        for e in edges:
            e["tree_id"] = shared
        with pytest.raises(ValueError, match="collides"):
            update_manifest_from_generated_edges(
                manifest,
                edges,
                strict=True,
                construction_summaries=[_all_zero_summary(tree_id=shared)],
            )

    def test_mixed_batch_counts_both_edge_and_all_zero_trees(self):
        manifest = _manifest()
        edges = _full_tree(k=2)  # tree_id "t0" (legacy, fine for edges)
        update_manifest_from_generated_edges(
            manifest,
            edges,
            strict=True,
            construction_summaries=[_all_zero_summary()],
        )
        assert manifest.unique_tree_ids_verified is True
