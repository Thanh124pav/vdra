"""PLAN.md P0.6: the manifest is set only from runtime observations.

A canonical main run stays invalid until at least one successful actor
update passes every invariant; any later failure keeps the run invalid.
Both ``mean`` and ``sum`` reductions produce a valid main manifest.
"""

from __future__ import annotations

import pytest

from recipe.gear_tree.manifest_lifecycle import (
    build_run_manifest,
    update_manifest_from_edges,
)
from recipe.gear_tree.run_manifest import (
    POLICY_AGGREGATION_SEGMENT_MEAN,
    SEGMENT_TOKEN_REDUCTION_MEAN,
    SEGMENT_TOKEN_REDUCTION_SUM,
    RunManifest,
    validate_main_run,
)


def _clean_edges():
    # PLAN.md P0.2: every realized segment is counted in
    # tree_total_segment_count *before* advantage filtering. Two retained
    # rows for a tree with 2 realized segments -> tree_total_segment_count=2.
    return [
        {
            "edge_id": "T0:e0",
            "tree_id": "T0",
            "parent_group_id": "T0:p0",
            "sample_multiplicity": 1,
            "allocated_k": 2,
            "tree_total_segment_count": 2,
            "queue_flush_id": "q0",
            "queue_released_segment_count": 2,
            "generation_rollout_iteration": 0,
        },
        {
            "edge_id": "T0:e1",
            "tree_id": "T0",
            "parent_group_id": "T0:p0",
            "sample_multiplicity": 1,
            "allocated_k": 2,
            "tree_total_segment_count": 2,
            "queue_flush_id": "q0",
            "queue_released_segment_count": 2,
            "generation_rollout_iteration": 0,
        },
    ]


def _cfg(policy_agg=POLICY_AGGREGATION_SEGMENT_MEAN, reduction=SEGMENT_TOKEN_REDUCTION_MEAN):
    tree_policy = {
        "policy_aggregation": policy_agg,
        "segment_token_reduction": reduction,
        "advantage_mode": "spo_local",
        "strict_group_integrity": True,
    }
    gear_tree_cfg = {
        "gear": {
            "enabled": True,
            "strict_vdra": True,
            "pilot_execution_mode": "fresh_iid",
            "allocation_runtime": "online_timeout",
        }
    }
    return tree_policy, gear_tree_cfg


def test_manifest_starts_invalid_even_with_canonical_config():
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_segment_mean_ppo",
    )
    # PLAN.md P0.6: config alone must NEVER validate the main-run contract.
    assert validate_main_run(manifest) is not None
    assert manifest.complete_tree_replay is False
    assert manifest.stored_old_log_probs_used is False
    assert manifest.rollout_scorer_weights_verified is False
    assert manifest.segment_count_invariants_passed is False
    assert manifest.no_truncation is False


def test_clean_synthetic_update_produces_valid_manifest_mean():
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_segment_mean_ppo",
    )
    update_manifest_from_edges(manifest, _clean_edges(), strict=True)
    manifest.rollout_scorer_weights_verified = True  # observed by trainer
    manifest.record_invariant_pass()
    # PLAN.md P0.7: trainer flips this after observing the correct number of
    # optimizer.step() calls (see P0.3). Simulate here.
    manifest.optimizer_step_accounting_valid = True
    manifest.num_optimizer_steps_total = 4  # PLAN.md P0.J: >=1 observed step
    assert validate_main_run(manifest) is None


def test_clean_synthetic_update_produces_valid_manifest_sum():
    # PLAN.md P0.6: sum-reduction ablation must also be a valid main manifest.
    tree_policy, gear_tree_cfg = _cfg(reduction=SEGMENT_TOKEN_REDUCTION_SUM)
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_segment_mean_ppo",
    )
    update_manifest_from_edges(manifest, _clean_edges(), strict=True)
    manifest.rollout_scorer_weights_verified = True
    manifest.record_invariant_pass()
    manifest.optimizer_step_accounting_valid = True
    manifest.num_optimizer_steps_total = 4  # PLAN.md P0.J: >=1 observed step
    assert manifest.segment_token_reduction == SEGMENT_TOKEN_REDUCTION_SUM
    assert validate_main_run(manifest) is None


def test_later_failure_keeps_run_invalid():
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_segment_mean_ppo",
    )
    update_manifest_from_edges(manifest, _clean_edges(), strict=True)
    manifest.rollout_scorer_weights_verified = True
    manifest.record_invariant_pass()
    manifest.optimizer_step_accounting_valid = True
    manifest.num_optimizer_steps_total = 4  # PLAN.md P0.J: >=1 observed step
    assert validate_main_run(manifest) is None

    # Second batch — inject a broken parent group so group-integrity fails.
    broken = _clean_edges()
    broken[1]["allocated_k"] = 99
    with pytest.raises(ValueError):
        update_manifest_from_edges(manifest, broken, strict=True)
    assert manifest.group_integrity_failures > 0
    assert manifest.complete_tree_replay is False
    assert validate_main_run(manifest) is not None


def test_manifest_save_load_preserves_all_fields(tmp_path):
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_segment_mean_ppo",
    )
    update_manifest_from_edges(manifest, _clean_edges(), strict=True)
    manifest.rollout_scorer_weights_verified = True
    manifest.record_invariant_pass()
    manifest.optimizer_step_accounting_valid = True
    manifest.num_optimizer_steps_total = 4  # PLAN.md P0.J: >=1 observed step

    p = tmp_path / "manifest.json"
    manifest.save(p)
    loaded = RunManifest.load(p)
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.segment_token_reduction == manifest.segment_token_reduction


class TestObservedFactsP0J:
    """PLAN.md P0.J: manifest bits flip only from observed runtime events."""

    def _manifest(self):
        tree_policy, gear_tree_cfg = _cfg()
        return build_run_manifest(
            tree_policy=tree_policy,
            gear_tree_cfg=gear_tree_cfg,
            actor_loss_mode="vdra_segment_mean_ppo",
        )

    def test_replay_stage_update_does_not_hard_set_observed_bits(self):
        from recipe.gear_tree.manifest_lifecycle import (
            update_manifest_from_replay_batch,
        )

        manifest = self._manifest()
        sampled = [
            {**e, "question_id": "q0", "advantage": 1.0}
            for e in _clean_edges()
        ]
        update_manifest_from_replay_batch(manifest, sampled, strict=True)
        # These flip ONLY from the actor metric / successful tensorization.
        assert manifest.stored_old_log_probs_used is False
        assert manifest.no_truncation is False
        # Replay-age observation IS a per-batch observed fact.
        assert manifest.replay_age_uses_rollout_iteration is True

    def test_invariant_recorders_are_independent_claims(self):
        manifest = self._manifest()
        manifest.record_segment_invariant_pass()
        assert manifest.segment_count_invariants_passed is True
        assert manifest.node_balanced_invariants_passed is False

        manifest2 = self._manifest()
        manifest2.record_node_balanced_invariant_pass()
        assert manifest2.node_balanced_invariants_passed is True
        assert manifest2.segment_count_invariants_passed is False

    def test_complete_tree_unit_is_never_a_valid_main_run(self):
        manifest = self._manifest()
        update_manifest_from_edges(manifest, _clean_edges(), strict=True)
        manifest.rollout_scorer_weights_verified = True
        manifest.record_segment_invariant_pass()
        manifest.optimizer_step_accounting_valid = True
        manifest.num_optimizer_steps_total = 4
        assert validate_main_run(manifest) is None
        manifest.replay_sampling_unit = "complete_tree"
        reason = validate_main_run(manifest)
        assert reason is not None and "complete_tree" in reason

    def test_main_manifest_invalid_before_first_optimizer_step(self):
        manifest = self._manifest()
        update_manifest_from_edges(manifest, _clean_edges(), strict=True)
        manifest.rollout_scorer_weights_verified = True
        manifest.record_segment_invariant_pass()
        manifest.optimizer_step_accounting_valid = True
        assert manifest.num_optimizer_steps_total == 0
        reason = validate_main_run(manifest)
        assert reason is not None and "num_optimizer_steps_total" in reason

    def test_trainer_and_actor_wiring_for_observed_facts(self):
        from pathlib import Path

        recipe_root = Path(__file__).resolve().parents[1]
        trainer_source = (recipe_root / "gear_ray_trainer.py").read_text()
        # The stored-log-prob bit comes from the actor metric only.
        assert "actor/used_stored_old_log_probs" in trainer_source
        assert "actor_used_stored_old_log_probs" in trainer_source
        # no_truncation flips after strict tensorization succeeds.
        assert "self.run_manifest.no_truncation = True" in trainer_source
        # The loss-mode-specific invariant recorders are used, not the alias.
        assert "record_segment_invariant_pass()" in trainer_source
        assert "record_node_balanced_invariant_pass()" in trainer_source
        assert "record_invariant_pass()" not in trainer_source.replace(
            "record_segment_invariant_pass()", ""
        ).replace("record_node_balanced_invariant_pass()", "")

        actor_source = (
            recipe_root.parents[1] / "verl" / "workers" / "actor" / "dp_actor.py"
        ).read_text()
        assert "actor/used_stored_old_log_probs" in actor_source


def test_queue_identity_failure_flags_segment_count():
    # PLAN.md P0.2 acceptance: sum_q queue_released_segment_count[q] must
    # match tree_total_segment_count; a mismatch is a segment-count failure.
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_segment_mean_ppo",
    )
    edges = _clean_edges()
    # Two queues that each claim 2 released segments = 4, but total is 2.
    edges[1]["queue_flush_id"] = "q1"
    edges[1]["queue_released_segment_count"] = 2
    update_manifest_from_edges(manifest, edges, strict=False)
    assert manifest.segment_count_failures > 0
