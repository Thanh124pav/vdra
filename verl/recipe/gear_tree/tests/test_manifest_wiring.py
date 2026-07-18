"""PLAN.md P0.6: manifest lifecycle helpers wired by RayGearTreeTrainer.

These tests exercise the standalone helpers in
:mod:`recipe.gear_tree.manifest_lifecycle`, which the trainer calls from
its ``fit()`` loop. Testing them directly keeps the coverage CPU-runnable
without importing the ray / torchdata stack.
"""

from __future__ import annotations

import pytest
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

from recipe.gear_tree.manifest_lifecycle import (
    build_run_manifest,
    update_manifest_from_edges,
)
from recipe.gear_tree.run_manifest import (
    POLICY_AGGREGATION_LEGACY,
    POLICY_AGGREGATION_SEGMENT_MEAN,
    POLICY_AGGREGATION_VDRA,
    SEGMENT_TOKEN_REDUCTION_MEAN,
    is_valid_main_run,
    validate_main_run,
)


def _vdra_config():
    return {
        "tree_policy": {
            "advantage_mode": "spo_local",
            "policy_aggregation": POLICY_AGGREGATION_SEGMENT_MEAN,
            "segment_token_reduction": SEGMENT_TOKEN_REDUCTION_MEAN,
            "strict_group_integrity": True,
            "include_root_parent_group": True,
        },
        "gear_tree": {
            "tree_shape": [6, 6, 6],
            "segment_length": 100,
            "gear": {
                "enabled": True,
                "strict_vdra": True,
                "k_algorithm": "budget_allocation",
                "pilot_execution_mode": "fresh_iid",
                "allocation_runtime": "online_timeout",
            },
        },
        "actor_loss_mode": "vdra_segment_mean_ppo",
    }


def _legacy_config():
    return {
        "tree_policy": {
            "advantage_mode": "spo_local",
            "policy_aggregation": POLICY_AGGREGATION_LEGACY,
            "strict_group_integrity": False,
        },
        "gear_tree": {
            "tree_shape": [6, 6, 6],
            "segment_length": 100,
            "gear": {
                "enabled": False,
                "strict_vdra": False,
                "k_algorithm": "simple",
                "pilot_execution_mode": "fresh_iid",
                "allocation_runtime": "online_timeout",
            },
        },
        "actor_loss_mode": "treetune_ppo",
    }


def _fresh_iid_group() -> list[dict]:
    return [
        {
            "edge_id": "e0",
            "tree_id": "T",
            "parent_group_id": "T/pg",
            "child_index": 0,
            "allocated_k": 2,
            "sample_multiplicity": 1,
            "queue_flush_id": 0,
            "queue_released_segment_count": 2,
            "tree_total_segment_count": 2,
            "advantage": 1.0,
            "response_token_ids": [1],
        },
        {
            "edge_id": "e1",
            "tree_id": "T",
            "parent_group_id": "T/pg",
            "child_index": 1,
            "allocated_k": 2,
            "sample_multiplicity": 1,
            "queue_flush_id": 0,
            "queue_released_segment_count": 2,
            "tree_total_segment_count": 2,
            "advantage": 1.0,
            "response_token_ids": [1],
        },
    ]


def test_manifest_reflects_canonical_vdra_config():
    cfg = _vdra_config()
    manifest = build_run_manifest(
        tree_policy=cfg["tree_policy"],
        gear_tree_cfg=cfg["gear_tree"],
        actor_loss_mode=cfg["actor_loss_mode"],
    )
    assert manifest.policy_aggregation == POLICY_AGGREGATION_SEGMENT_MEAN
    assert manifest.segment_token_reduction == SEGMENT_TOKEN_REDUCTION_MEAN
    assert manifest.advantage_mode == "spo_local"
    # PLAN.md P0.6: operational bits are set from runtime observation, not
    # from config. They start False.
    assert manifest.complete_tree_replay is False
    assert manifest.complete_parent_microbatches is False
    assert manifest.stored_old_log_probs_used is False
    assert manifest.rollout_scorer_weights_verified is False
    assert manifest.no_truncation is False
    assert manifest.segment_count_invariants_passed is False
    # Startup value: main-run is invalid until fit() flips runtime bits.
    assert not is_valid_main_run(manifest)
    assert manifest.extras["actor_loss_mode"] == "vdra_segment_mean_ppo"
    assert manifest.extras["gear_pilot_execution_mode"] == "fresh_iid"


def test_manifest_reflects_legacy_baseline_config():
    cfg = _legacy_config()
    manifest = build_run_manifest(
        tree_policy=cfg["tree_policy"],
        gear_tree_cfg=cfg["gear_tree"],
        actor_loss_mode=cfg["actor_loss_mode"],
    )
    assert manifest.policy_aggregation == POLICY_AGGREGATION_LEGACY
    assert manifest.complete_tree_replay is False
    reason = validate_main_run(manifest)
    assert reason is not None
    # PLAN.md P0.6: legacy policy_aggregation does not match main.
    assert POLICY_AGGREGATION_SEGMENT_MEAN in reason


def test_update_manifest_from_edges_reports_no_failure_for_valid_group():
    cfg = _vdra_config()
    manifest = build_run_manifest(
        tree_policy=cfg["tree_policy"],
        gear_tree_cfg=cfg["gear_tree"],
        actor_loss_mode=cfg["actor_loss_mode"],
    )
    metrics = update_manifest_from_edges(
        manifest, _fresh_iid_group(), strict=True
    )
    assert metrics["vdra/group_integrity_failures"] == 0
    assert manifest.group_integrity_failures == 0
    assert manifest.segment_count_failures == 0
    # After a clean step the trainer flips the invariants-passed bit.
    manifest.record_invariant_pass()
    # Still not a valid main run because scorer verification is not set here.
    assert not is_valid_main_run(manifest)
    manifest.rollout_scorer_weights_verified = True
    assert is_valid_main_run(manifest)


def test_update_manifest_from_edges_records_failure_on_partial_group_non_strict():
    cfg = _vdra_config()
    manifest = build_run_manifest(
        tree_policy=cfg["tree_policy"],
        gear_tree_cfg=cfg["gear_tree"],
        actor_loss_mode=cfg["actor_loss_mode"],
    )
    partial = _fresh_iid_group()[:1]  # drop sibling -> partial group
    metrics = update_manifest_from_edges(manifest, partial, strict=False)
    assert metrics["vdra/group_integrity_failures"] >= 1
    assert manifest.group_integrity_failures >= 1
    assert not is_valid_main_run(manifest)


def test_update_manifest_from_edges_raises_on_partial_group_strict():
    cfg = _vdra_config()
    manifest = build_run_manifest(
        tree_policy=cfg["tree_policy"],
        gear_tree_cfg=cfg["gear_tree"],
        actor_loss_mode=cfg["actor_loss_mode"],
    )
    partial = _fresh_iid_group()[:1]
    with pytest.raises(ValueError, match="Group-integrity"):
        update_manifest_from_edges(manifest, partial, strict=True)
    # Failure is recorded even when we raise, so a killed strict run leaves
    # evidence on disk.
    assert manifest.group_integrity_failures >= 1


def test_manifest_persists_across_json_round_trip(tmp_path):
    cfg = _vdra_config()
    manifest = build_run_manifest(
        tree_policy=cfg["tree_policy"],
        gear_tree_cfg=cfg["gear_tree"],
        actor_loss_mode=cfg["actor_loss_mode"],
    )
    # PLAN.md P0.6: flip the runtime bits by observing a clean batch.
    update_manifest_from_edges(manifest, _fresh_iid_group(), strict=True)
    manifest.record_invariant_pass()
    manifest.rollout_scorer_weights_verified = True
    path = tmp_path / "vdra_run_manifest.json"
    manifest.save(path)
    from recipe.gear_tree.run_manifest import RunManifest

    loaded = RunManifest.load(path)
    assert loaded.policy_aggregation == POLICY_AGGREGATION_SEGMENT_MEAN
    assert loaded.segment_token_reduction == SEGMENT_TOKEN_REDUCTION_MEAN
    assert loaded.segment_count_invariants_passed is True
    assert loaded.stored_old_log_probs_used is True
    assert loaded.no_truncation is True
    assert loaded.extras["actor_loss_mode"] == "vdra_segment_mean_ppo"
    assert is_valid_main_run(loaded)
