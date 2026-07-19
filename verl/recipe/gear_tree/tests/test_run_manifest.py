"""PLAN.md P0.6: run manifest contract tests for the segment-average main path."""

from __future__ import annotations

import pytest

from recipe.gear_tree.run_manifest import (
    POLICY_AGGREGATION_LEGACY,
    POLICY_AGGREGATION_SEGMENT_MEAN,
    POLICY_AGGREGATION_VDRA,
    SEGMENT_TOKEN_REDUCTION_MEAN,
    SEGMENT_TOKEN_REDUCTION_SUM,
    RunManifest,
    is_valid_main_run,
    validate_main_run,
)


def _valid_main_manifest(*, reduction: str = SEGMENT_TOKEN_REDUCTION_MEAN) -> RunManifest:
    m = RunManifest(
        policy_aggregation=POLICY_AGGREGATION_SEGMENT_MEAN,
        segment_token_reduction=reduction,
        advantage_mode="spo_local",
        complete_tree_replay=True,
        complete_parent_microbatches=True,
        node_balanced_invariants_passed=True,
        segment_count_invariants_passed=True,
        stored_old_log_probs_used=True,
        rollout_scorer_weights_verified=True,
        no_truncation=True,
        fresh_iid_row_count_matches_allocated_k=True,
        # PLAN.md P0.7 canonical bits — always required on a valid main run.
        replay_age_uses_rollout_iteration=True,
        optimizer_step_accounting_valid=True,
        unique_tree_ids_verified=True,
        # PLAN.md P0.J: valid only after >= 1 observed canonical optimizer
        # step on the edge-level replay unit.
        replay_sampling_unit="edge",
        num_optimizer_steps_total=4,
    )
    return m


def test_valid_main_run_passes_with_mean():
    assert is_valid_main_run(_valid_main_manifest())
    assert validate_main_run(_valid_main_manifest()) is None


def test_valid_main_run_passes_with_sum_override():
    # PLAN.md P0.6: sum-mode run is a valid main-run label too.
    assert is_valid_main_run(_valid_main_manifest(reduction=SEGMENT_TOKEN_REDUCTION_SUM))


def test_legacy_aggregation_is_not_a_valid_main_run():
    m = _valid_main_manifest()
    m.policy_aggregation = POLICY_AGGREGATION_LEGACY
    reason = validate_main_run(m)
    assert reason is not None
    assert POLICY_AGGREGATION_SEGMENT_MEAN in reason


def test_node_balanced_ablation_is_not_a_valid_main_run():
    # The old parent-balanced aggregation is now an ablation, not a valid main.
    m = _valid_main_manifest()
    m.policy_aggregation = POLICY_AGGREGATION_VDRA
    assert not is_valid_main_run(m)


def test_invalid_reduction_invalidates_run():
    m = _valid_main_manifest()
    m.segment_token_reduction = "banana"
    reason = validate_main_run(m)
    assert reason is not None
    assert "segment_token_reduction" in reason


def test_missing_stored_old_log_probs_invalidates_run():
    m = _valid_main_manifest()
    m.stored_old_log_probs_used = False
    assert not is_valid_main_run(m)


def test_truncation_invalidates_run():
    m = _valid_main_manifest()
    m.no_truncation = False
    assert not is_valid_main_run(m)


def test_segment_count_failure_invalidates_run():
    m = _valid_main_manifest()
    m.record_segment_count_failure()
    assert not is_valid_main_run(m)


def test_segment_count_invariants_not_passed_invalidates_run():
    m = _valid_main_manifest()
    m.segment_count_invariants_passed = False
    assert not is_valid_main_run(m)


def test_scorer_weight_mismatch_invalidates_run():
    m = _valid_main_manifest()
    m.rollout_scorer_weights_verified = False
    assert not is_valid_main_run(m)


def test_manifest_round_trips_through_json(tmp_path):
    m = _valid_main_manifest()
    m.extras["dataset_hash"] = "abc123"
    path = tmp_path / "manifest.json"
    m.save(path)
    loaded = RunManifest.load(path)
    assert loaded.policy_aggregation == m.policy_aggregation
    assert loaded.segment_token_reduction == m.segment_token_reduction
    assert loaded.extras["dataset_hash"] == "abc123"
    assert is_valid_main_run(loaded)


def test_manifest_round_trips_sum_reduction(tmp_path):
    m = _valid_main_manifest(reduction=SEGMENT_TOKEN_REDUCTION_SUM)
    path = tmp_path / "sum_manifest.json"
    m.save(path)
    loaded = RunManifest.load(path)
    assert loaded.segment_token_reduction == SEGMENT_TOKEN_REDUCTION_SUM
