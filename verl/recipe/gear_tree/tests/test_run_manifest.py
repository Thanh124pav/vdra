"""PLAN.md P0.N8: run manifest contract tests."""

from __future__ import annotations

import pytest

from recipe.gear_tree.run_manifest import (
    POLICY_AGGREGATION_LEGACY,
    POLICY_AGGREGATION_VDRA,
    RunManifest,
    is_valid_main_run,
    validate_main_run,
)


def _valid_main_manifest() -> RunManifest:
    m = RunManifest(
        policy_aggregation=POLICY_AGGREGATION_VDRA,
        advantage_mode="spo_local",
        complete_tree_replay=True,
        complete_parent_microbatches=True,
        node_balanced_invariants_passed=True,
        rollout_scorer_weights_verified=True,
        fresh_iid_row_count_matches_allocated_k=True,
    )
    return m


def test_valid_main_run_passes():
    assert is_valid_main_run(_valid_main_manifest())
    assert validate_main_run(_valid_main_manifest()) is None


def test_legacy_aggregation_is_not_a_valid_main_run():
    m = _valid_main_manifest()
    m.policy_aggregation = POLICY_AGGREGATION_LEGACY
    reason = validate_main_run(m)
    assert reason is not None
    assert POLICY_AGGREGATION_LEGACY in reason


def test_partial_parent_group_invalidates_run():
    m = _valid_main_manifest()
    m.record_parent_split(2)
    assert not is_valid_main_run(m)


def test_group_integrity_failure_invalidates_run():
    m = _valid_main_manifest()
    m.record_integrity_failure()
    assert not is_valid_main_run(m)


def test_undocumented_fallback_invalidates_run():
    m = _valid_main_manifest()
    m.node_balanced_invariants_passed = False
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
    assert loaded.extras["dataset_hash"] == "abc123"
    assert is_valid_main_run(loaded)
