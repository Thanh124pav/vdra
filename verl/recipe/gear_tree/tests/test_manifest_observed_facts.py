"""PLAN.md P0.6: the manifest is set only from runtime observations.

A canonical main run stays invalid until at least one successful actor
update passes every invariant; any later failure keeps the run invalid.
"""

from __future__ import annotations

import pytest

from recipe.gear_tree.manifest_lifecycle import (
    build_run_manifest,
    update_manifest_from_edges,
)
from recipe.gear_tree.run_manifest import (
    POLICY_AGGREGATION_VDRA,
    RunManifest,
    validate_main_run,
)


def _clean_edges():
    return [
        {
            "tree_id": "T0",
            "parent_group_id": "T0:p0",
            "sample_multiplicity": 1,
            "allocated_k": 2,
        },
        {
            "tree_id": "T0",
            "parent_group_id": "T0:p0",
            "sample_multiplicity": 1,
            "allocated_k": 2,
        },
    ]


def _cfg(policy_agg=POLICY_AGGREGATION_VDRA):
    tree_policy = {
        "policy_aggregation": policy_agg,
        "advantage_mode": "spo_local",
        "strict_group_integrity": True,
    }
    gear_tree_cfg = {"gear": {"enabled": True, "strict_vdra": True, "pilot_execution_mode": "fresh_iid"}}
    return tree_policy, gear_tree_cfg


def test_manifest_starts_invalid_even_with_canonical_config():
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_node_balanced_ppo",
    )
    # PLAN.md P0.6: config alone must NEVER validate the main-run contract.
    assert validate_main_run(manifest) is not None
    assert manifest.complete_tree_replay is False
    assert manifest.complete_parent_microbatches is False
    assert manifest.rollout_scorer_weights_verified is False
    assert manifest.node_balanced_invariants_passed is False


def test_clean_synthetic_update_produces_valid_manifest():
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_node_balanced_ppo",
    )
    update_manifest_from_edges(manifest, _clean_edges(), strict=True)
    manifest.rollout_scorer_weights_verified = True  # observed by trainer
    manifest.record_invariant_pass()
    assert validate_main_run(manifest) is None


def test_partial_parent_group_invalidates_manifest():
    # sample_multiplicity==1 but only 1 row for allocated_k=2.
    edges = [_clean_edges()[0]]
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_node_balanced_ppo",
    )
    with pytest.raises(ValueError):
        update_manifest_from_edges(manifest, edges, strict=True)
    assert manifest.group_integrity_failures > 0
    assert manifest.complete_tree_replay is False
    assert validate_main_run(manifest) is not None


def test_later_failure_keeps_run_invalid():
    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_node_balanced_ppo",
    )
    update_manifest_from_edges(manifest, _clean_edges(), strict=True)
    manifest.rollout_scorer_weights_verified = True
    manifest.record_invariant_pass()
    assert validate_main_run(manifest) is None

    # Second batch — inject a broken parent group.
    broken = _clean_edges()
    broken[1]["allocated_k"] = 99
    with pytest.raises(ValueError):
        update_manifest_from_edges(manifest, broken, strict=True)
    assert manifest.group_integrity_failures > 0
    assert manifest.complete_tree_replay is False
    assert validate_main_run(manifest) is not None


def test_manifest_save_load_preserves_all_fields():
    import json
    import tempfile
    from pathlib import Path

    tree_policy, gear_tree_cfg = _cfg()
    manifest = build_run_manifest(
        tree_policy=tree_policy,
        gear_tree_cfg=gear_tree_cfg,
        actor_loss_mode="vdra_node_balanced_ppo",
    )
    update_manifest_from_edges(manifest, _clean_edges(), strict=True)
    manifest.rollout_scorer_weights_verified = True
    manifest.record_invariant_pass()

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.json"
        manifest.save(p)
        loaded = RunManifest.load(p)
        assert loaded.to_dict() == manifest.to_dict()
