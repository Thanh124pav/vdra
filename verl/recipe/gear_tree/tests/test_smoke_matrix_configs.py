"""PLAN.md §8: four-way smoke-matrix config-shape tests.

These pytests verify that the four smoke overlays
(``smoke_a_spo_baseline``, ``smoke_b_vdra_alloc_legacy_loss``,
``smoke_c_uniform_alloc_node_balanced``, ``smoke_d_full_vdra``) compose
correctly with the base ``gear_tree_trainer`` config and land the
combination the matrix cell claims. Catching config drift here means CI
fails before a cluster launch instead of hours into a wasted run.

The tests only touch YAML shape — they do NOT instantiate the trainer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _dict_deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (Hydra-like semantics for tests)."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _dict_deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolved(name: str) -> dict:
    base = _load_yaml("gear_tree_trainer")
    overlay = _load_yaml(name)
    return _dict_deep_merge(base, overlay)


def _matrix_cell(resolved: dict) -> tuple[str, str, str, str]:
    tp = resolved.get("tree_policy", {}) or {}
    gear = (resolved.get("gear_tree", {}) or {}).get("gear", {}) or {}
    actor_loss = (
        (resolved.get("actor_rollout_ref") or {}).get("actor") or {}
    ).get("policy_loss", {}).get("loss_mode")
    return (
        str(tp.get("policy_aggregation")),
        str(gear.get("k_algorithm")),
        str(gear.get("pilot_execution_mode")),
        str(actor_loss),
    )


@pytest.mark.parametrize(
    "overlay,expected_cell",
    [
        (
            # Smoke A: uniform allocation + legacy token mean.
            "smoke_a_spo_baseline",
            ("legacy_token_mean", "simple", "fresh_iid", "treetune_ppo"),
        ),
        (
            # Smoke B: VDRA construction + legacy token mean.
            "smoke_b_vdra_alloc_legacy_loss",
            ("legacy_token_mean", "budget_allocation", "fresh_iid", "treetune_ppo"),
        ),
        (
            # Smoke C: uniform construction + node-balanced (labeled ablation).
            "smoke_c_uniform_alloc_node_balanced",
            ("vdra_node_balanced", "simple", "fresh_iid", "vdra_node_balanced_ppo"),
        ),
        (
            # Smoke D: full VDRA (segment-average main path, PLAN.md P0.1).
            "smoke_d_full_vdra",
            ("segment_mean", "budget_allocation", "fresh_iid", "vdra_segment_mean_ppo"),
        ),
    ],
)
def test_smoke_overlay_lands_expected_matrix_cell(overlay: str, expected_cell: tuple):
    resolved = _resolved(overlay)
    assert _matrix_cell(resolved) == expected_cell, overlay


def test_smoke_matrix_cells_are_pairwise_distinct():
    """The four overlays must land on four distinct matrix cells."""
    cells = {
        overlay: _matrix_cell(_resolved(overlay))
        for overlay in (
            "smoke_a_spo_baseline",
            "smoke_b_vdra_alloc_legacy_loss",
            "smoke_c_uniform_alloc_node_balanced",
            "smoke_d_full_vdra",
        )
    }
    assert len(set(cells.values())) == 4, cells


def test_smoke_d_full_vdra_enables_strict_group_integrity():
    resolved = _resolved("smoke_d_full_vdra")
    assert resolved["tree_policy"]["strict_group_integrity"] is True
    # PLAN.md P0.1: main VDRA is the segment-average path.
    assert resolved["tree_policy"]["policy_aggregation"] == "segment_mean"
    assert resolved["tree_policy"]["segment_token_reduction"] == "mean"
    policy_loss = resolved["actor_rollout_ref"]["actor"]["policy_loss"]
    assert policy_loss["loss_mode"] == "vdra_segment_mean_ppo"
    assert policy_loss["segment_token_reduction"] == "mean"
    gear = resolved["gear_tree"]["gear"]
    assert gear["strict_vdra"] is True
    assert gear["pilot_execution_mode"] == "fresh_iid"
    assert gear["allocation_runtime"] == "online_timeout"
    assert gear["bound_form"] == "linear"
    assert gear["tail_mode"] == "none"
    assert float(gear["eps_tail"]) == 0.0
    assert gear["allocation_scope"] == "per_queue_flush_within_tree"


def test_smoke_a_baseline_labels_itself_as_legacy():
    resolved = _resolved("smoke_a_spo_baseline")
    assert resolved["tree_policy"]["policy_aggregation"] == "legacy_token_mean"
    assert resolved["gear_tree"]["gear"]["enabled"] is False


def test_segment_token_reduction_sum_override_preserves_pipeline():
    """PLAN.md P0.7: overriding segment_token_reduction=sum must not change
    allocation, replay, segment counting, or outer aggregation settings.
    """
    base = _load_yaml("gear_tree_trainer")
    overridden = _dict_deep_merge(
        base,
        {
            "tree_policy": {"segment_token_reduction": "sum"},
            "actor_rollout_ref": {"actor": {"policy_loss": {"segment_token_reduction": "sum"}}},
        },
    )
    # Only the two segment_token_reduction knobs differ.
    assert overridden["tree_policy"]["policy_aggregation"] == "segment_mean"
    assert (
        overridden["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"]
        == "vdra_segment_mean_ppo"
    )
    assert overridden["gear_tree"]["gear"]["allocation_runtime"] == "online_timeout"
    assert (
        overridden["gear_tree"]["gear"]["allocation_scope"]
        == "per_queue_flush_within_tree"
    )
    assert overridden["gear_tree"]["gear"]["pilot_execution_mode"] == "fresh_iid"
    assert overridden["gear_tree"]["gear"]["bound_form"] == "linear"


def test_smoke_b_and_c_are_labelled_as_ablations():
    b = _resolved("smoke_b_vdra_alloc_legacy_loss")
    c = _resolved("smoke_c_uniform_alloc_node_balanced")
    # B: VDRA construction but legacy loss.
    assert b["gear_tree"]["gear"]["k_algorithm"] == "budget_allocation"
    assert (
        b["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == "treetune_ppo"
    )
    # C: no allocator, canonical loss.
    assert c["gear_tree"]["gear"]["k_algorithm"] == "simple"
    assert (
        c["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"]
        == "vdra_node_balanced_ppo"
    )
    # PLAN.md M5: non-canonical loss/aggregation ablations must NOT run under
    # strict_vdra (the strict canonical triple would reject them at startup).
    assert b["gear_tree"]["gear"]["strict_vdra"] is False
    assert c["gear_tree"]["gear"]["strict_vdra"] is False


def test_smoke_ablations_pass_config_consistency_validation():
    # PLAN.md M5: every shipped smoke overlay must pass the same startup
    # config validation the trainer runs, so none of them can be launched into
    # an inconsistent aggregation/loss pairing.
    from omegaconf import OmegaConf

    from recipe.gear_tree.config_validation import (
        validate_policy_loss_consistency,
    )

    for name in (
        "smoke_a_spo_baseline",
        "smoke_b_vdra_alloc_legacy_loss",
        "smoke_c_uniform_alloc_node_balanced",
        "smoke_d_full_vdra",
    ):
        cfg = OmegaConf.create(_resolved(name))
        # Should not raise.
        assert validate_policy_loss_consistency(cfg) is None, name
