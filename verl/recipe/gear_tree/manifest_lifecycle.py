"""PLAN.md P0.N8: RunManifest lifecycle helpers.

Extracted to a standalone module (no verl / torchdata / ray imports) so the
manifest lifecycle can be unit-tested on CPU without loading the whole
RayGearTreeTrainer stack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from recipe.gear_tree.run_manifest import (
    ADVANTAGE_MODE_ABLATION,
    ADVANTAGE_MODE_SPO_LOCAL,
    POLICY_AGGREGATION_LEGACY,
    POLICY_AGGREGATION_VDRA,
    SEGMENT_DEFINITION_FIXED,
    RunManifest,
)
from recipe.gear_tree.tree_data import (
    compute_group_metrics,
    validate_group_integrity,
)


def build_run_manifest(
    *,
    tree_policy: Mapping[str, Any],
    gear_tree_cfg: Mapping[str, Any],
    actor_loss_mode: str,
) -> RunManifest:
    """PLAN.md P0.N8: derive the run manifest from config."""
    gear_cfg = dict(gear_tree_cfg.get("gear") or {})
    policy_agg = str(tree_policy.get("policy_aggregation", POLICY_AGGREGATION_LEGACY))
    advantage_mode = str(tree_policy.get("advantage_mode", ADVANTAGE_MODE_ABLATION))
    if advantage_mode not in {ADVANTAGE_MODE_SPO_LOCAL, ADVANTAGE_MODE_ABLATION}:
        advantage_mode = ADVANTAGE_MODE_ABLATION
    complete_tree_replay = bool(
        tree_policy.get(
            "strict_group_integrity", policy_agg == POLICY_AGGREGATION_VDRA
        )
    )
    complete_parent_microbatches = complete_tree_replay
    manifest = RunManifest(
        policy_aggregation=policy_agg,
        advantage_mode=advantage_mode,
        segment_definition=SEGMENT_DEFINITION_FIXED,
        complete_tree_replay=complete_tree_replay,
        complete_parent_microbatches=complete_parent_microbatches,
        node_balanced_invariants_passed=False,
        rollout_scorer_weights_verified=False,
        fresh_iid_row_count_matches_allocated_k=True,
    )
    manifest.extras.update(
        {
            "gear_enabled": bool(gear_cfg.get("enabled", False)),
            "gear_strict_vdra": bool(gear_cfg.get("strict_vdra", True)),
            "gear_k_algorithm": str(gear_cfg.get("k_algorithm", "simple")),
            "gear_pilot_execution_mode": str(
                gear_cfg.get("pilot_execution_mode", "fresh_iid")
            ),
            "gear_allocation_runtime": str(
                gear_cfg.get("allocation_runtime", "online_timeout")
            ),
            "tree_shape": list(gear_tree_cfg.get("tree_shape") or []),
            "segment_length_M": int(gear_tree_cfg.get("segment_length", 0) or 0),
            "actor_loss_mode": str(actor_loss_mode),
        }
    )
    return manifest


def update_manifest_from_edges(
    manifest: RunManifest,
    sampled_edges: List[Dict[str, Any]],
    *,
    strict: bool,
) -> Dict[str, Any]:
    """PLAN.md P0.N7/N8: check group integrity + emit metrics.

    Always records failures on the manifest (so a non-strict run still
    leaves evidence for the run manifest). In strict mode the raised
    ValueError propagates so the trainer stops before the actor step
    corrupts state.
    """
    integrity_metrics: Dict[str, Any] = {}
    raised: Exception | None = None
    try:
        integrity_metrics = validate_group_integrity(
            sampled_edges, strict_fresh_iid=strict
        )
    except ValueError as exc:
        raised = exc
        integrity_metrics = {
            "vdra/group_integrity_failures": 1,
            "vdra/group_integrity_error": str(exc),
        }
    failures = int(integrity_metrics.get("vdra/group_integrity_failures", 0) or 0)
    if failures:
        manifest.record_integrity_failure(failures)
    if raised is not None and strict:
        raise raised
    integrity_metrics.update(compute_group_metrics(sampled_edges))
    return integrity_metrics
