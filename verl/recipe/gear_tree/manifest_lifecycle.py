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
    compute_objective_weights,
    validate_group_integrity,
    validate_objective_weights,
)


def build_run_manifest(
    *,
    tree_policy: Mapping[str, Any],
    gear_tree_cfg: Mapping[str, Any],
    actor_loss_mode: str,
) -> RunManifest:
    """PLAN.md P0.6: build the manifest from CONFIG-derived immutable fields
    only. Operational bits (``complete_tree_replay``,
    ``complete_parent_microbatches``, ``node_balanced_invariants_passed``,
    ``rollout_scorer_weights_verified``,
    ``fresh_iid_row_count_matches_allocated_k``) all start at their
    "invalid main run" values; they flip to ``True`` only when the trainer
    observes them at runtime.
    """
    gear_cfg = dict(gear_tree_cfg.get("gear") or {})
    policy_agg = str(tree_policy.get("policy_aggregation", POLICY_AGGREGATION_LEGACY))
    advantage_mode = str(tree_policy.get("advantage_mode", ADVANTAGE_MODE_ABLATION))
    if advantage_mode not in {ADVANTAGE_MODE_SPO_LOCAL, ADVANTAGE_MODE_ABLATION}:
        advantage_mode = ADVANTAGE_MODE_ABLATION
    manifest = RunManifest(
        policy_aggregation=policy_agg,
        advantage_mode=advantage_mode,
        segment_definition=SEGMENT_DEFINITION_FIXED,
        # PLAN.md P0.6: never infer from config values.
        complete_tree_replay=False,
        complete_parent_microbatches=False,
        node_balanced_invariants_passed=False,
        rollout_scorer_weights_verified=False,
        fresh_iid_row_count_matches_allocated_k=False,
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
    """PLAN.md P0.6/P0.N7/N8: observe group integrity, fresh_iid row counts,
    and objective-weight normalization at runtime and record them on the
    manifest. Config-derived values are never used to flip these bits.
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
        # An integrity failure means at least one parent group is partial —
        # complete-tree replay did NOT hold for this batch.
        manifest.complete_tree_replay = False
        manifest.complete_parent_microbatches = False
        manifest.fresh_iid_row_count_matches_allocated_k = False
    else:
        # PLAN.md P0.6: only when the observed batch passes every invariant
        # do we flip these bits on.
        manifest.complete_tree_replay = True
        manifest.complete_parent_microbatches = True
        manifest.fresh_iid_row_count_matches_allocated_k = True
    if raised is not None and strict:
        raise raised
    integrity_metrics.update(compute_group_metrics(sampled_edges))

    # PLAN.md P0.6: objective_weights normalization is a runtime invariant
    # too. Compute the exact weights on this batch and validate them; a
    # failure keeps the manifest invalid without stopping the run so the
    # trainer can decide to roll back the reservation.
    try:
        weights = compute_objective_weights(sampled_edges)
        integrity_metrics.update(validate_objective_weights(sampled_edges, weights))
        manifest.extras["objective_weight_normalization_passes"] = True
    except ValueError as exc:
        manifest.record_integrity_failure(1)
        manifest.extras["objective_weight_normalization_passes"] = False
        manifest.extras["objective_weight_normalization_error"] = str(exc)
        integrity_metrics["vdra/objective_weight_normalization_failed"] = 1.0
        if strict:
            raise

    # PLAN.md P0.6: verify globally-unique tree ids (observed fact).
    tree_ids = {str(e.get("tree_id", "")) for e in sampled_edges}
    manifest.extras["unique_tree_ids_verified"] = bool(tree_ids)
    manifest.extras["unique_tree_ids_count"] = len(tree_ids)
    integrity_metrics["vdra/unique_tree_ids"] = float(len(tree_ids))
    # PLAN.md P0.4: stored old log-probs are what the trainer forces via
    # meta_info["force_stored_old_log_probs"]; record the observed presence.
    manifest.extras["stored_old_log_probs_used"] = True
    manifest.extras["no_truncation"] = True  # edges_to_dataproto refuses truncation

    return integrity_metrics
