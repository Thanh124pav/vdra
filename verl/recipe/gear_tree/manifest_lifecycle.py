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
    POLICY_AGGREGATION_SEGMENT_MEAN,
    POLICY_AGGREGATION_VDRA,
    SEGMENT_DEFINITION_FIXED,
    SEGMENT_TOKEN_REDUCTION_MEAN,
    RunManifest,
)
from recipe.gear_tree.tree_data import (
    compute_group_metrics,
    compute_objective_weights,
    validate_group_integrity,
    validate_objective_weights,
    validate_replay_batch,
    validate_tree_construction,
    verify_tree_instance_id_uniqueness,
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
    segment_reduction = str(
        tree_policy.get("segment_token_reduction", SEGMENT_TOKEN_REDUCTION_MEAN)
    )
    advantage_mode = str(tree_policy.get("advantage_mode", ADVANTAGE_MODE_ABLATION))
    if advantage_mode not in {ADVANTAGE_MODE_SPO_LOCAL, ADVANTAGE_MODE_ABLATION}:
        advantage_mode = ADVANTAGE_MODE_ABLATION
    manifest = RunManifest(
        policy_aggregation=policy_agg,
        segment_token_reduction=segment_reduction,
        advantage_mode=advantage_mode,
        segment_definition=SEGMENT_DEFINITION_FIXED,
        # PLAN.md P0.6: never infer operational bits from config values.
        complete_tree_replay=False,
        complete_parent_microbatches=False,
        node_balanced_invariants_passed=False,
        segment_count_invariants_passed=False,
        stored_old_log_probs_used=False,
        rollout_scorer_weights_verified=False,
        no_truncation=False,
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


def update_manifest_from_generated_edges(
    manifest: RunManifest,
    generated_edges: List[Dict[str, Any]],
    *,
    strict: bool,
) -> Dict[str, Any]:
    """PLAN.md P0.B: CONSTRUCTION-stage manifest update.

    Runs on the complete batch of edges extracted from freshly generated
    trees, before replay insertion. This is the only stage allowed to
    require complete parent groups and full-tree queue identities. The
    canonical path validates observed accounting facts only; objective
    weights are computed/validated solely for the node-balanced ablation
    (PLAN.md M4).
    """
    integrity_metrics: Dict[str, Any] = {}
    raised: Exception | None = None
    try:
        integrity_metrics = validate_tree_construction(
            generated_edges, strict_fresh_iid=strict
        )
    except ValueError as exc:
        raised = exc
        integrity_metrics = {
            "vdra/construction_failures": 1,
            "vdra/group_integrity_failures": 1,
            "vdra/queue_segment_identity_failures": 1.0,
            "vdra/construction_error": str(exc),
        }
    group_failures = int(
        integrity_metrics.get("vdra/group_integrity_failures", 0) or 0
    )
    if group_failures:
        manifest.record_integrity_failure(group_failures)
        manifest.fresh_iid_row_count_matches_allocated_k = False
    else:
        manifest.fresh_iid_row_count_matches_allocated_k = True
    queue_failures = int(
        float(integrity_metrics.get("vdra/queue_segment_identity_failures", 0) or 0)
    )
    if queue_failures:
        manifest.record_segment_count_failure(queue_failures)
    if raised is not None and strict:
        raise raised
    integrity_metrics.update(compute_group_metrics(generated_edges))

    # PLAN.md P0.C: parent-/tree-normalized node-balanced weights are an
    # ablation concept. Compute and validate them ONLY when the run actually
    # trains with the node-balanced loss — the canonical segment-mean path
    # must not validate (or depend on) these float weights at all.
    if (
        str(manifest.extras.get("actor_loss_mode", ""))
        == "vdra_node_balanced_ppo"
    ):
        try:
            weights = compute_objective_weights(generated_edges)
            integrity_metrics.update(
                validate_objective_weights(generated_edges, weights)
            )
            manifest.extras["objective_weight_normalization_passes"] = True
        except ValueError as exc:
            manifest.record_integrity_failure(1)
            manifest.extras["objective_weight_normalization_passes"] = False
            manifest.extras["objective_weight_normalization_error"] = str(exc)
            integrity_metrics["vdra/objective_weight_normalization_failed"] = 1.0
            # Non-fatal for the segment-mean main path.

    # PLAN.md P0.H: REAL tree-identity verification at construction time —
    # detects two stochastic trees for the same (question, snapshot)
    # colliding under one tree_id, and forbids the ambiguous
    # snapshot:question fallback identity. Far stronger than the old
    # "tree-id set is non-empty" check.
    tree_ids = {str(e.get("tree_id", "")) for e in generated_edges}
    ids_ok, id_failures = verify_tree_instance_id_uniqueness(generated_edges)
    manifest.extras["unique_tree_ids_verified"] = ids_ok and bool(tree_ids)
    manifest.extras["unique_tree_ids_count"] = len(tree_ids)
    manifest.unique_tree_ids_verified = ids_ok and bool(tree_ids)
    integrity_metrics["vdra/unique_tree_ids"] = float(len(tree_ids))
    integrity_metrics["vdra/tree_id_collisions"] = float(len(id_failures))
    if not ids_ok:
        manifest.extras["tree_id_collision_details"] = id_failures[:5]
        manifest.record_integrity_failure(len(id_failures))
        if strict:
            raise ValueError(
                "Tree-identity verification failed (PLAN.md P0.H):\n  "
                + "\n  ".join(id_failures)
            )

    # PLAN.md M4: the canonical segment-count claim rides on OBSERVED
    # construction accounting facts only — objective-weight normalization is
    # NOT a canonical dependency:
    #   (a) every parent realized its allocation BEFORE zero-advantage
    #       filtering (pre-filter realized_child_count == allocated_k;
    #       never the retained row count, which zero-filtering may shrink);
    #   (b) full-tree queue identity: per tree, the sum of unique
    #       queue_released_segment_count values equals
    #       tree_total_segment_count;
    #   (c) no missing or duplicate edge IDs in the generated batch;
    #   (d) no pruned placeholder counted as a trainable segment.
    duplicate_edge_ids = int(
        float(integrity_metrics.get("vdra/generated_duplicate_edge_ids", 0) or 0)
    )
    missing_edge_ids = sum(
        1 for e in generated_edges if not str(e.get("edge_id", ""))
    )
    pruned_rows = int(
        float(integrity_metrics.get("vdra/generated_pruned_rows", 0) or 0)
    )
    allocation_failures = _pre_filter_allocation_failures(generated_edges)
    integrity_metrics["vdra/generated_missing_edge_ids"] = float(missing_edge_ids)
    integrity_metrics["vdra/pre_filter_allocation_failures"] = float(
        allocation_failures
    )
    if (
        queue_failures == 0
        and allocation_failures == 0
        and duplicate_edge_ids == 0
        and missing_edge_ids == 0
        and pruned_rows == 0
    ):
        manifest.segment_count_invariants_passed = True

    return integrity_metrics


def _pre_filter_allocation_failures(edges: List[Dict[str, Any]]) -> int:
    """PLAN.md M4 fact (a): per parent group, the PRE-FILTER realized child
    count must equal ``allocated_k``.

    Uses the ``realized_child_count`` stamped on every edge at extraction
    time (unaffected by zero-advantage filtering). Edges without the stamp
    (legacy fixtures that predate zero-filtering) fall back to the retained
    row count, where retained == realized by construction.
    """
    from collections import defaultdict

    by_parent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_parent[str(edge.get("parent_group_id", ""))].append(edge)
    failures = 0
    for group in by_parent.values():
        allocated_values = {int(e.get("allocated_k", 0) or 0) for e in group}
        if len(allocated_values) != 1:
            failures += 1
            continue
        allocated = next(iter(allocated_values))
        if not allocated:
            continue
        realized_values = {
            int(e["realized_child_count"])
            for e in group
            if e.get("realized_child_count") is not None
        }
        if len(realized_values) > 1:
            failures += 1
            continue
        realized = next(iter(realized_values), len(group))
        if realized != allocated:
            failures += 1
    return failures


def update_manifest_from_replay_batch(
    manifest: RunManifest,
    sampled_edges: List[Dict[str, Any]],
    *,
    strict: bool,
    target_edges_per_iteration: int | None = None,
    max_edges_per_question_per_iteration: int | None = None,
    max_edge_age_iterations: int | None = None,
    current_rollout_iteration: int | None = None,
) -> Dict[str, Any]:
    """PLAN.md P0.B: REPLAY-stage manifest update (row-local checks only).

    Edge-level replay legitimately samples partial trees and partial parent
    groups, so this update never requires group completeness and never
    increments ``group_integrity_failures`` for partial sampling.
    """
    replay_metrics: Dict[str, Any] = {}
    raised: Exception | None = None
    try:
        replay_metrics = validate_replay_batch(
            sampled_edges,
            target_edges_per_iteration=target_edges_per_iteration,
            max_edges_per_question_per_iteration=(
                max_edges_per_question_per_iteration
            ),
            max_edge_age_iterations=max_edge_age_iterations,
            current_rollout_iteration=current_rollout_iteration,
            strict=strict,
        )
    except ValueError as exc:
        raised = exc
        replay_metrics = {
            "vdra/replay_batch_failures": 1,
            "vdra/replay_batch_error": str(exc),
        }
    failures = int(replay_metrics.get("vdra/replay_batch_failures", 0) or 0)
    if failures:
        manifest.record_replay_batch_failure(failures)
    if raised is not None and strict:
        raise raised

    # PLAN.md P0.7: replay age uses rollout_iteration when at least one
    # observed edge carries the canonical stamp.
    if any("generation_rollout_iteration" in edge for edge in sampled_edges):
        manifest.replay_age_uses_rollout_iteration = True
    # PLAN.md P0.J: `stored_old_log_probs_used` and `no_truncation` are NOT
    # set here. The trainer flips them only from actually observed runtime
    # events: the actor's actor/used_stored_old_log_probs metric and a
    # successful strict tensorization respectively.

    return replay_metrics


def update_manifest_from_edges(
    manifest: RunManifest,
    sampled_edges: List[Dict[str, Any]],
    *,
    strict: bool,
) -> Dict[str, Any]:
    """Deprecated P0.B alias: treats the batch as a COMPLETE generated batch.

    Kept for backwards compatibility with callers written before the
    construction/replay validator split. The production trainer calls
    :func:`update_manifest_from_generated_edges` at generation time and
    :func:`update_manifest_from_replay_batch` on sampled batches instead.
    """
    try:
        integrity_metrics = update_manifest_from_generated_edges(
            manifest, sampled_edges, strict=strict
        )
    except ValueError:
        manifest.complete_tree_replay = False
        manifest.complete_parent_microbatches = False
        raise
    failures = int(
        integrity_metrics.get("vdra/construction_failures", 0) or 0
    )
    manifest.complete_tree_replay = failures == 0
    manifest.complete_parent_microbatches = failures == 0
    if any("generation_rollout_iteration" in edge for edge in sampled_edges):
        manifest.replay_age_uses_rollout_iteration = True
    manifest.stored_old_log_probs_used = True
    manifest.no_truncation = True
    manifest.extras["stored_old_log_probs_used"] = True
    manifest.extras["no_truncation"] = True
    return integrity_metrics
