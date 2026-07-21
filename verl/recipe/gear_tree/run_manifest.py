"""PLAN.md P0.N8: main-run manifest contract.

This module owns the small, well-typed record that every long / paper run
must carry alongside its metrics. Trainers build a manifest at fit()
startup, mutate it as invariants pass or fail (parent split counts,
node-balanced normalization, fresh_iid row_count == allocated_k, etc.), and
call :func:`validate_main_run` before the run is treated as a canonical
VDRA main run.

The manifest is intentionally engine-free: no verl / torch imports. That
lets both the trainer and offline analysis scripts read/write it via a
plain JSON round-trip.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# Canonical values used by the manifest — mirror the tree_policy config.
# PLAN.md §1.3 (2026-07-21): the main VDRA path uses the paper objectives
# `segment_mean` (default) or `token_mean`. The historical tree-balanced
# objective (formerly named `global_segment_mean`) and the parent-balanced
# value are preserved for ablation runs but are NOT valid main-run choices.
POLICY_AGGREGATION_SEGMENT_MEAN = "segment_mean"
POLICY_AGGREGATION_TOKEN_MEAN = "token_mean"
POLICY_AGGREGATION_TREE_BALANCED = "tree_balanced_segment_mean"  # ablation
POLICY_AGGREGATION_VDRA = "vdra_node_balanced"  # deprecated main; kept as ablation
POLICY_AGGREGATION_LEGACY = "legacy_token_mean"

# PLAN.md §5: what the last training iteration actually did. A boolean
# cannot distinguish a zero-signal skip from a postponed or empty iteration,
# so this status is the authoritative record; ``actor_update_skipped`` is
# derived from it.
ITERATION_STATUS_NOT_STARTED = "not_started"
ITERATION_STATUS_RUNNING = "running"
ITERATION_STATUS_UPDATED = "updated"
ITERATION_STATUS_ALL_ZERO_SKIPPED = "all_zero_skipped"
ITERATION_STATUS_ZERO_ACTIVE_SKIPPED = "zero_active_skipped"
ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED = "mixed_zero_signal_skipped"
ITERATION_STATUS_POSTPONED = "postponed"
ITERATION_STATUS_NO_SAMPLE = "no_sample"
ITERATION_STATUS_FAILED_BEFORE_ACTOR = "failed_before_actor"
ITERATION_STATUS_ACTOR_FAILED = "actor_failed"

VALID_ITERATION_STATUSES = (
    ITERATION_STATUS_NOT_STARTED,
    ITERATION_STATUS_RUNNING,
    ITERATION_STATUS_UPDATED,
    ITERATION_STATUS_ALL_ZERO_SKIPPED,
    ITERATION_STATUS_ZERO_ACTIVE_SKIPPED,
    ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED,
    ITERATION_STATUS_POSTPONED,
    ITERATION_STATUS_NO_SAMPLE,
    ITERATION_STATUS_FAILED_BEFORE_ACTOR,
    ITERATION_STATUS_ACTOR_FAILED,
)

# The statuses that mean "a reservation was consumed but no optimizer step
# ran because it carried no learning signal".
ZERO_SIGNAL_SKIP_STATUSES = (
    ITERATION_STATUS_ALL_ZERO_SKIPPED,
    ITERATION_STATUS_ZERO_ACTIVE_SKIPPED,
    ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED,
)

# PLAN.md P0.1: within-segment token reduction must be exactly one of these.
SEGMENT_TOKEN_REDUCTION_MEAN = "mean"
SEGMENT_TOKEN_REDUCTION_SUM = "sum"
_VALID_SEGMENT_TOKEN_REDUCTIONS = (
    SEGMENT_TOKEN_REDUCTION_MEAN,
    SEGMENT_TOKEN_REDUCTION_SUM,
)

ADVANTAGE_MODE_SPO_LOCAL = "spo_local"
ADVANTAGE_MODE_ABLATION = "configured_ablation"

SEGMENT_DEFINITION_FIXED = "fixed_length_M"
SEGMENT_DEFINITION_CUSTOM = "custom"


@dataclass
class RunManifest:
    """The scientific contract for one VDRA training run.

    All fields default to conservative "invalid main run" values so that
    a manifest that skipped some update remains detectable as invalid.
    """

    # Configuration snapshot (PLAN.md P0.6).
    policy_aggregation: str = POLICY_AGGREGATION_LEGACY
    segment_token_reduction: str = SEGMENT_TOKEN_REDUCTION_MEAN
    advantage_mode: str = ADVANTAGE_MODE_ABLATION
    segment_definition: str = SEGMENT_DEFINITION_FIXED

    # PLAN.md §14: objective-mask snapshot + the OBSERVED logical denominator
    # mode, so a run cannot claim one objective while normalizing by another.
    use_prob_mask: bool = True
    probability_mask_threshold: float = 0.9
    logical_slot_schema_version: int = 0
    # "" (not a canonical sparse run) | "segment_slots" |
    # "response_tokens" | "prob_mask_tokens"
    observed_logical_denominator: str = ""

    # PLAN.md P0.7 replay-cadence snapshot (declared config → observed cap).
    replay_sampling_unit: str = "edge"
    target_edges_per_iteration: int = 0
    resolved_max_edges_per_question_per_iteration: int = 0
    max_edge_age_iterations: int = 0
    ppo_mini_batch_size: int = 0
    ppo_epochs: int = 1

    # PLAN.md P0.7 counter snapshot (observed by the trainer).
    rollout_iteration: int = 0
    global_step: int = 0
    optimizer_steps_last_iteration: int = 0
    # PLAN.md §8: expectation counts TRAINABLE logical batches only.
    expected_optimizer_steps_last_iteration: int = 0
    num_optimizer_steps_total: int = 0

    # Operational invariants (updated by the trainer at runtime — never
    # inferred from config). PLAN.md P0.6 / P0.7.
    # Legacy (kept for backwards compat; not required for the main path):
    complete_tree_replay: bool = False
    complete_parent_microbatches: bool = False
    node_balanced_invariants_passed: bool = False
    # PLAN.md P0.6 / P0.7 canonical bits.
    segment_count_invariants_passed: bool = False
    stored_old_log_probs_used: bool = False
    rollout_scorer_weights_verified: bool = False
    no_truncation: bool = False
    fresh_iid_row_count_matches_allocated_k: bool = True
    replay_age_uses_rollout_iteration: bool = False
    optimizer_step_accounting_valid: bool = False
    # PLAN.md §5: what the LAST iteration actually did. A single boolean
    # cannot distinguish a zero-signal skip from a postponed or empty
    # iteration, so the status string is authoritative and
    # ``actor_update_skipped`` is DERIVED from it for compatibility.
    # Observational: neither gates main-run validity.
    last_iteration_status: str = ITERATION_STATUS_NOT_STARTED
    actor_update_skipped: bool = False
    unique_tree_ids_verified: bool = False

    # Running counters (updated by the trainer)
    parent_split_count: int = 0
    tree_split_count: int = 0
    group_integrity_failures: int = 0
    segment_count_failures: int = 0
    # PLAN.md P0.B: row-local failures observed on sampled replay batches
    # (missing metadata, duplicate edge ids, cap/target violations, bad
    # ages). Partial trees/parent groups are NOT failures at this stage.
    replay_batch_failures: int = 0

    # PLAN.md P0.7 replay diagnostics from the most recent iteration.
    selected_edges_last_iteration: int = 0
    unique_questions_last_iteration: int = 0
    mean_edge_age_last_iteration: float = 0.0
    max_edge_age_last_iteration: int = 0
    per_question_selected_count_max_last_iteration: int = 0
    zero_contribution_selected_slots_last_iteration: int = 0
    edge_age_histogram_last_iteration: Dict[int, int] = field(default_factory=dict)

    # Free-form extra provenance (e.g., dataset hash, GPU count).
    extras: Dict[str, Any] = field(default_factory=dict)

    def record_segment_invariant_pass(self) -> None:
        """PLAN.md P0.J: the canonical segment-mean invariant claim."""
        self.segment_count_invariants_passed = True

    def record_node_balanced_invariant_pass(self) -> None:
        """PLAN.md P0.J: the node-balanced ABLATION invariant claim."""
        self.node_balanced_invariants_passed = True

    def record_invariant_pass(self) -> None:
        # Deprecated legacy alias — the two bits are DIFFERENT claims
        # (PLAN.md P0.J); production code calls the specific recorder for
        # its configured loss mode. Kept only for pre-split callers.
        self.node_balanced_invariants_passed = True
        self.segment_count_invariants_passed = True

    def record_parent_split(self, count: int = 1) -> None:
        self.parent_split_count += int(count)

    def record_tree_split(self, count: int = 1) -> None:
        self.tree_split_count += int(count)

    def record_integrity_failure(self, count: int = 1) -> None:
        self.group_integrity_failures += int(count)

    def record_segment_count_failure(self, count: int = 1) -> None:
        self.segment_count_failures += int(count)

    def record_replay_batch_failure(self, count: int = 1) -> None:
        self.replay_batch_failures += int(count)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        extras = data.pop("extras", {})
        # Backwards-compat: older manifests do not include the new P0.6
        # fields. Drop unknown keys and rely on the dataclass defaults for
        # anything new so save/load remains lossless in both directions.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        cleaned = {k: v for k, v in data.items() if k in known}
        m = cls(**cleaned)
        m.extras = extras
        m.actor_update_skipped = m.last_iteration_status in ZERO_SIGNAL_SKIP_STATUSES
        return m


def validate_main_run(manifest: RunManifest) -> Optional[str]:
    """PLAN.md P0.6: return an error string when the manifest fails the
    main-run contract, or ``None`` when the run is a valid canonical VDRA
    main run.

    A main run is invalid when:
      * policy_aggregation is neither segment_mean nor token_mean
        (the tree_balanced_segment_mean / vdra_node_balanced /
        legacy_token_mean ablations never validate as main);
      * no successful outer actor update was observed
        (``global_step < 1``);
      * segment_token_reduction is not exactly ``mean`` or ``sum``;
      * complete-tree replay was ever violated;
      * segment-count invariants failed
        (``segment_count_invariants_passed=False`` or
        ``segment_count_failures > 0``);
      * stored generation-time old log-probs were not used
        (``stored_old_log_probs_used=False``);
      * rollout/scorer weight versions were not independently verified;
      * silent truncation was observed (``no_truncation=False``).

    Canonical validity hinges on ``global_step`` — the host VERL outer-update
    unit. The internal PPO ``num_optimizer_steps_total`` and the derived
    ``optimizer_step_accounting_valid`` are DIAGNOSTICS only and never gate
    the main run (PLAN.md M1/M4).

    The legacy ``vdra_node_balanced`` aggregation remains a supported
    ablation configuration but never validates as a canonical main run —
    that is by design; the ablation manifest is expected to be labeled
    accordingly and reported as an ablation.
    """
    failures = []
    if manifest.policy_aggregation not in (
        POLICY_AGGREGATION_SEGMENT_MEAN,
        POLICY_AGGREGATION_TOKEN_MEAN,
    ):
        failures.append(
            f"policy_aggregation={manifest.policy_aggregation!r} not in "
            f"({POLICY_AGGREGATION_SEGMENT_MEAN!r}, "
            f"{POLICY_AGGREGATION_TOKEN_MEAN!r})"
        )
    # PLAN.md §14: the OBSERVED logical denominator must match the selected
    # objective — a masked numerator normalized by the unmasked token count
    # (or vice versa) silently changes the objective.
    expected_denominator = {
        POLICY_AGGREGATION_SEGMENT_MEAN: "segment_slots",
        POLICY_AGGREGATION_TOKEN_MEAN: (
            "prob_mask_tokens" if manifest.use_prob_mask else "response_tokens"
        ),
    }.get(manifest.policy_aggregation)
    if expected_denominator is not None:
        if manifest.observed_logical_denominator != expected_denominator:
            failures.append(
                "observed_logical_denominator="
                f"{manifest.observed_logical_denominator!r} != "
                f"{expected_denominator!r} for policy_aggregation="
                f"{manifest.policy_aggregation!r} with use_prob_mask="
                f"{manifest.use_prob_mask}"
            )
    # PLAN.md P0.J: canonical replay is edge-level; a complete_tree run is a
    # labeled ablation, never a valid main run.
    if manifest.replay_sampling_unit != "edge":
        failures.append(
            f"replay_sampling_unit={manifest.replay_sampling_unit!r} != 'edge'"
        )
    # PLAN.md M4: canonical validity requires at least one successful OUTER
    # actor update (global_step >= 1) — the host-framework training unit.
    # num_optimizer_steps_total and optimizer_step_accounting_valid are
    # diagnostics and must NOT gate the main run.
    if manifest.global_step < 1:
        failures.append(f"global_step={manifest.global_step} < 1")
    if manifest.segment_token_reduction not in _VALID_SEGMENT_TOKEN_REDUCTIONS:
        failures.append(
            f"segment_token_reduction={manifest.segment_token_reduction!r} not in {_VALID_SEGMENT_TOKEN_REDUCTIONS}"
        )
    if manifest.segment_count_failures > 0:
        failures.append(
            f"segment_count_failures={manifest.segment_count_failures} > 0"
        )
    # PLAN.md P0.7: an observed group-integrity failure — even one — must
    # keep the run invalid. Edge-level replay is canonical, but a broken
    # parent group still means the batch's row alignment is wrong.
    if manifest.group_integrity_failures > 0:
        failures.append(
            f"group_integrity_failures={manifest.group_integrity_failures} > 0"
        )
    # PLAN.md P0.B: row-local replay-batch failures (duplicate ids, missing
    # metadata, cap/target/age violations) also invalidate the run. Partial
    # trees or parent groups in a sampled batch are NOT counted here.
    if manifest.replay_batch_failures > 0:
        failures.append(
            f"replay_batch_failures={manifest.replay_batch_failures} > 0"
        )
    if not manifest.segment_count_invariants_passed:
        failures.append("segment_count_invariants_passed=False")
    # PLAN.md P0.7: complete-tree replay is NOT canonical anymore; edge-level
    # replay is intentional. We keep the field for backwards-compat and only
    # require it in a labeled "complete-tree ablation" manifest.
    if not manifest.stored_old_log_probs_used:
        failures.append("stored_old_log_probs_used=False")
    if not manifest.rollout_scorer_weights_verified:
        failures.append("rollout_scorer_weights_verified=False")
    if not manifest.no_truncation:
        failures.append("no_truncation=False")
    # PLAN.md P0.7 canonical bits.
    if not manifest.replay_age_uses_rollout_iteration:
        failures.append("replay_age_uses_rollout_iteration=False")
    # PLAN.md M4: optimizer_step_accounting_valid is a DIAGNOSTIC, not a
    # validity requirement — intentionally NOT checked here.
    if not manifest.unique_tree_ids_verified:
        failures.append("unique_tree_ids_verified=False")
    if failures:
        return (
            "Manifest is invalid for a canonical VDRA main run (PLAN.md P0.6):\n  "
            + "\n  ".join(failures)
        )
    return None


def is_valid_main_run(manifest: RunManifest) -> bool:
    return validate_main_run(manifest) is None
