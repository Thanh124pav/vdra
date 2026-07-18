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
# PLAN.md P0.1 / P0.6: the main VDRA path uses `global_segment_mean`; the
# legacy parent-balanced value is preserved for ablation runs but is NOT a
# valid main-run choice.
POLICY_AGGREGATION_SEGMENT_MEAN = "global_segment_mean"
POLICY_AGGREGATION_VDRA = "vdra_node_balanced"  # deprecated main; kept as ablation
POLICY_AGGREGATION_LEGACY = "legacy_token_mean"

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

    # Operational invariants (updated by the trainer at runtime — never
    # inferred from config). PLAN.md P0.6.
    complete_tree_replay: bool = False
    complete_parent_microbatches: bool = False
    # Legacy: node_balanced_invariants_passed (kept for backwards compat).
    node_balanced_invariants_passed: bool = False
    # PLAN.md P0.6: canonical segment-count invariants (the main path).
    segment_count_invariants_passed: bool = False
    stored_old_log_probs_used: bool = False
    rollout_scorer_weights_verified: bool = False
    no_truncation: bool = False
    fresh_iid_row_count_matches_allocated_k: bool = True

    # Running counters (updated by the trainer)
    parent_split_count: int = 0
    tree_split_count: int = 0
    group_integrity_failures: int = 0
    segment_count_failures: int = 0

    # Free-form extra provenance (e.g., dataset hash, GPU count).
    extras: Dict[str, Any] = field(default_factory=dict)

    def record_invariant_pass(self) -> None:
        # Legacy alias — flips the segment-count bit on for the segment-mean
        # main path AND keeps the node-balanced bit compatible for ablation
        # manifests.
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
        return m


def validate_main_run(manifest: RunManifest) -> Optional[str]:
    """PLAN.md P0.6: return an error string when the manifest fails the
    main-run contract, or ``None`` when the run is a valid canonical VDRA
    main run.

    A main run is invalid when:
      * policy_aggregation != global_segment_mean;
      * segment_token_reduction is not exactly ``mean`` or ``sum``;
      * complete-tree replay was ever violated;
      * segment-count invariants failed
        (``segment_count_invariants_passed=False`` or
        ``segment_count_failures > 0``);
      * stored generation-time old log-probs were not used
        (``stored_old_log_probs_used=False``);
      * rollout/scorer weight versions were not independently verified;
      * silent truncation was observed (``no_truncation=False``).

    The legacy ``vdra_node_balanced`` aggregation remains a supported
    ablation configuration but never validates as a canonical main run —
    that is by design; the ablation manifest is expected to be labeled
    accordingly and reported as an ablation.
    """
    failures = []
    if manifest.policy_aggregation != POLICY_AGGREGATION_SEGMENT_MEAN:
        failures.append(
            f"policy_aggregation={manifest.policy_aggregation!r} != {POLICY_AGGREGATION_SEGMENT_MEAN!r}"
        )
    if manifest.segment_token_reduction not in _VALID_SEGMENT_TOKEN_REDUCTIONS:
        failures.append(
            f"segment_token_reduction={manifest.segment_token_reduction!r} not in {_VALID_SEGMENT_TOKEN_REDUCTIONS}"
        )
    if manifest.tree_split_count > 0:
        failures.append(f"tree_split_count={manifest.tree_split_count} > 0")
    if manifest.segment_count_failures > 0:
        failures.append(
            f"segment_count_failures={manifest.segment_count_failures} > 0"
        )
    if not manifest.segment_count_invariants_passed:
        failures.append("segment_count_invariants_passed=False")
    if not manifest.complete_tree_replay:
        failures.append("complete_tree_replay=False")
    if not manifest.stored_old_log_probs_used:
        failures.append("stored_old_log_probs_used=False")
    if not manifest.rollout_scorer_weights_verified:
        failures.append("rollout_scorer_weights_verified=False")
    if not manifest.no_truncation:
        failures.append("no_truncation=False")
    if failures:
        return (
            "Manifest is invalid for a canonical VDRA main run (PLAN.md P0.6):\n  "
            + "\n  ".join(failures)
        )
    return None


def is_valid_main_run(manifest: RunManifest) -> bool:
    return validate_main_run(manifest) is None
