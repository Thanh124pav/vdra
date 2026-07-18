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
POLICY_AGGREGATION_VDRA = "vdra_node_balanced"
POLICY_AGGREGATION_LEGACY = "legacy_token_mean"

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

    # Configuration snapshot
    policy_aggregation: str = POLICY_AGGREGATION_LEGACY
    advantage_mode: str = ADVANTAGE_MODE_ABLATION
    segment_definition: str = SEGMENT_DEFINITION_FIXED

    # Operational invariants (updated by the trainer)
    complete_tree_replay: bool = False
    complete_parent_microbatches: bool = False
    node_balanced_invariants_passed: bool = False
    rollout_scorer_weights_verified: bool = False
    fresh_iid_row_count_matches_allocated_k: bool = True

    # Running counters (updated by the trainer)
    parent_split_count: int = 0
    tree_split_count: int = 0
    group_integrity_failures: int = 0

    # Free-form extra provenance (e.g., dataset hash, GPU count).
    extras: Dict[str, Any] = field(default_factory=dict)

    def record_invariant_pass(self) -> None:
        self.node_balanced_invariants_passed = True

    def record_parent_split(self, count: int = 1) -> None:
        self.parent_split_count += int(count)

    def record_tree_split(self, count: int = 1) -> None:
        self.tree_split_count += int(count)

    def record_integrity_failure(self, count: int = 1) -> None:
        self.group_integrity_failures += int(count)

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
        m = cls(**data)
        m.extras = extras
        return m


def validate_main_run(manifest: RunManifest) -> Optional[str]:
    """PLAN.md P0.N8: return an error string when the manifest fails the
    main-run contract, or ``None`` when the run is a valid canonical VDRA
    main run.

    A main run is invalid when:
      * policy_aggregation != vdra_node_balanced;
      * any parent group is partial (parent_split_count > 0 or
        fresh_iid_row_count_matches_allocated_k=False);
      * any tree reduction used an undocumented fallback
        (node_balanced_invariants_passed=False);
      * node-balanced weights failed normalization
        (group_integrity_failures > 0);
      * rollout/scorer weight versions were not independently verified.
    """
    failures = []
    if manifest.policy_aggregation != POLICY_AGGREGATION_VDRA:
        failures.append(
            f"policy_aggregation={manifest.policy_aggregation!r} != {POLICY_AGGREGATION_VDRA!r}"
        )
    if manifest.parent_split_count > 0:
        failures.append(f"parent_split_count={manifest.parent_split_count} > 0")
    if manifest.tree_split_count > 0:
        failures.append(f"tree_split_count={manifest.tree_split_count} > 0")
    if manifest.group_integrity_failures > 0:
        failures.append(
            f"group_integrity_failures={manifest.group_integrity_failures} > 0"
        )
    if not manifest.node_balanced_invariants_passed:
        failures.append("node_balanced_invariants_passed=False")
    if not manifest.complete_tree_replay:
        failures.append("complete_tree_replay=False")
    if not manifest.complete_parent_microbatches:
        failures.append("complete_parent_microbatches=False")
    if not manifest.fresh_iid_row_count_matches_allocated_k:
        failures.append("fresh_iid_row_count_matches_allocated_k=False")
    if not manifest.rollout_scorer_weights_verified:
        failures.append("rollout_scorer_weights_verified=False")
    if failures:
        return (
            "Manifest is invalid for a canonical VDRA main run (PLAN.md P0.N8):\n  "
            + "\n  ".join(failures)
        )
    return None


def is_valid_main_run(manifest: RunManifest) -> bool:
    return validate_main_run(manifest) is None
