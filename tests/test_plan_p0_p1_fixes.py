"""Behavioral CPU tests for PLAN.md P0/P1 fixes.

Covers:
    * P1.6 — simulation_lemma_gap numeric formula
    * P1.7 — strict-mode rejects n_min=0
    * P1.3 — run manifest validity built from evidence and sticky-False
    * P0.1 — GearGate.bind_snapshot rollout/scorer server-version handshake
    * P0.2 — one L_edge_max resolved end to end
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vdra_core import (
    build_run_manifest,
    compute_run_valid_for_main_results,
    simulation_lemma_gap,
    value_gap_bound,
)
from vdra_core.logging_schema import persist_vdra_artifacts


# The gate and trainer tests reach into recipe.gear_tree.* which transitively
# imports verl and omegaconf. Skip cleanly on a lighter CPU env rather than
# failing collection.
_gate_mod = pytest.importorskip(
    "recipe.gear_tree.gear_gate",
    reason="verl+recipe.gear_tree required",
    exc_type=ImportError,
)


# --------------------------------------------------------------------------- #
# P1.6 — Simulation Lemma denominator
# --------------------------------------------------------------------------- #
def test_p16_simulation_lemma_denominator_uses_gamma_times_tv():
    # Reference numeric check: gamma=0.5, tv=0.25 ->
    # gap = gamma * tv / ((1-gamma) * (1-gamma + gamma*tv))
    #     = 0.5 * 0.25 / (0.5 * (0.5 + 0.125))
    #     = 0.125 / (0.5 * 0.625)
    #     = 0.4
    assert simulation_lemma_gap(0.25, 0.5) == pytest.approx(0.4, rel=1e-12)


def test_p16_value_gap_bound_uses_corrected_formula_before_clamp():
    # gamma=0.5, tv=0.1, r_max=2.0
    # gap = 0.5*0.1 / (0.5 * (0.5 + 0.05)) = 0.05 / 0.275 ≈ 0.18181818
    # value_gap_bound scales by min(r_max, ...) but never clamps here (< 1).
    expected = 2.0 * (0.5 * 0.1) / (0.5 * (0.5 + 0.5 * 0.1))
    got = value_gap_bound(0.1, gamma=0.5, r_max=2.0, bound_form="simulation_lemma")
    assert got == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# P1.7 — strict-mode rejects n_min=0
# --------------------------------------------------------------------------- #
def test_p17_strict_gate_rejects_n_min_zero():
    GearGate = _gate_mod.GearGate

    gate = GearGate(
        k_algorithm="budget_allocation",
        n_min=0,
        pilot_branch_factor=8,
        scorer=SimpleNamespace(server_weight_version=None),
        strict_vdra=True,
    )
    with pytest.raises(ValueError, match="n_min"):
        gate.validate_main_config(max_default_branch_factor=6, segment_length=100)


def test_p17_strict_gate_accepts_n_min_one():
    GearGate = _gate_mod.GearGate

    gate = GearGate(
        k_algorithm="budget_allocation",
        n_min=1,
        pilot_branch_factor=8,
        scorer=SimpleNamespace(server_weight_version=None),
        strict_vdra=True,
    )
    # Must NOT raise. The other strict checks pass because pilot_branch_factor
    # (8) > max_default_branch_factor (6) and rollout temp/top_p are 1.0.
    gate.validate_main_config(max_default_branch_factor=6, segment_length=100)


# --------------------------------------------------------------------------- #
# P1.3 — Run manifest validity
# --------------------------------------------------------------------------- #
def _full_valid_evidence(**overrides):
    base = {
        "policy_snapshot_id": "global_step:1",
        "rollout_server_weight_version": "sha:abc",
        "scorer_server_weight_version": "sha:abc",
        "weight_version_verified": True,
        "allocation_scope": "per_queue_flush_within_tree",
        "allocation_proxy": "vdra",
        "flush_depths": [0, 1],
        "pilot_execution_mode": "fresh_iid",
        "weighted_reuse_fallback_count": 0,
        "token_cap_hit_count": 0,
        "unexpected_fallback": False,
        "unexpected_token_cap_hit": False,
        "all_node_accounting_invariants_passed": True,
        "all_snapshot_invariants_passed": True,
        "context_contract_passed": True,
        "successful_actor_updates": 2,
        "rollout_iterations": 2,
    }
    base.update(overrides)
    return base


def test_p13_valid_manifest_computed_from_evidence():
    manifest = build_run_manifest(_full_valid_evidence())
    assert manifest["run_valid_for_main_results"] is True
    # All required fields are present in the returned manifest.
    from vdra_core import RUN_MANIFEST_REQUIRED_FIELDS

    for field in RUN_MANIFEST_REQUIRED_FIELDS:
        assert field in manifest


def test_p13_missing_weight_version_verified_invalidates_run():
    manifest = build_run_manifest(_full_valid_evidence(weight_version_verified=False))
    assert manifest["run_valid_for_main_results"] is False


def test_p13_oracle_proxy_invalidates_main_run():
    assert compute_run_valid_for_main_results(_full_valid_evidence(allocation_proxy="oracle")) is False


@pytest.mark.parametrize(
    "field",
    [
        "unexpected_fallback",
        "unexpected_token_cap_hit",
    ],
)
def test_p13_any_unexpected_flag_invalidates_run(field):
    assert compute_run_valid_for_main_results(_full_valid_evidence(**{field: True})) is False


@pytest.mark.parametrize(
    "field",
    [
        "all_node_accounting_invariants_passed",
        "all_snapshot_invariants_passed",
        "context_contract_passed",
    ],
)
def test_p13_missing_positive_invariant_invalidates_run(field):
    assert compute_run_valid_for_main_results(_full_valid_evidence(**{field: False})) is False


def _write_tree_manifest_stub(tmp_path: Path, run_valid: bool) -> Path:
    # Minimal tree that persist_vdra_artifacts accepts.
    tree = {
        "gear_algorithm_mode": "test",
        "gear_segment_id": "root",
        "vdra_default_k": 1,
        "vdra_allocated_k": 1,
        "children": [],
    }
    manifest = _full_valid_evidence()
    manifest = build_run_manifest(manifest)
    manifest["run_valid_for_main_results"] = run_valid
    persist_vdra_artifacts(tmp_path, tree, run_id="r", tree_id="t", run_manifest=manifest)
    return tmp_path / "run_manifest.json"


def test_p13_persist_never_weakens_prior_false_manifest(tmp_path):
    # First write invalidates the run.
    path = _write_tree_manifest_stub(tmp_path, run_valid=False)
    assert json.loads(path.read_text(encoding="utf-8"))["run_valid_for_main_results"] is False
    # Second write claims the run is valid; the stronger prior wins.
    _write_tree_manifest_stub(tmp_path, run_valid=True)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["run_valid_for_main_results"] is False


# --------------------------------------------------------------------------- #
# P0.1 — Server-version handshake in GearGate.bind_snapshot
# --------------------------------------------------------------------------- #
def _make_gate_with_scorer_server_version(server_version, *, strict=True):
    GearGate = _gate_mod.GearGate

    scorer = SimpleNamespace(server_weight_version=server_version)
    return GearGate(
        k_algorithm="budget_allocation",
        n_min=1,
        pilot_branch_factor=8,
        scorer=scorer,
        strict_vdra=strict,
    )


def test_p01_matching_server_versions_verify_and_do_not_raise():
    gate = _make_gate_with_scorer_server_version("sha:abc")
    gate.bind_snapshot(
        "global_step:1",
        weight_version_verified=True,
        rollout_server_weight_version="sha:abc",
    )
    assert gate.weight_version_verified is True
    assert gate.rollout_server_weight_version == "sha:abc"
    assert gate.scorer_server_weight_version == "sha:abc"


def test_p01_mismatched_server_versions_fail_strict_mode():
    gate = _make_gate_with_scorer_server_version("sha:aaa")
    with pytest.raises(RuntimeError, match="weight-version handshake"):
        gate.bind_snapshot(
            "global_step:1",
            weight_version_verified=True,
            rollout_server_weight_version="sha:bbb",
        )


def test_p01_missing_server_version_never_claims_verified_in_strict_mode():
    gate = _make_gate_with_scorer_server_version(None)
    # Caller cannot claim verified when the scorer replica gave no fingerprint.
    with pytest.raises(RuntimeError, match="server-reported weight fingerprint"):
        gate.bind_snapshot(
            "global_step:1",
            weight_version_verified=True,
            rollout_server_weight_version=None,
        )


def test_p01_static_model_id_only_is_not_a_verified_weight_version():
    # A static model id equal on both replicas but no reported fingerprint is
    # NOT sufficient for strict-mode verification.
    gate = _make_gate_with_scorer_server_version(None)
    gate.bind_snapshot(
        "global_step:1",
        weight_version_verified=False,
        rollout_server_weight_version=None,
    )
    assert gate.weight_version_verified is False


# --------------------------------------------------------------------------- #
# P0.2 — Unified context-length contract (exercised via the standalone helper
# so these tests do not require the full verl/transformers stack).
# --------------------------------------------------------------------------- #
from recipe.gear_tree.context_contract import (  # noqa: E402
    resolve_max_edge_prompt_length,
    resolve_max_original_prompt_length,
    validate_context_contract,
    worst_case_edge_prompt_length,
)


def _data_cfg(**kwargs):
    base = {"max_prompt_length": 1024, "max_response_length": 256}
    base.update({k: v for k, v in kwargs.items() if v is not None})
    return base


def test_p02_resolver_prefers_explicit_edge_cap():
    cfg = _data_cfg(max_edge_prompt_length=2048, max_original_prompt_length=1024)
    assert resolve_max_edge_prompt_length(cfg) == 2048
    assert resolve_max_original_prompt_length(cfg) == 1024


def test_p02_resolver_falls_back_to_max_prompt_length():
    cfg = _data_cfg(max_prompt_length=2000)
    assert resolve_max_edge_prompt_length(cfg) == 2000
    assert resolve_max_original_prompt_length(cfg) == 2000


def test_p02_worst_case_formula():
    # PLAN.md P0.2: L_original + (D-1) * M.
    assert worst_case_edge_prompt_length(
        max_original=1024, tree_shape=[6, 6, 6], segment_length=100
    ) == 1224
    assert worst_case_edge_prompt_length(
        max_original=1024, tree_shape=[6], segment_length=100
    ) == 1024


def test_p02_validator_rejects_edge_overflow():
    cfg = _data_cfg(max_edge_prompt_length=1000, max_original_prompt_length=1024)
    with pytest.raises(ValueError, match="max_edge_prompt_length"):
        validate_context_contract(
            data_cfg=cfg, tree_shape=[6, 6, 6], segment_length=100
        )


def test_p02_validator_accepts_boundary_equality():
    cfg = _data_cfg(max_edge_prompt_length=1224, max_original_prompt_length=1024)
    validate_context_contract(data_cfg=cfg, tree_shape=[6, 6, 6], segment_length=100)


def test_p02_validator_accepts_one_token_under_limit():
    cfg = _data_cfg(max_edge_prompt_length=1225, max_original_prompt_length=1024)
    validate_context_contract(data_cfg=cfg, tree_shape=[6, 6, 6], segment_length=100)


def test_p02_validator_rejects_one_token_overflow():
    cfg = _data_cfg(max_edge_prompt_length=1223, max_original_prompt_length=1024)
    with pytest.raises(ValueError, match="max_edge_prompt_length"):
        validate_context_contract(
            data_cfg=cfg, tree_shape=[6, 6, 6], segment_length=100
        )


def test_p02_validator_allows_dynamic_response_within_model_context():
    cfg = _data_cfg(
        max_edge_prompt_length=1500,
        max_original_prompt_length=500,
        max_response_length=1024,
    )
    validate_context_contract(
        data_cfg=cfg,
        tree_shape=[6, 6, 6],
        segment_length=100,
        model_context_length=2048,
    )


def test_p02_validator_rejects_edge_prompt_model_context_overflow():
    cfg = _data_cfg(
        max_edge_prompt_length=1500,
        max_original_prompt_length=500,
        max_response_length=1024,
    )
    with pytest.raises(ValueError, match="model context length"):
        validate_context_contract(
            data_cfg=cfg,
            tree_shape=[6, 6, 6],
            segment_length=100,
            model_context_length=1024,
        )


def test_p02_edges_to_dataproto_uses_same_limit(monkeypatch):
    """A dataset row whose query is 1 token longer than the resolved edge cap
    must be rejected before tensorization, matching the startup validator."""

    torch = pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("tensordict", exc_type=ImportError)
    pytest.importorskip("verl", exc_type=ImportError)
    from recipe.gear_tree.tree_data import edges_to_dataproto

    class _T:
        pad_token_id = 0
        eos_token_id = 0

    max_edge = 8
    edges = [
        {
            "query_token_ids": list(range(1, max_edge + 2)),  # length = max_edge+1
            "response_token_ids": [1, 2, 3],
            "question_id": "q",
            "instance": {},
        }
    ]
    with pytest.raises(ValueError, match="max_prompt_length"):
        edges_to_dataproto(
            edges, _T(), max_prompt_length=max_edge, max_response_length=16
        )


# --------------------------------------------------------------------------- #
# §6.1 #9 — Mixed-depth queue-flush test
# --------------------------------------------------------------------------- #
def test_mixed_depth_queue_flush_preserves_exact_budget():
    """PLAN.md §1.1: queues may contain nodes from multiple depths and the
    solver must preserve the exact flush budget."""
    import asyncio

    from vdra_core.online_budget import (
        OnlineQueueItem,
        RootQueueManager,
        SharedReservePool,
    )

    async def run():
        pool = SharedReservePool(queue_count=1)
        manager = RootQueueManager(
            queue_count=1,
            queue_capacity=3,
            timeout_seconds=99.0,
            reserve_pool=pool,
            policy_snapshot_id="p0",
        )
        # Three ready nodes from three different depths of the same tree.
        for depth, name in enumerate(["d0", "d1", "d2"]):
            node = {
                "vdra_node_id": name,
                "vdra_default_k": 4,
                "vdra_predicted_k": depth + 2,
                "vdra_dispersion_C": 0.1 * (depth + 1),
            }
            manager.enqueue(
                OnlineQueueItem(node, 4, depth, policy_snapshot_id="p0"), now=0.0
            )
        return await manager.flush_ready(now=0.0)

    flushed = asyncio.run(run())
    assert len(flushed) == 1
    result = flushed[0]
    # Exact bounded integer allocation must preserve the flush budget.
    assert sum(result.summary.allocations.values()) == result.summary.allocated_budget
    assert result.summary.allocated_budget == result.base_budget
    # Multiple distinct depths were legally coalesced into one flush.
    depths_in_flush = {item.depth for item in result.items}
    assert len(depths_in_flush) >= 2
