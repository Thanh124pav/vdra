"""PLAN.md P0.N7 (metrics) + P0.R3 + P1.R6 + P1.R7 CPU coverage."""

from __future__ import annotations

import pytest
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

from vdra_core import allocate_branch_factors

from recipe.gear_tree.tree_advantage import extract_edges_from_tree
from recipe.gear_tree.tree_data import compute_group_metrics


def _tree(k_a: int, k_b: int) -> dict:
    return {
        "reward": 0.5,
        "reward_std": 0.25,
        "full_text": "Q",
        "_request_object": {
            "_treetune__idx": 1,
            "policy_snapshot_id": "snap:1",
        },
        "gear_segment_id": "root",
        "vdra_allocated_k": 2,
        "children": [
            {
                "text": "A",
                "full_text": "QA",
                "reward": 0.9,
                "leaf": False,
                "gear_segment_id": "root/0/0",
                "vdra_allocated_k": k_a,
                "response_token_ids": [1, 2],
                "actor_shifted_log_probs": [-0.1, -0.2],
                "children": [
                    {
                        "text": f"A{i}",
                        "full_text": f"QAA{i}",
                        "reward": 0.9,
                        "leaf": True,
                        "gear_segment_id": f"root/0/0/1/{i}",
                        "response_token_ids": [3, 4],
                        "actor_shifted_log_probs": [-0.3, -0.4],
                    }
                    for i in range(k_a)
                ],
            },
            {
                "text": "B",
                "full_text": "QB",
                "reward": 0.4,
                "leaf": False,
                "gear_segment_id": "root/0/1",
                "vdra_allocated_k": k_b,
                "response_token_ids": [5, 6],
                "actor_shifted_log_probs": [-0.5, -0.6],
                "children": [
                    {
                        "text": f"B{i}",
                        "full_text": f"QBB{i}",
                        "reward": 0.4,
                        "leaf": True,
                        "gear_segment_id": f"root/0/1/1/{i}",
                        "response_token_ids": [7, 8],
                        "actor_shifted_log_probs": [-0.7, -0.8],
                    }
                    for i in range(k_b)
                ],
            },
        ],
    }


def test_compute_group_metrics_reports_parent_and_tree_counts():
    edges = extract_edges_from_tree(_tree(k_a=2, k_b=4), only_adv_greater_than_zero=False)
    metrics = compute_group_metrics(edges)
    # 3 parents in this tree (root + A + B).
    assert metrics["vdra/parent_groups_per_tree"] == 3.0
    assert metrics["vdra/trees_in_batch"] == 1.0
    # child_weight_sum_per_parent normalises to 1 under fresh_iid.
    assert metrics["vdra/child_weight_sum_per_parent"] == 1.0
    # parent_weight_sum_per_tree also normalises to 1.
    assert metrics["vdra/parent_weight_sum_per_tree"] == 1.0


def test_effective_segment_weight_anti_correlates_with_branch_factor():
    # Under fresh_iid node-balanced aggregation, a segment's effective
    # weight is 1/(|P(T)| * k_p), so seg weight and branch factor must be
    # negatively correlated. The Pearson corr is not exactly -1 whenever
    # more than two distinct k values appear because the relationship is
    # 1/k (nonlinear); we assert the sign and magnitude only.
    edges = extract_edges_from_tree(_tree(k_a=1, k_b=3), only_adv_greater_than_zero=False)
    metrics = compute_group_metrics(edges)
    corr = metrics["vdra/effective_segment_weight_vs_branch_factor_corr"]
    assert corr < -0.9, corr


def test_bounded_allocation_reports_slack_instead_of_raising():
    # Two nodes, upper cap 3 each => upper_sum=6. Request 100 exact-budget:
    # old behaviour raised; PLAN.md P0.R3 now spends 6 and reports slack=94.
    nodes = [
        {"id": "n0", "vdra_dispersion_C": 0.3, "vdra_default_k": 3},
        {"id": "n1", "vdra_dispersion_C": 0.2, "vdra_default_k": 3},
    ]
    result = allocate_branch_factors(
        nodes,
        total_budget=100,
        n_min=1,
        max_k_per_node=3,
        predicted_k_cap_mode="configured_max_for_all_nodes",
        infeasible_upper_policy="expand_nonredundant_caps",
    )
    assert result.allocated_budget == 6
    assert result.underallocated_budget == 94


def test_bounded_allocation_still_raises_when_below_lower_bound():
    nodes = [
        {"id": "n0", "vdra_dispersion_C": 0.5, "vdra_default_k": 5},
        {"id": "n1", "vdra_dispersion_C": 0.5, "vdra_default_k": 5},
    ]
    with pytest.raises(ValueError, match="below lower-bound"):
        allocate_branch_factors(nodes, total_budget=1, n_min=2, max_k_per_node=5)


class _StubScorer:
    """Placeholder scorer for GearGate config-only tests."""


def test_gear_gate_accepts_pilot_equal_to_default_branch_factor():
    # PLAN.md P1.R6: (pilot=8, tree=[8,8,8]) must no longer be rejected.
    from recipe.gear_tree.gear_gate import GearGate

    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_StubScorer(),
        pilot_branch_factor=8,
        strict_vdra=True,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
    )
    # Should not raise even though pilot == max_default_branch_factor.
    gate.validate_main_config(max_default_branch_factor=8, segment_length=100)


def test_gear_gate_rejects_weighted_reuse_in_strict_main_run():
    from recipe.gear_tree.gear_gate import GearGate

    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_StubScorer(),
        pilot_branch_factor=8,
        pilot_execution_mode="weighted_reuse",
        strict_vdra=True,
    )
    with pytest.raises(ValueError, match="weighted_reuse"):
        gate.validate_main_config(max_default_branch_factor=8, segment_length=100)


def test_gear_gate_rejects_depth_batch_in_strict_main_run():
    from recipe.gear_tree.gear_gate import GearGate

    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_StubScorer(),
        pilot_branch_factor=8,
        allocation_runtime="depth_batch",
        strict_vdra=True,
    )
    with pytest.raises(ValueError, match="depth_batch"):
        gate.validate_main_config(max_default_branch_factor=8, segment_length=100)
