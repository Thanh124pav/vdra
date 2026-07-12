import math

import pytest

from treetune.gear.budget_allocation import (
    allocate_branch_factors,
    apply_tail_correction,
    reward_variance_from_pair_tvs,
    simulation_lemma_gap,
    value_gap_bound,
)


def _node(name, default_k, predicted_k, dispersion_C):
    return {
        "vdra_node_id": name,
        "vdra_default_k": default_k,
        "vdra_predicted_k": predicted_k,
        "vdra_dispersion_C": dispersion_C,
    }


def test_tail_and_value_bounds():
    assert apply_tail_correction(0.3, 0.5) == pytest.approx(0.65)
    assert value_gap_bound(0.3, r_max=2.0) == pytest.approx(0.6)
    assert value_gap_bound(1.0, bound_form="simulation_lemma") == 1.0
    assert math.isfinite(simulation_lemma_gap(0.2, 1.0))


def test_dispersion_bound_matches_summary_normalization():
    pair_tvs = {(0, 1): 0.2, (0, 2): 0.4, (1, 2): 0.1}
    expected = sum(tv * tv for tv in pair_tvs.values()) / 9.0
    assert reward_variance_from_pair_tvs(pair_tvs, n=3, gamma=0.5) == pytest.approx(expected)


def test_pruning_only_reports_unallocated_residual():
    nodes = [_node("a", 6, 2, 0.1), _node("b", 6, 3, 1.0)]
    out = allocate_branch_factors(nodes, total_budget=12, n_min=1)
    assert out.base_allocations == {"a": 2, "b": 3}
    assert out.allocations == out.base_allocations
    assert out.saved_allocations == {"a": 4, "b": 3}
    assert out.underallocated_budget == 7


def test_residual_budget_moves_to_high_dispersion_demand():
    nodes = [
        _node("low", 6, 2, 0.01),
        _node("hot", 6, 10, 1.0),
        _node("warm", 6, 10, 0.25),
    ]
    out = allocate_branch_factors(nodes, total_budget=18, n_min=1)
    assert out.allocations["low"] == 2
    assert out.allocations["hot"] > out.allocations["warm"]
    assert sum(out.allocations.values()) == 18
    assert all(
        out.base_allocations[key] <= out.allocations[key] <= out.cap_allocations[key]
        for key in out.allocations
    )


def test_high_dispersion_cannot_exceed_demand_cap():
    nodes = [_node("capped", 4, 5, 100.0), _node("other", 4, 10, 0.5)]
    out = allocate_branch_factors(nodes, total_budget=8, n_min=1)
    assert out.allocations["capped"] <= 5
    assert out.allocations["other"] >= 3


def test_minimum_branch_factor_clamps_nonpositive_prediction():
    out = allocate_branch_factors([_node("a", 6, -2, 0.0)], total_budget=6, n_min=1)
    assert out.base_allocations["a"] == 1
    assert out.allocations["a"] == 1


def test_capped_largest_remainder_is_deterministic():
    nodes = [_node("a", 1, 5, 1.0), _node("b", 1, 5, 1.0)]
    out = allocate_branch_factors(nodes, total_budget=5, n_min=1)
    assert out.allocations == {"a": 3, "b": 2}
    assert out.allocated_budget == 5


def test_invalid_dispersion_fails_in_strict_mode():
    with pytest.raises(ValueError, match="Invalid VDRA dispersion"):
        allocate_branch_factors([_node("a", 1, 2, float("nan"))], total_budget=1)


def test_budget_below_mandatory_base_fails():
    with pytest.raises(ValueError, match="mandatory base"):
        allocate_branch_factors([_node("a", 4, 4, 1.0)], total_budget=3)
