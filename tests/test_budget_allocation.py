import math

import pytest

from treetune.gear.budget_allocation import (
    allocate_branch_factors,
    reward_variance_from_pair_tvs,
    simulation_lemma_gap,
)


def test_simulation_lemma_gap_formula():
    tv = 0.2
    gamma = 0.5
    expected = gamma * tv / ((1 - gamma) * (1 - gamma + tv))
    assert simulation_lemma_gap(tv, gamma) == pytest.approx(expected)


def test_reward_variance_uses_ordered_pair_normalization():
    pair_tvs = {(0, 1): 0.2, (0, 2): 0.4, (1, 2): 0.1}
    gamma = 0.5
    expected = sum(2 * simulation_lemma_gap(tv, gamma) ** 2 for tv in pair_tvs.values()) / (
        2 * 3 * 2
    )
    assert reward_variance_from_pair_tvs(pair_tvs, n=3, gamma=gamma) == pytest.approx(expected)


def test_allocate_branch_factors_keeps_floor_underallocation():
    nodes = [
        {"gear_segment_id": "a", "gear_reward_variance": 0.2},
        {"gear_segment_id": "b", "gear_reward_variance": 1.0},
    ]
    summary = allocate_branch_factors(nodes, total_budget=5, lambda_=0.02)
    assert summary.allocated_budget <= 5
    assert summary.underallocated_budget == 5 - summary.allocated_budget
    assert summary.allocations["b"] >= summary.allocations["a"]


def test_allocate_branch_factors_returns_n_min_below_lambda():
    nodes = [
        {"gear_segment_id": "below", "gear_reward_variance": 0.1},
        {"gear_segment_id": "above", "gear_reward_variance": 0.5},
    ]

    summary = allocate_branch_factors(
        nodes,
        total_budget=7,
        lambda_=0.02,
        n_min=2,
    )

    assert summary.weights["below"] == 0.0
    assert summary.weights["above"] == pytest.approx(math.sqrt(0.5**2 - 0.02))
    assert summary.allocations == {"below": 2, "above": 5}


def test_allocate_branch_factors_defaults_n_min_to_zero():
    nodes = [{"gear_segment_id": "below", "gear_reward_variance": 0.1}]

    summary = allocate_branch_factors(nodes, total_budget=5, lambda_=0.02)

    assert summary.allocations == {"below": 0}
    assert summary.underallocated_budget == 5


def test_allocate_branch_factors_handles_zero_margin():
    nodes = [{"gear_segment_id": "equal", "gear_reward_variance": 0.5}]

    summary = allocate_branch_factors(nodes, total_budget=5, lambda_=0.25, n_min=2)

    assert summary.weights == {"equal": 0.0}
    assert summary.allocations == {"equal": 0}


def test_allocate_branch_factors_handles_zero_budget_and_fallback_ids():
    nodes = [
        {"gear_reward_variance": 0.1},
        {"id": "explicit", "gear_reward_variance": 0.2},
    ]

    summary = allocate_branch_factors(nodes, total_budget=-3, lambda_=0.02)

    assert summary.requested_budget == 0
    assert summary.allocated_budget == 0
    assert summary.underallocated_budget == 0
    assert summary.allocations == {"node_0": 0, "explicit": 0}


def test_reward_variance_and_gap_clamp_degenerate_inputs():
    assert reward_variance_from_pair_tvs({}, n=1, gamma=0.5) == 0.0
    assert simulation_lemma_gap(tv=-1.0, gamma=0.5) == 0.0
    assert math.isfinite(simulation_lemma_gap(tv=0.2, gamma=1.0))


def test_allocate_branch_factors_can_distribute_largest_remainders():
    nodes = [
        {"gear_segment_id": "a", "gear_reward_variance": 0.6},
        {"gear_segment_id": "b", "gear_reward_variance": 0.6},
    ]

    summary = allocate_branch_factors(
        nodes,
        total_budget=5,
        lambda_=0.0,
        distribute_remainder=True,
    )

    assert summary.allocated_budget == 5
    assert sorted(summary.allocations.values()) == [2, 3]
    assert summary.underallocated_budget == 0


def test_allocate_branch_factors_can_use_simple_k_weight_override():
    nodes = [
        {"gear_segment_id": "low", "gear_allocation_weight_override": 1},
        {"gear_segment_id": "high", "gear_allocation_weight_override": 3},
    ]

    summary = allocate_branch_factors(
        nodes,
        total_budget=8,
        weight_key="gear_allocation_weight_override",
        distribute_remainder=True,
    )

    assert summary.allocations == {"low": 2, "high": 6}
