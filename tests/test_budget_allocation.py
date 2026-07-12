import math

import pytest

from treetune.gear.budget_allocation import (
    allocate_branch_factors,
    apply_tail_correction,
    reward_variance_from_pair_tvs,
    simulation_lemma_gap,
    value_gap_bound,
)


def test_simulation_lemma_gap_formula():
    tv = 0.2
    gamma = 0.5
    expected = gamma * tv / ((1 - gamma) * (1 - gamma + tv))
    assert simulation_lemma_gap(tv, gamma) == pytest.approx(expected)


def test_apply_tail_correction():
    assert apply_tail_correction(0.0, 0.0) == 0.0
    assert apply_tail_correction(0.3, 0.0) == pytest.approx(0.3)
    assert apply_tail_correction(0.3, 0.5) == pytest.approx(0.3 + 0.7 * 0.5)
    # eps_tail = 1 saturates the bound regardless of the short-horizon TV.
    assert apply_tail_correction(0.3, 1.0) == pytest.approx(1.0)
    # Inputs are clamped into [0, 1].
    assert apply_tail_correction(1.7, 0.0) == 1.0
    assert apply_tail_correction(-0.2, 0.0) == 0.0


def test_value_gap_bound_linear_default():
    # Default: |V_i - V_j| <= r_max * TV, no gamma amplification.
    assert value_gap_bound(0.3) == pytest.approx(0.3)
    assert value_gap_bound(0.3, r_max=2.0) == pytest.approx(0.6)
    # eps_tail widens the bound before scaling.
    assert value_gap_bound(0.3, eps_tail=0.5) == pytest.approx(0.3 + 0.7 * 0.5)


def test_value_gap_bound_simulation_lemma_form_is_clamped():
    # gamma=0.9, tv=1.0: raw gap = 0.9/(0.1*1.1) ≈ 8.18 must clamp to r_max.
    bound = value_gap_bound(1.0, gamma=0.9, r_max=1.0, bound_form="simulation_lemma")
    assert bound == pytest.approx(1.0)
    # Small TV keeps the un-clamped discounted form.
    tv = 0.001
    raw = simulation_lemma_gap(tv, 0.9)
    assert raw < 1.0
    assert value_gap_bound(
        tv, gamma=0.9, r_max=1.0, bound_form="simulation_lemma"
    ) == pytest.approx(raw)


def test_value_gap_bound_rejects_unknown_form():
    with pytest.raises(ValueError):
        value_gap_bound(0.3, bound_form="quadratic")


def test_reward_variance_matches_summary_normalization():
    # Summary.md §8: C_s = sum_{i<j} B_ij^2 / n^2 with B = value_gap_bound(TV).
    pair_tvs = {(0, 1): 0.2, (0, 2): 0.4, (1, 2): 0.1}
    expected = sum(tv * tv for tv in pair_tvs.values()) / 9.0
    assert reward_variance_from_pair_tvs(pair_tvs, n=3, gamma=0.5) == pytest.approx(expected)


def test_reward_variance_canonicalizes_ordered_pairs():
    unordered = {(0, 1): 0.2, (1, 2): 0.4}
    with_duplicates = {(0, 1): 0.2, (1, 0): 0.2, (1, 2): 0.4, (2, 2): 0.9}
    assert reward_variance_from_pair_tvs(
        with_duplicates, n=3, gamma=0.5
    ) == pytest.approx(reward_variance_from_pair_tvs(unordered, n=3, gamma=0.5))


def test_reward_variance_honors_bound_options():
    pair_tvs = {(0, 1): 0.2}
    expected_gap = value_gap_bound(
        0.2, gamma=0.9, r_max=2.0, eps_tail=0.1, bound_form="simulation_lemma"
    )
    assert reward_variance_from_pair_tvs(
        pair_tvs, n=2, gamma=0.9, r_max=2.0, eps_tail=0.1, bound_form="simulation_lemma"
    ) == pytest.approx(expected_gap * expected_gap / 4.0)


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

    # Zero-margin nodes keep zero weight but still receive the n_min floor.
    assert summary.weights == {"equal": 0.0}
    assert summary.allocations == {"equal": 2}


def test_allocation_floor_prevents_discarding_zero_dispersion_nodes():
    # Summary.md §10: k_s = n_min + (B - |Q|*n_min) * w_s / sum w.  A node with
    # C_s = 0 must keep the floor instead of being pruned to zero.
    nodes = [
        {"gear_segment_id": "flat", "gear_reward_variance": 0.0},
        {"gear_segment_id": "spread", "gear_reward_variance": 0.9},
    ]

    summary = allocate_branch_factors(
        nodes, total_budget=8, lambda_=0.0, n_min=1, distribute_remainder=True
    )

    assert summary.allocations["flat"] == 1
    assert summary.allocations["spread"] == 7
    assert summary.allocated_budget == 8
    assert summary.underallocated_budget == 0


def test_allocation_floor_falls_back_to_even_split_when_floor_exceeds_budget():
    nodes = [
        {"gear_segment_id": "a", "gear_reward_variance": 0.9},
        {"gear_segment_id": "b", "gear_reward_variance": 0.1},
        {"gear_segment_id": "c", "gear_reward_variance": 0.5},
    ]

    summary = allocate_branch_factors(nodes, total_budget=4, lambda_=0.0, n_min=2)

    assert sum(summary.allocations.values()) == 4
    assert all(v >= 1 for v in summary.allocations.values())


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
