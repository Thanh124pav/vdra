import itertools
import statistics
import time

import pytest

from vdra_core.core import allocate_branch_factors, allocate_branch_factors_integer


def _node(name, default_k, predicted_k, dispersion_C):
    return {
        "vdra_node_id": name,
        "vdra_default_k": default_k,
        "vdra_predicted_k": predicted_k,
        "vdra_dispersion_C": dispersion_C,
    }


def _objective(nodes, allocation):
    c = {node["vdra_node_id"]: node["vdra_dispersion_C"] for node in nodes}
    return sum(c[key] / allocation[key] for key in allocation)


def _bruteforce(nodes, lower, upper, target):
    keys = [node["vdra_node_id"] for node in nodes]
    best = None
    for values in itertools.product(*(range(lower[key], upper[key] + 1) for key in keys)):
        if sum(values) != target:
            continue
        alloc = dict(zip(keys, values))
        value = _objective(nodes, alloc)
        if best is None or value < best[0] - 1e-12:
            best = (value, alloc)
    return best


def test_exact_budget_and_bounds():
    nodes = [_node("a", 6, 3, 0.1), _node("b", 6, 10, 1.0), _node("c", 6, 8, 0.3)]
    out = allocate_branch_factors(nodes, total_budget=18, n_min=1)
    assert sum(out.allocations.values()) == 18
    assert all(out.lower_bounds[k] <= v <= out.upper_bounds[k] for k, v in out.allocations.items())
    assert out.allocations["a"] <= 3
    assert out.allocations["b"] > 6


def test_unified_pruning_can_allocate_below_prediction_for_uncapped_node():
    nodes = [_node("low", 6, 9, 0.0), _node("hot", 6, 9, 10.0)]
    out = allocate_branch_factors(nodes, total_budget=12, n_min=1)
    assert out.allocations["low"] < 6
    assert out.allocations["hot"] > 6


def test_zero_dispersion_gets_slots_after_positive_gains_are_capped():
    nodes = [_node("zero", 1, 12, 0.0), _node("hot", 1, 12, 1.0)]
    out = allocate_branch_factors(nodes, total_budget=13, n_min=1, max_k_per_node=12)
    assert out.allocations["hot"] == 12
    assert out.allocations["zero"] == 1


def test_tie_determinism_uses_stable_node_id():
    nodes = [_node("b", 1, 12, 1.0), _node("a", 1, 12, 1.0)]
    out = allocate_branch_factors(nodes, total_budget=3, n_min=1)
    assert out.allocations == {"b": 1, "a": 2}


def test_infeasible_lower_sum_fails():
    with pytest.raises(ValueError, match="below lower-bound"):
        allocate_branch_factors([_node("a", 1, 1, 1.0), _node("b", 1, 1, 1.0)], total_budget=1, n_min=1)


def test_infeasible_upper_sum_fails_without_silent_budget_drop():
    with pytest.raises(ValueError, match="exceeds upper-bound"):
        allocate_branch_factors([_node("a", 6, 2, 1.0)], total_budget=6, n_min=1)


def test_bruteforce_optimality_small_instances():
    nodes = [_node("a", 4, 2, 0.2), _node("b", 4, 7, 1.5), _node("c", 4, 7, 0.7)]
    out = allocate_branch_factors_integer(nodes, total_budget=12, n_min=1, max_k_per_node=7)
    brute_value, brute_alloc = _bruteforce(nodes, out.lower_bounds, out.upper_bounds, 12)
    assert _objective(nodes, out.allocations) == pytest.approx(brute_value)
    assert out.allocations == brute_alloc


def test_marginal_monotonicity_and_objective_nonincrease():
    nodes = [_node("a", 1, 12, 4.0), _node("b", 1, 12, 1.0)]
    out = allocate_branch_factors(nodes, total_budget=8, n_min=1)
    assert 4.0 / (2 * 3) <= 4.0 / (1 * 2)
    assert out.objective_after <= out.objective_before + 1e-12
    assert out.solver_name == "bounded_marginal_integer"


def test_latency_smoke_for_normal_queue_size():
    nodes = [_node(str(i), 6, 12, 1.0 + i / 32.0) for i in range(32)]
    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        out = allocate_branch_factors(nodes, total_budget=384, n_min=1, max_k_per_node=16)
        timings.append((time.perf_counter() - t0) * 1000.0)
        assert out.allocated_budget == 384
    assert statistics.median(timings) < 5.0
