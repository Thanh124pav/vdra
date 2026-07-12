import pytest

from vdra_core.core import allocate_branch_factors


def _nodes():
    return [
        {"vdra_node_id": "a", "vdra_default_k": 1, "vdra_predicted_k": 5, "vdra_dispersion_C": 1.0},
        {"vdra_node_id": "b", "vdra_default_k": 1, "vdra_predicted_k": 5, "vdra_dispersion_C": 0.7},
        {"vdra_node_id": "c", "vdra_default_k": 1, "vdra_predicted_k": 5, "vdra_dispersion_C": 0.2},
    ]


@pytest.mark.parametrize("strategy", ["largest_remainder", "nearest_repair", "stochastic"])
def test_rounding_strategies_preserve_budget_and_caps(strategy):
    out = allocate_branch_factors(
        _nodes(), total_budget=8, rounding_strategy=strategy, rounding_seed=7
    )
    assert sum(out.allocations.values()) == 8
    assert all(
        out.base_allocations[key] <= value <= out.cap_allocations[key]
        for key, value in out.allocations.items()
    )


def test_unknown_rounding_strategy_fails():
    with pytest.raises(ValueError, match="rounding strategy"):
        allocate_branch_factors(_nodes(), total_budget=8, rounding_strategy="bad")
