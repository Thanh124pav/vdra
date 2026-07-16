import pytest

from vdra_core.core import allocate_branch_factors


def _nodes():
    return [
        {"vdra_node_id": "a", "vdra_default_k": 1, "vdra_predicted_k": 5, "vdra_dispersion_C": 1.0},
        {"vdra_node_id": "b", "vdra_default_k": 1, "vdra_predicted_k": 5, "vdra_dispersion_C": 0.7},
        {"vdra_node_id": "c", "vdra_default_k": 1, "vdra_predicted_k": 5, "vdra_dispersion_C": 0.2},
    ]


def test_integer_solver_preserves_budget_and_caps():
    out = allocate_branch_factors(_nodes(), total_budget=8, rounding_strategy="integer_marginal")
    assert sum(out.allocations.values()) == 8
    assert all(
        out.lower_bounds[key] <= value <= out.upper_bounds[key]
        for key, value in out.allocations.items()
    )


def test_old_rounding_strategy_names_fail_on_default_path():
    with pytest.raises(ValueError, match="Unknown rounding strategy"):
        allocate_branch_factors(_nodes(), total_budget=8, rounding_strategy="largest_remainder")


def test_unknown_rounding_strategy_fails():
    with pytest.raises(ValueError, match="Unknown rounding strategy"):
        allocate_branch_factors(_nodes(), total_budget=8, rounding_strategy="bad")
