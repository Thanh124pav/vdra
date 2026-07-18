"""PLAN.md P0.3: precomputed objective_weights on the full update batch."""

from __future__ import annotations

import pytest

from recipe.gear_tree.tree_data import (
    compute_objective_weights,
    validate_objective_weights,
)


def _edge(tree_id: str, parent_group_id: str, m: int = 1) -> dict:
    return {
        "tree_id": tree_id,
        "parent_group_id": parent_group_id,
        "sample_multiplicity": m,
    }


def test_single_tree_two_parents_uniform_normalization():
    edges = [
        _edge("T", "p0"),
        _edge("T", "p0"),
        _edge("T", "p1"),
        _edge("T", "p1"),
    ]
    w = compute_objective_weights(edges)
    # N_tree=1, |P|=2. Each row 1/(1*2*2) = 0.25.
    assert w == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sum(w) == pytest.approx(1.0)
    validate_objective_weights(edges, w)


def test_non_uniform_branch_factor_does_not_change_parent_importance():
    # PLAN.md P0.3 acceptance: parent with 1 child gets same total mass as
    # parent with 3 children in the same tree.
    edges = [
        _edge("T", "p0"),           # single child
        _edge("T", "p1"),
        _edge("T", "p1"),
        _edge("T", "p1"),           # three children
    ]
    w = compute_objective_weights(edges)
    p0_mass = sum(w[:1])
    p1_mass = sum(w[1:])
    # Both parents claim half the tree.
    assert p0_mass == pytest.approx(0.5)
    assert p1_mass == pytest.approx(0.5)
    assert sum(w) == pytest.approx(1.0)


def test_multiplicity_shifts_within_parent_only():
    # sample_multiplicity moves child fractions inside the parent, but the
    # parent's mass stays 1/(N_tree * |P|).
    edges = [
        _edge("T", "p0", m=1),
        _edge("T", "p0", m=3),
    ]
    w = compute_objective_weights(edges)
    assert sum(w) == pytest.approx(1.0)  # single parent, single tree
    # Ratios follow multiplicity.
    assert w[1] / w[0] == pytest.approx(3.0)


def test_multi_tree_normalization():
    edges = [
        _edge("T0", "a"),
        _edge("T0", "b"),
        _edge("T1", "c"),
        _edge("T1", "d"),
        _edge("T1", "d"),
    ]
    w = compute_objective_weights(edges)
    # Each tree contributes 1/2 mass.
    t0 = w[0] + w[1]
    t1 = w[2] + w[3] + w[4]
    assert t0 == pytest.approx(0.5)
    assert t1 == pytest.approx(0.5)
    assert sum(w) == pytest.approx(1.0)


def test_variable_segment_lengths_do_not_affect_weights():
    # Weights are computed before batching; segment lengths / token counts
    # are absorbed by the token-mean per child at loss time.
    edges = [
        _edge("T", "p0"),
        _edge("T", "p0"),
    ]
    w = compute_objective_weights(edges)
    # Weights are identical regardless of token counts (which live elsewhere).
    assert w[0] == pytest.approx(w[1])


def test_validate_rejects_broken_weights():
    edges = [_edge("T", "p0"), _edge("T", "p0")]
    with pytest.raises(ValueError, match="batch sum"):
        validate_objective_weights(edges, [0.1, 0.1])
