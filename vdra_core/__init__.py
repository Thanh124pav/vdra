"""Dependency-free mathematical core shared by treetune and verl VDRA."""

from .core import (
    AllocationSummary,
    allocate_branch_factors,
    apply_tail_correction,
    dispersion_bound_from_pair_tvs,
    largest_remainder_rounding,
    simulation_lemma_gap,
    value_gap_bound,
)
from .logging_schema import (
    node_allocated_k,
    summarize_vdra_tree,
    validate_node_accounting,
    write_node_accounting,
)

__all__ = [
    "AllocationSummary",
    "allocate_branch_factors",
    "apply_tail_correction",
    "dispersion_bound_from_pair_tvs",
    "largest_remainder_rounding",
    "simulation_lemma_gap",
    "node_allocated_k",
    "summarize_vdra_tree",
    "validate_node_accounting",
    "value_gap_bound",
    "write_node_accounting",
]
