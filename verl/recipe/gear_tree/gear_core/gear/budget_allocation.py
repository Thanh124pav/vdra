"""VERL adapter for the shared VDRA mathematical core."""

from vdra_core import (
    AllocationSummary,
    allocate_branch_factors,
    apply_tail_correction,
    dispersion_bound_from_pair_tvs,
    largest_remainder_rounding,
    simulation_lemma_gap,
    value_gap_bound,
)

reward_variance_from_pair_tvs = dispersion_bound_from_pair_tvs

__all__ = [
    "AllocationSummary",
    "allocate_branch_factors",
    "apply_tail_correction",
    "dispersion_bound_from_pair_tvs",
    "largest_remainder_rounding",
    "reward_variance_from_pair_tvs",
    "simulation_lemma_gap",
    "value_gap_bound",
]
