"""Simulation-lemma budget allocation utilities for GEAR.

This module is intentionally independent from the legacy SHARE/PRUNE triggers.
It converts pairwise TV estimates into per-node reward variance and then
allocates a depth budget across nodes with the floor-only rule requested for
budget-allocation runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


PairKey = Tuple[int, int]


@dataclass(frozen=True)
class AllocationSummary:
    allocations: Dict[str, int]
    weights: Dict[str, float]
    raw_allocations: Dict[str, float]
    requested_budget: int
    allocated_budget: int
    underallocated_budget: int


def simulation_lemma_gap(tv: float, gamma: float) -> float:
    """Return gamma*TV / ((1-gamma)*(1-gamma+TV))."""

    gamma = min(max(float(gamma), 0.0), 1.0 - 1e-8)
    tv = max(float(tv), 0.0)
    denom = (1.0 - gamma) * (1.0 - gamma + tv)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return gamma * tv / denom


def apply_tail_correction(tv: float, eps_tail: float) -> float:
    """Short-horizon -> full-horizon TV bound: D_L <= D_m + (1 - D_m) * eps_tail."""

    tv = min(max(float(tv), 0.0), 1.0)
    eps_tail = min(max(float(eps_tail), 0.0), 1.0)
    return tv + (1.0 - tv) * eps_tail


def value_gap_bound(
    tv: float,
    *,
    gamma: float = 0.9,
    r_max: float = 1.0,
    eps_tail: float = 0.0,
    bound_form: str = "linear",
) -> float:
    """Upper bound on |V(u_i) - V(u_j)| from a short-horizon TV estimate.

    Shared by the pruning path (``thresholds.tv_to_value_bound``) and the
    budget-allocation path (``reward_variance_from_pair_tvs``) so both use the
    same scale.  Steps:

      1. tail correction  tv_eff = tv + (1 - tv) * eps_tail;
      2. ``bound_form='linear'``: g = tv_eff — the exact TV bound for bounded
         terminal reward, |V_i - V_j| <= R_max * D_TV;
         ``bound_form='simulation_lemma'``: legacy discounted form
         g = gamma*tv_eff / ((1-gamma)*(1-gamma+tv_eff));
      3. clamp: values live in [0, R_max], so the bound is R_max * min(g, 1).
    """

    tv_eff = apply_tail_correction(tv, eps_tail)
    if bound_form == "linear":
        g = tv_eff
    elif bound_form == "simulation_lemma":
        g = simulation_lemma_gap(tv_eff, gamma)
    else:
        raise ValueError(f"Unknown bound_form: {bound_form!r}")
    return max(float(r_max), 0.0) * min(g, 1.0)


def _canonical_pair_items(
    pair_tvs: Mapping[PairKey, float],
) -> Dict[PairKey, float]:
    """Normalize keys to unordered ``i < j`` pairs, dropping diagonals."""

    out: Dict[PairKey, float] = {}
    for (i, j), tv in pair_tvs.items():
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        out.setdefault(key, float(tv))
    return out


def reward_variance_from_pair_tvs(
    pair_tvs: Mapping[PairKey, float],
    *,
    n: int,
    gamma: float,
    r_max: float = 1.0,
    eps_tail: float = 0.0,
    bound_form: str = "linear",
) -> float:
    """Node value-dispersion upper bound C_s from pairwise TV estimates.

    Implements Summary.md §8 with uniform pilot weights q_i = 1/n:

        C_s = 1/(2*n^2) * sum_{i,j} B_ij^2,

    where ``B_ij = value_gap_bound(TV_ij)``.  ``pair_tvs`` stores unordered
    pairs (canonicalized here); each contributes twice to the ordered sum and
    diagonal terms are zero, so C_s = sum_{i<j} B_ij^2 / n^2.
    """

    if n <= 1:
        return 0.0
    total = 0.0
    for tv in _canonical_pair_items(pair_tvs).values():
        gap = value_gap_bound(
            tv, gamma=gamma, r_max=r_max, eps_tail=eps_tail, bound_form=bound_form
        )
        total += gap * gap
    return total / (float(n) * float(n))


def _node_id(node: Mapping[str, Any], fallback: int) -> str:
    return str(
        node.get("gear_segment_id")
        or node.get("segment_id")
        or node.get("id")
        or f"node_{fallback}"
    )


def allocate_branch_factors(
    nodes: Sequence[Mapping[str, Any]],
    *,
    total_budget: int,
    lambda_: float = 0.0,
    n_min: int = 0,
    distribute_remainder: bool = False,
    weight_key: Optional[str] = None,
    fallback_uniform: bool = False,
) -> AllocationSummary:
    """Allocate branch factors per Summary.md §10 with an allocation floor.

    Every node first receives the floor ``n_min`` (the floor prevents a node
    from being completely discarded because of an inaccurate proxy), then the
    remaining budget ``B - |Q| * n_min`` is distributed proportionally to the
    node weights:

        k_s = n_min + (B - |Q| * n_min) * w_s / sum_j w_j,

    with ``w_s = sqrt(max(sigma_s^2 - lambda_, 0))`` and largest-remainder
    rounding when ``distribute_remainder=True``.

    NOTE — intentional deviation from treetune (per user request): treetune used
    ``sqrt(sigma_i^4 - lambda_)`` with ``lambda_ = 0.02``. This recipe uses
    ``sqrt(sigma_i^2 - lambda_)`` with default ``lambda_ = 0``.
    Queue-based online GEAR calls this with ``distribute_remainder=True`` so all
    queue resources are consumed by the largest fractional remainders.
    """

    total_budget = max(int(total_budget), 0)
    lambda_ = max(float(lambda_), 0.0)
    n_min = max(int(n_min), 0)
    weights: Dict[str, float] = {}
    allocations: Dict[str, int] = {}
    raw_allocations: Dict[str, float] = {}
    for idx, node in enumerate(nodes):
        node_id = _node_id(node, idx)
        if weight_key is not None and node.get(weight_key) is not None:
            weights[node_id] = max(float(node.get(weight_key) or 0.0), 0.0)
            continue

        sigma2 = max(float(node.get("gear_reward_variance", 0.0) or 0.0), 0.0)
        # Weight by sqrt(sigma^2 - lambda) (was sqrt(sigma^4 - lambda) in treetune).
        margin = sigma2 - lambda_
        weights[node_id] = math.sqrt(margin) if margin > 0.0 else 0.0

    weight_sum = sum(weights.values())
    if fallback_uniform and total_budget > 0 and weight_sum <= 0.0 and weights:
        for node_id in weights:
            weights[node_id] = 1.0
        weight_sum = sum(weights.values())

    # Floor: every node keeps at least n_min so no node is fully discarded.
    # If the floor alone exceeds the budget, fall back to an even split.
    if weights and n_min * len(weights) > total_budget:
        even, extra = divmod(total_budget, len(weights))
        for idx, node_id in enumerate(sorted(weights)):
            allocations[node_id] = even + (1 if idx < extra else 0)
            raw_allocations[node_id] = float(allocations[node_id])
    else:
        for node_id in weights:
            allocations[node_id] = n_min
        remaining_budget = max(total_budget - n_min * len(weights), 0)
        if remaining_budget > 0 and weight_sum > 0.0:
            for node_id, weight in weights.items():
                raw = remaining_budget * weight / weight_sum
                raw_allocations[node_id] = n_min + raw
                allocations[node_id] = n_min + int(math.floor(raw))

    for node_id, value in allocations.items():
        raw_allocations.setdefault(node_id, float(value))

    if distribute_remainder:
        leftover = max(total_budget - sum(allocations.values()), 0)
        fractional = sorted(
            (
                (
                    raw_allocations.get(node_id, 0.0)
                    - math.floor(raw_allocations.get(node_id, 0.0)),
                    node_id,
                )
                for node_id in weights
            ),
            key=lambda item: (-item[0], item[1]),
        )
        idx = 0
        while leftover > 0 and fractional:
            _, node_id = fractional[idx % len(fractional)]
            allocations[node_id] = allocations.get(node_id, 0) + 1
            leftover -= 1
            idx += 1

    for node_id in weights:
        allocations.setdefault(node_id, 0)
        raw_allocations.setdefault(node_id, 0.0)

    allocated = sum(allocations.values())
    return AllocationSummary(
        allocations=allocations,
        weights=weights,
        raw_allocations=raw_allocations,
        requested_budget=total_budget,
        allocated_budget=allocated,
        underallocated_budget=max(total_budget - allocated, 0),
    )
