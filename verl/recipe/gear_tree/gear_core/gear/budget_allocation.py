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


def reward_variance_from_pair_tvs(
    pair_tvs: Mapping[PairKey, float],
    *,
    n: int,
    gamma: float,
) -> float:
    """Compute Var(P) from unordered pairwise TV estimates.

    The requested formula is

        1 / (2*n*(n-1)) * sum_{i,j} gap(TV_ij)^2.

    ``pair_tvs`` normally stores unordered pairs with ``i < j``.  We multiply
    the unordered-pair contribution by two so the normalization matches the
    ordered ``sum_{i,j}``; diagonal terms are zero and omitted.
    """

    if n <= 1:
        return 0.0
    total = 0.0
    for tv in pair_tvs.values():
        gap = simulation_lemma_gap(tv, gamma)
        total += 2.0 * gap * gap
    return total / (2.0 * float(n) * float(n - 1))


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
    """Allocate branch factors with optional largest-remainder rounding.

    Nodes are weighted by ``sqrt(sigma_i^2 - lambda_)`` when the margin is
    non-negative, where ``sigma_i^2`` is stored as ``gear_reward_variance``.

    NOTE — intentional deviation from treetune (per user request): treetune used
    ``sqrt(sigma_i^4 - lambda_)`` with ``lambda_ = 0.02``. This recipe uses
    ``sqrt(sigma_i^2 - lambda_)`` with default ``lambda_ = 0``. Everything else
    (floor rounding, remainder distribution, ``n_min`` floor) is unchanged.
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
        if margin < 0.0:
            weights[node_id] = 0.0
            allocations[node_id] = n_min
        else:
            weights[node_id] = math.sqrt(margin)

    weight_sum = sum(weights.values())
    if fallback_uniform and total_budget > 0 and weight_sum <= 0.0 and weights:
        for node_id in weights:
            weights[node_id] = 1.0
        weight_sum = sum(weights.values())

    remaining_budget = max(total_budget - sum(allocations.values()), 0)
    if remaining_budget > 0 and weight_sum > 0.0:
        for node_id, weight in weights.items():
            if node_id in allocations:
                raw_allocations[node_id] = float(allocations[node_id])
                continue
            raw = remaining_budget * weight / weight_sum
            raw_allocations[node_id] = raw
            allocations[node_id] = int(math.floor(raw))

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
            frac, node_id = fractional[idx % len(fractional)]
            if frac <= 0.0 and weight_sum > 0.0:
                break
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
