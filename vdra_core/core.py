"""Canonical VDRA bounds and capped residual-budget allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from .rounding import round_bounded
from .logging_schema import node_id, write_node_accounting

PairKey = Tuple[int, int]


@dataclass(frozen=True)
class AllocationSummary:
    allocations: Dict[str, int]
    weights: Dict[str, float]
    raw_allocations: Dict[str, float]
    requested_budget: int
    allocated_budget: int
    underallocated_budget: int
    base_allocations: Dict[str, int]
    cap_allocations: Dict[str, int]
    saved_allocations: Dict[str, int]
    unmet_demands: Dict[str, int]
    additional_allocations: Dict[str, int]
    dual_lambda: Optional[float]


def simulation_lemma_gap(tv: float, gamma: float) -> float:
    gamma = min(max(float(gamma), 0.0), 1.0 - 1e-8)
    tv = max(float(tv), 0.0)
    denom = (1.0 - gamma) * (1.0 - gamma + tv)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return gamma * tv / denom


def apply_tail_correction(tv: float, eps_tail: float) -> float:
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
    tv_eff = apply_tail_correction(tv, eps_tail)
    if bound_form == "linear":
        gap = tv_eff
    elif bound_form == "simulation_lemma":
        gap = simulation_lemma_gap(tv_eff, gamma)
    else:
        raise ValueError(f"Unknown bound_form: {bound_form!r}")
    return max(float(r_max), 0.0) * min(gap, 1.0)


def dispersion_bound_from_pair_tvs(
    pair_tvs: Mapping[PairKey, float],
    *,
    n: int,
    gamma: float,
    r_max: float = 1.0,
    eps_tail: float = 0.0,
    bound_form: str = "linear",
) -> float:
    """Return C_s = sum_{i<j} B_ij^2 / n^2 for uniform pilot weights."""

    if n <= 1:
        return 0.0
    canonical: Dict[PairKey, float] = {}
    for (i, j), tv in pair_tvs.items():
        if i == j:
            continue
        canonical.setdefault((i, j) if i < j else (j, i), float(tv))
    total = 0.0
    for tv in canonical.values():
        bound = value_gap_bound(
            tv,
            gamma=gamma,
            r_max=r_max,
            eps_tail=eps_tail,
            bound_form=bound_form,
        )
        total += bound * bound
    return total / float(n * n)


def largest_remainder_rounding(
    raw: Mapping[str, float],
    *,
    lower: Mapping[str, int],
    upper: Mapping[str, int],
    target: int,
) -> Dict[str, int]:
    """Round a bounded continuous allocation while preserving its target."""

    out = {
        key: min(max(int(math.floor(raw[key])), lower[key]), upper[key])
        for key in raw
    }
    remaining = max(int(target) - sum(out.values()), 0)
    order = sorted(
        raw,
        key=lambda key: (-(raw[key] - math.floor(raw[key])), key),
    )
    while remaining:
        eligible = [key for key in order if out[key] < upper[key]]
        if not eligible:
            break
        for key in eligible:
            if not remaining:
                break
            out[key] += 1
            remaining -= 1
    return out


def _continuous_capped_allocation(
    dispersion: Mapping[str, float],
    base: Mapping[str, int],
    cap: Mapping[str, int],
    target: int,
) -> Tuple[Dict[str, float], Optional[float]]:
    if target <= sum(base.values()):
        return {key: float(value) for key, value in base.items()}, None

    positive = [key for key, value in dispersion.items() if value > 0.0]
    if not positive:
        raw = {key: float(value) for key, value in base.items()}
        remaining = target - sum(base.values())
        for key in sorted(raw):
            add = min(remaining, cap[key] - base[key])
            raw[key] += add
            remaining -= add
        return raw, None

    def total_for(dual_lambda: float) -> float:
        return sum(
            min(
                max(math.sqrt(dispersion[key] / dual_lambda), base[key]),
                cap[key],
            )
            if dispersion[key] > 0.0
            else float(base[key])
            for key in dispersion
        )

    lo, hi = 1e-18, 1.0
    while total_for(hi) > target:
        hi *= 2.0
    for _ in range(160):
        mid = (lo + hi) / 2.0
        if total_for(mid) > target:
            lo = mid
        else:
            hi = mid
    dual_lambda = hi
    raw = {
        key: (
            min(
                max(math.sqrt(dispersion[key] / dual_lambda), base[key]),
                cap[key],
            )
            if dispersion[key] > 0.0
            else float(base[key])
        )
        for key in dispersion
    }

    # If positive-dispersion nodes saturate, zero-dispersion demand is still
    # valid demand and must be filled to preserve the requested queue budget.
    remaining = target - sum(raw.values())
    if remaining > 1e-9:
        for key in sorted(raw):
            room = cap[key] - raw[key]
            add = min(remaining, room)
            raw[key] += add
            remaining -= add
            if remaining <= 1e-9:
                break
    return raw, dual_lambda


def allocate_branch_factors(
    nodes: Sequence[Mapping[str, Any]],
    *,
    total_budget: int,
    n_min: int = 1,
    weight_key: Optional[str] = None,
    strict: bool = True,
    rounding_strategy: str = "largest_remainder",
    rounding_seed: int = 0,
) -> AllocationSummary:
    """Solve VDRA pruning plus capped residual-budget allocation.

    ``predicted_k`` defines useful demand; ``dispersion_C`` determines which
    eligible nodes receive saved budget. Legacy field names are read only for
    checkpoint compatibility and are never emitted.
    """

    requested = max(int(total_budget), 0)
    floor = max(int(n_min), 0)
    dispersion: Dict[str, float] = {}
    base: Dict[str, int] = {}
    cap: Dict[str, int] = {}
    saved: Dict[str, int] = {}
    unmet: Dict[str, int] = {}

    for idx, node in enumerate(nodes):
        key = node_id(node, idx)
        raw_c = node.get("vdra_dispersion_C", node.get("gear_reward_variance", 0.0))
        c_value = float(raw_c or 0.0)
        if not math.isfinite(c_value) or c_value < 0.0:
            if strict:
                raise ValueError(f"Invalid VDRA dispersion bound for {key}: {raw_c!r}")
            c_value = 0.0
        if weight_key and node.get(weight_key) is not None:
            weight = max(float(node[weight_key]), 0.0)
            c_value = weight * weight
        default_k = max(
            int(node.get("vdra_default_k", node.get("default_k", requested)) or 0),
            floor,
        )
        predicted_k = int(
            node.get(
                "vdra_predicted_k",
                node.get("gear_predicted_k", node.get("predicted_k", default_k)),
            )
            or 0
        )
        cap_k = max(floor, predicted_k)
        base_k = min(default_k, cap_k)
        dispersion[key] = c_value
        cap[key] = cap_k
        base[key] = base_k
        saved[key] = max(default_k - base_k, 0)
        unmet[key] = max(cap_k - base_k, 0)

    target = min(requested, sum(cap.values()))
    minimum = sum(base.values())
    if target < minimum:
        raise ValueError(
            f"Queue budget {requested} is below mandatory base allocation {minimum}"
        )
    raw, dual_lambda = _continuous_capped_allocation(dispersion, base, cap, target)
    allocations = round_bounded(
        raw,
        lower=base,
        upper=cap,
        target=target,
        strategy=rounding_strategy,
        seed=rounding_seed,
    )
    weights = {key: math.sqrt(value) for key, value in dispersion.items()}
    additional = {key: allocations[key] - base[key] for key in allocations}
    for idx, node in enumerate(nodes):
        if isinstance(node, MutableMapping):
            key = node_id(node, idx)
            write_node_accounting(
                node,
                default_k=int(node.get("vdra_default_k", node.get("default_k", requested)) or 0),
                predicted_k=int(
                    node.get(
                        "vdra_predicted_k",
                        node.get("gear_predicted_k", node.get("predicted_k", base[key])),
                    )
                    or 0
                ),
                dispersion_C=dispersion[key],
                allocated_k=allocations[key],
                k_min=floor,
                allocation_weight=weights[key],
            )
    allocated = sum(allocations.values())
    return AllocationSummary(
        allocations=allocations,
        weights=weights,
        raw_allocations=raw,
        requested_budget=requested,
        allocated_budget=allocated,
        underallocated_budget=max(requested - allocated, 0),
        base_allocations=base,
        cap_allocations=cap,
        saved_allocations=saved,
        unmet_demands=unmet,
        additional_allocations=additional,
        dual_lambda=dual_lambda,
    )
