"""Canonical VDRA bounds and exact bounded integer allocation."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from .rounding import round_bounded
from .logging_schema import node_id, write_node_accounting

PairKey = Tuple[int, int]


@dataclass(frozen=True)
class AllocationSummary:
    allocations: Dict[str, int]
    dispersion: Dict[str, float]
    default_allocations: Dict[str, int]
    predicted_allocations: Dict[str, int]
    lower_bounds: Dict[str, int]
    upper_bounds: Dict[str, int]
    pruned_allocations: Dict[str, int]
    expanded_allocations: Dict[str, int]
    transferred_budget: int
    requested_budget: int
    allocated_budget: int
    objective_before: float
    objective_after: float
    solver_time_ms: float
    solver_name: str
    feasibility_repair_count: int = 0
    weights: Dict[str, float] = field(default_factory=dict)
    # Compatibility aliases for older callers while the migration finishes.
    raw_allocations: Dict[str, float] = field(default_factory=dict)
    underallocated_budget: int = 0
    base_allocations: Dict[str, int] = field(default_factory=dict)
    cap_allocations: Dict[str, int] = field(default_factory=dict)
    saved_allocations: Dict[str, int] = field(default_factory=dict)
    unmet_demands: Dict[str, int] = field(default_factory=dict)
    additional_allocations: Dict[str, int] = field(default_factory=dict)
    dual_lambda: Optional[float] = None


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
            min(max(math.sqrt(dispersion[key] / dual_lambda), base[key]), cap[key])
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
            min(max(math.sqrt(dispersion[key] / dual_lambda), base[key]), cap[key])
            if dispersion[key] > 0.0
            else float(base[key])
        )
        for key in dispersion
    }
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


def _objective(dispersion: Mapping[str, float], allocations: Mapping[str, int]) -> float:
    total = 0.0
    for key, value in dispersion.items():
        k = max(int(allocations[key]), 1)
        total += float(value) / float(k)
    return total


def _resolve_int(node: Mapping[str, Any], keys: Sequence[str], default: int) -> int:
    for key in keys:
        if node.get(key) is not None:
            return int(node[key] or 0)
    return int(default)


def _resolve_dispersion(node: Mapping[str, Any], *, weight_key: Optional[str], strict: bool) -> float:
    raw_c = node.get("vdra_dispersion_C", node.get("gear_reward_variance", 0.0))
    c_value = float(raw_c or 0.0)
    if weight_key and node.get(weight_key) is not None:
        weight = max(float(node[weight_key]), 0.0)
        c_value = weight * weight
    if not math.isfinite(c_value) or c_value < 0.0:
        if strict:
            raise ValueError(f"Invalid VDRA dispersion bound: {raw_c!r}")
        c_value = 0.0
    return c_value


def _marginal_gain(c_value: float, k: int) -> float:
    return float(c_value) / float(k * (k + 1))


def allocate_branch_factors_integer(
    nodes: Sequence[Mapping[str, Any]],
    *,
    total_budget: int,
    n_min: int = 1,
    weight_key: Optional[str] = None,
    strict: bool = True,
    max_k_per_node: int = 12,
    predicted_k_cap_mode: str = "below_default_only",
    infeasible_upper_policy: str = "expand_nonredundant_caps",
    max_repair_k_per_node: Optional[int] = None,
) -> AllocationSummary:
    """Solve the unified bounded integer allocation exactly by marginal gains."""

    if predicted_k_cap_mode not in {
        "below_default_only",
        "predicted_k_for_all_nodes",
        "configured_max_for_all_nodes",
    }:
        raise ValueError(f"Unknown predicted_k_cap_mode: {predicted_k_cap_mode!r}")
    if infeasible_upper_policy not in {"expand_nonredundant_caps", "error"}:
        raise ValueError(f"Unknown infeasible_upper_policy: {infeasible_upper_policy!r}")
    if not nodes:
        raise ValueError("allocate_branch_factors received an empty node list")

    start = time.perf_counter()
    requested = int(total_budget)
    floor = max(int(n_min), 1)
    max_k = max(int(max_k_per_node), floor)
    repair_cap = max(int(max_repair_k_per_node if max_repair_k_per_node is not None else max_k), max_k)

    dispersion: Dict[str, float] = {}
    default: Dict[str, int] = {}
    predicted: Dict[str, int] = {}
    lower: Dict[str, int] = {}
    upper: Dict[str, int] = {}
    hard_redundancy_cap: Dict[str, bool] = {}

    for idx, node in enumerate(nodes):
        key = node_id(node, idx)
        if key in dispersion:
            raise ValueError(f"Duplicate allocation node id: {key!r}")
        c_value = _resolve_dispersion(node, weight_key=weight_key, strict=strict)
        default_k = max(
            _resolve_int(node, ("vdra_default_k", "default_k", "gear_default_branch_factor"), requested),
            floor,
        )
        predicted_k = _resolve_int(
            node,
            ("vdra_predicted_k", "gear_predicted_k", "predicted_k"),
            default_k,
        )
        predicted_k = max(predicted_k, floor)
        if predicted_k_cap_mode == "below_default_only" and predicted_k < default_k:
            upper_k = max(floor, predicted_k)
            capped = True
        elif predicted_k_cap_mode == "predicted_k_for_all_nodes":
            upper_k = max(floor, predicted_k)
            capped = predicted_k < default_k
        else:
            upper_k = max_k
            capped = False
        dispersion[key] = c_value
        default[key] = default_k
        predicted[key] = predicted_k
        lower[key] = floor
        upper[key] = max(upper_k, floor)
        hard_redundancy_cap[key] = capped

    lower_sum = sum(lower.values())
    upper_sum = sum(upper.values())
    if requested < lower_sum:
        raise ValueError(
            f"Queue budget {requested} is below lower-bound allocation {lower_sum}"
        )
    repair_count = 0
    if requested > upper_sum and infeasible_upper_policy == "expand_nonredundant_caps":
        for key in sorted(upper):
            if hard_redundancy_cap[key]:
                continue
            if upper[key] < repair_cap:
                upper[key] = repair_cap
                repair_count += 1
        upper_sum = sum(upper.values())
    if requested > upper_sum:
        raise ValueError(
            f"Queue budget {requested} exceeds upper-bound allocation {upper_sum}"
        )

    allocation = dict(lower)
    remaining = requested - lower_sum
    heap = []
    for key in sorted(allocation):
        if allocation[key] < upper[key]:
            heapq.heappush(heap, (-_marginal_gain(dispersion[key], allocation[key]), key))

    for _ in range(remaining):
        if not heap:
            raise RuntimeError("No feasible allocation capacity for remaining budget")
        _neg_gain, key = heapq.heappop(heap)
        allocation[key] += 1
        if allocation[key] < upper[key]:
            heapq.heappush(heap, (-_marginal_gain(dispersion[key], allocation[key]), key))

    allocated = sum(allocation.values())
    if allocated != requested:
        raise RuntimeError(f"Integer allocation produced {allocated}, expected {requested}")
    for key, value in allocation.items():
        if value < lower[key] or value > upper[key]:
            raise RuntimeError(f"Allocation for {key!r} violates bounds: {value}")

    solver_elapsed_ms = (time.perf_counter() - start) * 1000.0

    reference = {
        key: min(max(default[key], lower[key]), upper[key])
        for key in allocation
    }
    if sum(reference.values()) == requested:
        objective_before = _objective(dispersion, reference)
    else:
        objective_before = _objective(dispersion, default)
    objective_after = _objective(dispersion, allocation)
    pruned = {key: max(default[key] - allocation[key], 0) for key in allocation}
    expanded = {key: max(allocation[key] - default[key], 0) for key in allocation}
    transferred = min(sum(pruned.values()), sum(expanded.values()))
    weights = {key: math.sqrt(value) for key, value in dispersion.items()}
    for idx, node in enumerate(nodes):
        if isinstance(node, MutableMapping):
            key = node_id(node, idx)
            write_node_accounting(
                node,
                default_k=default[key],
                predicted_k=predicted[key],
                dispersion_C=dispersion[key],
                allocated_k=allocation[key],
                k_min=floor,
                lower_bound=lower[key],
                upper_bound=upper[key],
                allocation_weight=weights[key],
            )

    return AllocationSummary(
        allocations=allocation,
        dispersion=dispersion,
        default_allocations=default,
        predicted_allocations=predicted,
        lower_bounds=lower,
        upper_bounds=upper,
        pruned_allocations=pruned,
        expanded_allocations=expanded,
        transferred_budget=transferred,
        requested_budget=requested,
        allocated_budget=allocated,
        objective_before=objective_before,
        objective_after=objective_after,
        solver_time_ms=solver_elapsed_ms,
        solver_name="bounded_marginal_integer",
        feasibility_repair_count=repair_count,
        weights=weights,
        raw_allocations={key: float(value) for key, value in allocation.items()},
        underallocated_budget=0,
        base_allocations=lower,
        cap_allocations=upper,
        saved_allocations=pruned,
        unmet_demands=expanded,
        additional_allocations=expanded,
        dual_lambda=None,
    )


def allocate_branch_factors(
    nodes: Sequence[Mapping[str, Any]],
    *,
    total_budget: int,
    n_min: int = 1,
    weight_key: Optional[str] = None,
    strict: bool = True,
    rounding_strategy: str = "integer_marginal",
    rounding_seed: int = 0,
    max_k_per_node: int = 12,
    predicted_k_cap_mode: str = "below_default_only",
    infeasible_upper_policy: str = "expand_nonredundant_caps",
    max_repair_k_per_node: Optional[int] = None,
) -> AllocationSummary:
    """Solve unified VDRA pruning/expansion allocation.

    The default path is the exact bounded marginal integer solver. The old
    continuous-plus-rounding path is intentionally unavailable as a silent
    default; callers must use this function's integer semantics.
    """

    allowed_legacy_names = {"integer_marginal", "bounded_marginal_integer"}
    if rounding_strategy not in allowed_legacy_names:
        raise ValueError(
            f"Unknown rounding strategy for default VDRA solver: {rounding_strategy!r}. "
            "Use 'integer_marginal'/'bounded_marginal_integer'."
        )
    _ = rounding_seed
    return allocate_branch_factors_integer(
        nodes,
        total_budget=total_budget,
        n_min=n_min,
        weight_key=weight_key,
        strict=strict,
        max_k_per_node=max_k_per_node,
        predicted_k_cap_mode=predicted_k_cap_mode,
        infeasible_upper_policy=infeasible_upper_policy,
        max_repair_k_per_node=max_repair_k_per_node,
    )
