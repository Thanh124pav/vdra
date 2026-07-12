"""Canonical VDRA logging schema shared by treetune and verl."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


ALLOCATED_K_ALIASES = (
    "vdra_allocated_k",
    "gear_branch_allocation",
    "gear_allocated_branch_factor",
    "allocated_k",
)


def node_id(node: Mapping[str, Any], fallback: int = 0) -> str:
    return str(
        node.get("vdra_node_id")
        or node.get("gear_segment_id")
        or node.get("segment_id")
        or node.get("id")
        or f"node_{fallback}"
    )


def node_allocated_k(node: Mapping[str, Any]) -> Optional[int]:
    for key in ALLOCATED_K_ALIASES:
        if node.get(key) is not None:
            return int(node[key])
    return None


def write_node_accounting(
    node: MutableMapping[str, Any],
    *,
    default_k: Optional[int] = None,
    predicted_k: Optional[int] = None,
    dispersion_C: Optional[float] = None,
    allocated_k: Optional[int] = None,
    k_min: int = 1,
    allocation_weight: Optional[float] = None,
) -> MutableMapping[str, Any]:
    """Write the complete canonical pruning/allocation trace onto ``node``.

    The canonical VDRA fields are always written. Temporary legacy aliases are
    also written so older rollout code can keep consuming the allocation while
    downstream logs and tests move to ``vdra_*`` names.
    """

    floor = max(int(k_min), 0)
    default = int(
        default_k
        if default_k is not None
        else node.get("vdra_default_k", node.get("default_k", node.get("gear_default_branch_factor", 0)))
        or 0
    )
    predicted = int(
        predicted_k
        if predicted_k is not None
        else node.get("vdra_predicted_k", node.get("gear_predicted_k", node.get("predicted_k", default)))
        or 0
    )
    cap = max(floor, predicted)
    base = min(default, cap)
    saved = max(default - base, 0)
    unmet = max(cap - base, 0)
    existing_allocated = node_allocated_k(node)
    allocated = int(allocated_k if allocated_k is not None else existing_allocated if existing_allocated is not None else base)
    allocated = min(max(allocated, base), cap)
    additional = max(allocated - base, 0)
    c_value = (
        float(dispersion_C)
        if dispersion_C is not None
        else float(node.get("vdra_dispersion_C", node.get("gear_reward_variance", 0.0)) or 0.0)
    )
    if not math.isfinite(c_value) or c_value < 0.0:
        c_value = 0.0
    weight = float(allocation_weight) if allocation_weight is not None else float(math.sqrt(c_value))

    node["vdra_default_k"] = default
    node["vdra_predicted_k"] = predicted
    node["vdra_cap_k"] = cap
    node["vdra_base_k"] = base
    node["vdra_saved_k"] = saved
    node["vdra_unmet_demand"] = unmet
    node["vdra_dispersion_C"] = c_value
    node["vdra_allocation_weight"] = weight
    node["vdra_additional_k"] = additional
    node["vdra_allocated_k"] = allocated
    node["vdra_reserve_contribution"] = saved
    node["vdra_reserve_received"] = additional

    # Temporary read-compatibility aliases.
    node["gear_predicted_k"] = predicted
    node["gear_branch_allocation"] = allocated
    node["gear_allocated_branch_factor"] = allocated
    node["gear_reward_variance"] = c_value
    node["gear_budget_weight"] = weight
    return node


def validate_node_accounting(node: Mapping[str, Any], *, k_min: int = 1) -> None:
    default = int(node["vdra_default_k"])
    predicted = int(node["vdra_predicted_k"])
    cap = int(node["vdra_cap_k"])
    base = int(node["vdra_base_k"])
    saved = int(node["vdra_saved_k"])
    unmet = int(node["vdra_unmet_demand"])
    additional = int(node["vdra_additional_k"])
    allocated = int(node["vdra_allocated_k"])
    reserve_contribution = int(node["vdra_reserve_contribution"])
    reserve_received = int(node["vdra_reserve_received"])
    floor = max(int(k_min), 0)

    assert cap == max(floor, predicted)
    assert base == min(default, cap)
    assert saved == default - base
    assert unmet == max(cap - base, 0)
    assert additional == allocated - base
    assert reserve_contribution == saved
    assert reserve_received == additional
    assert base <= allocated <= cap


def iter_tree_nodes(tree: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.get("children") or []))


def summarize_vdra_tree(tree: Mapping[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {
        "vdra_total_saved_branches": 0.0,
        "vdra_total_redistributed_branches": 0.0,
        "vdra_reserve_contributed": 0.0,
        "vdra_reserve_consumed": 0.0,
        "vdra_main_expansion_requested_branches": 0.0,
        "vdra_main_expansion_allocated_branches": 0.0,
        "vdra_main_expansion_built_branches": 0.0,
        "vdra_pilot_children_generated": 0.0,
        "vdra_pilot_children_reused": 0.0,
        "vdra_pilot_children_discarded": 0.0,
        "vdra_additional_children_generated": 0.0,
        "vdra_pilot_generated_tokens": 0.0,
        "vdra_main_expansion_generated_tokens": 0.0,
        "vdra_scoring_request_count": 0.0,
        "vdra_scoring_prefill_tokens": 0.0,
        "vdra_scoring_continuation_tokens": 0.0,
    }
    for node in iter_tree_nodes(tree):
        allocated = node_allocated_k(node)
        if allocated is not None:
            totals["vdra_main_expansion_allocated_branches"] += allocated
            totals["vdra_main_expansion_built_branches"] += len(node.get("children") or [])
        totals["vdra_main_expansion_requested_branches"] += float(node.get("vdra_default_k", 0) or 0)
        totals["vdra_total_saved_branches"] += float(node.get("vdra_saved_k", 0) or 0)
        totals["vdra_total_redistributed_branches"] += float(node.get("vdra_additional_k", 0) or 0)
        totals["vdra_reserve_contributed"] += float(node.get("vdra_reserve_contribution", 0) or 0)
        totals["vdra_reserve_consumed"] += float(node.get("vdra_reserve_received", 0) or 0)
        totals["vdra_pilot_children_generated"] += float(node.get("vdra_pilot_children_generated", 0) or 0)
        totals["vdra_pilot_children_reused"] += float(node.get("vdra_pilot_children_reused", 0) or 0)
        totals["vdra_pilot_children_discarded"] += float(node.get("vdra_pilot_children_discarded", 0) or 0)
        totals["vdra_additional_children_generated"] += float(node.get("vdra_additional_children_generated", 0) or 0)
        totals["vdra_pilot_generated_tokens"] += float(node.get("vdra_pilot_generated_tokens", 0) or 0)
        totals["vdra_main_expansion_generated_tokens"] += float(node.get("vdra_main_expansion_generated_tokens", 0) or 0)
        totals["vdra_scoring_request_count"] += float(node.get("vdra_likelihood_scoring_requests", 0) or 0)
        totals["vdra_scoring_prefill_tokens"] += float(node.get("vdra_likelihood_scored_prompt_tokens", 0) or 0)
        totals["vdra_scoring_continuation_tokens"] += float(node.get("vdra_likelihood_scored_continuation_tokens", 0) or 0)

    totals["vdra_total_unallocated_reserve"] = max(
        totals["vdra_reserve_contributed"] - totals["vdra_reserve_consumed"], 0.0
    )
    totals["vdra_reserve_remaining"] = totals["vdra_total_unallocated_reserve"]
    totals["vdra_total_generated_tokens"] = (
        totals["vdra_pilot_generated_tokens"] + totals["vdra_main_expansion_generated_tokens"]
    )
    totals["vdra_total_scored_tokens"] = (
        totals["vdra_scoring_prefill_tokens"] + totals["vdra_scoring_continuation_tokens"]
    )
    totals["vdra_generation_forward_calls"] = totals["vdra_total_generated_tokens"]
    totals["vdra_scoring_forward_calls"] = totals["vdra_scoring_request_count"]
    totals["vdra_total_model_forward_calls"] = (
        totals["vdra_generation_forward_calls"] + totals["vdra_scoring_forward_calls"]
    )
    # Compute proxy: decode token count plus scored prefill/continuation tokens.
    totals["vdra_total_forward_pass_cost"] = totals["vdra_total_generated_tokens"] + totals["vdra_total_scored_tokens"]
    generated = totals["vdra_pilot_children_generated"]
    totals["vdra_pilot_reuse_rate"] = totals["vdra_pilot_children_reused"] / generated if generated else 0.0
    return totals
