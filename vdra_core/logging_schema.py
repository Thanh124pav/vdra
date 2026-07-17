"""Canonical VDRA logging schema shared by treetune and verl."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


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
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
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
    existing_allocated = node_allocated_k(node)
    lower = max(int(lower_bound if lower_bound is not None else floor), floor)
    allocated_seed = int(
        allocated_k
        if allocated_k is not None
        else existing_allocated
        if existing_allocated is not None
        else lower
    )
    cap = max(
        int(upper_bound if upper_bound is not None else max(floor, predicted, allocated_seed)),
        lower,
    )
    base = lower
    allocated = min(max(allocated_seed, lower), cap)
    saved = max(default - allocated, 0)
    unmet = max(allocated - default, 0)
    additional = max(allocated - default, 0)
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
    node["vdra_lower_bound_k"] = lower
    node["vdra_upper_bound_k"] = cap
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

    assert base >= floor
    assert cap >= base
    assert saved == max(default - allocated, 0)
    assert unmet == max(allocated - default, 0)
    assert additional == max(allocated - default, 0)
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
        "vdra_pilot_children_shortcut": 0.0,
        "vdra_shortcut_overage": 0.0,
        "vdra_pilot_children_discarded": 0.0,
        "vdra_additional_children_generated": 0.0,
        "vdra_pilot_generated_tokens": 0.0,
        "vdra_pilot_support_children_generated": 0.0,
        "vdra_pilot_support_generated_tokens": 0.0,
        "vdra_main_expansion_generated_tokens": 0.0,
        "vdra_proxy_rollout_tokens": 0.0,
        "vdra_generation_request_count": 0.0,
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
        totals["vdra_pilot_children_shortcut"] += float(node.get("vdra_pilot_children_shortcut", 0) or 0)
        totals["vdra_shortcut_overage"] += float(node.get("vdra_shortcut_overage", 0) or 0)
        totals["vdra_pilot_children_discarded"] += float(node.get("vdra_pilot_children_discarded", 0) or 0)
        totals["vdra_additional_children_generated"] += float(node.get("vdra_additional_children_generated", 0) or 0)
        totals["vdra_pilot_generated_tokens"] += float(node.get("vdra_pilot_generated_tokens", 0) or 0)
        totals["vdra_pilot_support_children_generated"] += float(node.get("vdra_pilot_support_children_generated", 0) or 0)
        totals["vdra_pilot_support_generated_tokens"] += float(node.get("vdra_pilot_support_generated_tokens", 0) or 0)
        totals["vdra_main_expansion_generated_tokens"] += float(node.get("vdra_main_expansion_generated_tokens", 0) or 0)
        totals["vdra_proxy_rollout_tokens"] += float(node.get("vdra_proxy_rollout_tokens", 0) or 0)
        totals["vdra_generation_request_count"] += float(node.get("vdra_generation_request_count", 0) or 0)
        totals["vdra_scoring_request_count"] += float(node.get("vdra_likelihood_scoring_requests", 0) or 0)
        totals["vdra_scoring_prefill_tokens"] += float(node.get("vdra_likelihood_scored_prompt_tokens", 0) or 0)
        totals["vdra_scoring_continuation_tokens"] += float(node.get("vdra_likelihood_scored_continuation_tokens", 0) or 0)

    totals["vdra_total_unallocated_reserve"] = max(
        totals["vdra_reserve_contributed"] - totals["vdra_reserve_consumed"], 0.0
    )
    totals["vdra_reserve_remaining"] = totals["vdra_total_unallocated_reserve"]
    totals["vdra_total_generated_tokens"] = (
        totals["vdra_pilot_generated_tokens"]
        + totals["vdra_pilot_support_generated_tokens"]
        + totals["vdra_main_expansion_generated_tokens"]
    )
    totals["vdra_total_scored_tokens"] = (
        totals["vdra_scoring_prefill_tokens"] + totals["vdra_scoring_continuation_tokens"]
    )
    totals["vdra_generation_decode_tokens"] = totals["vdra_total_generated_tokens"]
    totals["vdra_token_equivalent_compute_proxy"] = (
        totals["vdra_generation_decode_tokens"]
        + totals["vdra_total_scored_tokens"]
        + totals["vdra_proxy_rollout_tokens"]
    )
    # Back-compat alias with explicit units; not a forward-call count.
    totals["vdra_total_forward_pass_cost"] = totals["vdra_token_equivalent_compute_proxy"]
    generated = totals["vdra_pilot_children_generated"]
    totals["vdra_pilot_reuse_rate"] = totals["vdra_pilot_children_reused"] / generated if generated else 0.0
    return totals


NODE_RECORD_FIELDS = (
    "run_id",
    "tree_id",
    "node_id",
    "parent_id",
    "depth",
    "default_k",
    "predicted_k",
    "cap_k",
    "base_k",
    "saved_k",
    "unmet_demand",
    "dispersion_C",
    "allocation_weight",
    "additional_k",
    "allocated_k",
    "reserve_contribution",
    "reserve_received",
    "pilot_children_generated",
    "pilot_children_reused",
    "pilot_children_shortcut",
    "shortcut_overage",
    "pilot_children_discarded",
    "additional_children_generated",
    "pilot_generated_tokens",
    "pilot_support_generated_tokens",
    "main_expansion_generated_tokens",
    "proxy_rollout_tokens",
    "scored_tokens",
    "queue_id",
    "queue_wait_seconds",
    "flush_reason",
)


BUDGET_CLAIMS = {
    "fixed_main": (
        "fixed main expansion budget; pilot and scoring overhead reported separately"
    ),
    "fixed_total_generated": (
        "legacy alias for uniform_full_tree_token_cap; pilot + support + main "
        "expansion share one maximum-style full-tree token cap; likelihood scoring reported separately"
    ),
    "uniform_full_tree_token_cap": (
        "pilot + support + main expansion share one maximum-style full-tree token cap; "
        "likelihood scoring reported separately"
    ),
}

COMPUTE_PROXY_DEFINITION = (
    "pilot decode tokens + pilot-support decode tokens + main-expansion decode tokens "
    "+ scored prompt tokens + scored continuation tokens"
)


def budget_claim_for_mode(budget_mode: Optional[str]) -> str:
    """Return the manifest budget claim string for a VDRA budget mode."""

    mode = str(budget_mode or "fixed_main")
    if mode not in BUDGET_CLAIMS:
        raise ValueError(f"Unknown VDRA budget mode: {mode!r}")
    return BUDGET_CLAIMS[mode]


def node_record(
    node: Mapping[str, Any],
    *,
    run_id: Optional[str] = None,
    tree_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return one canonical JSONL node record for allocation-eligible parents."""

    return {
        "run_id": run_id,
        "tree_id": tree_id,
        "node_id": node_id(node),
        "parent_id": node.get("gear_parent_segment_id") or node.get("parent_id"),
        "depth": int(node.get("depth", node.get("gear_depth", 0)) or 0),
        "default_k": int(node.get("vdra_default_k", 0) or 0),
        "predicted_k": int(node.get("vdra_predicted_k", 0) or 0),
        "cap_k": int(node.get("vdra_cap_k", 0) or 0),
        "base_k": int(node.get("vdra_base_k", 0) or 0),
        "saved_k": int(node.get("vdra_saved_k", 0) or 0),
        "unmet_demand": int(node.get("vdra_unmet_demand", 0) or 0),
        "dispersion_C": float(node.get("vdra_dispersion_C", 0.0) or 0.0),
        "allocation_weight": float(node.get("vdra_allocation_weight", node.get("gear_budget_weight", 0.0)) or 0.0),
        "additional_k": int(node.get("vdra_additional_k", 0) or 0),
        "allocated_k": int(node.get("vdra_allocated_k", node_allocated_k(node) or 0) or 0),
        "reserve_contribution": int(node.get("vdra_reserve_contribution", 0) or 0),
        "reserve_received": int(node.get("vdra_reserve_received", 0) or 0),
        "pilot_children_generated": int(node.get("vdra_pilot_children_generated", 0) or 0),
        "pilot_children_reused": int(node.get("vdra_pilot_children_reused", 0) or 0),
        "pilot_children_shortcut": int(node.get("vdra_pilot_children_shortcut", 0) or 0),
        "shortcut_overage": int(node.get("vdra_shortcut_overage", 0) or 0),
        "pilot_children_discarded": int(node.get("vdra_pilot_children_discarded", 0) or 0),
        "additional_children_generated": int(node.get("vdra_additional_children_generated", 0) or 0),
        "pilot_generated_tokens": int(node.get("vdra_pilot_generated_tokens", 0) or 0),
        "pilot_support_generated_tokens": int(node.get("vdra_pilot_support_generated_tokens", 0) or 0),
        "main_expansion_generated_tokens": int(node.get("vdra_main_expansion_generated_tokens", 0) or 0),
        "proxy_rollout_tokens": int(node.get("vdra_proxy_rollout_tokens", 0) or 0),
        "scored_tokens": int(node.get("vdra_total_scored_tokens", 0) or 0),
        "queue_id": node.get("vdra_queue_id"),
        "queue_wait_seconds": float(node.get("vdra_queue_wait_seconds", 0.0) or 0.0),
        "flush_reason": node.get("vdra_flush_reason"),
    }


def queue_flush_record(
    flush_record: Mapping[str, Any],
    *,
    run_id: Optional[str] = None,
    tree_id: Optional[str] = None,
) -> Dict[str, Any]:
    out = {"run_id": run_id, "tree_id": tree_id}
    out.update(dict(flush_record))
    return out


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dict(record), sort_keys=True, default=str) + "\n")


def allocation_node_records(
    tree: Mapping[str, Any],
    *,
    run_id: Optional[str] = None,
    tree_id: Optional[str] = None,
) -> Sequence[Dict[str, Any]]:
    return [
        node_record(node, run_id=run_id, tree_id=tree_id)
        for node in iter_tree_nodes(tree)
        if "vdra_default_k" in node
    ]


def build_compute_summary(tree: Mapping[str, Any]) -> Dict[str, Any]:
    raw = summarize_vdra_tree(tree)
    return {
        key.removeprefix("vdra_"): value
        for key, value in raw.items()
    }


# --- Run manifest evidence (PLAN.md P1.3) ------------------------------------ #

RUN_MANIFEST_REQUIRED_FIELDS: Sequence[str] = (
    "policy_snapshot_id",
    "rollout_server_weight_version",
    "scorer_server_weight_version",
    "weight_version_verified",
    "allocation_scope",
    "flush_depths",
    "pilot_execution_mode",
    "weighted_reuse_fallback_count",
    "token_cap_hit_count",
    "successful_actor_updates",
    "rollout_iterations",
)


def compute_run_valid_for_main_results(evidence: Mapping[str, Any]) -> bool:
    """Return ``run_valid_for_main_results`` from runtime evidence.

    Per PLAN.md P1.3 a run is a valid main-paper result only when every
    named invariant is confirmed by runtime evidence. Missing evidence is
    treated as failure — the trainer must supply positive proof for each
    condition, not rely on defaults.
    """

    allocation_proxy = evidence.get("allocation_proxy")
    if allocation_proxy == "oracle":
        return False
    if not bool(evidence.get("weight_version_verified", False)):
        return False
    if bool(evidence.get("unexpected_fallback", False)):
        return False
    if bool(evidence.get("unexpected_token_cap_hit", False)):
        return False
    if not bool(evidence.get("all_node_accounting_invariants_passed", False)):
        return False
    if not bool(evidence.get("all_snapshot_invariants_passed", False)):
        return False
    if not bool(evidence.get("context_contract_passed", False)):
        return False
    return True


def build_run_manifest(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the canonical run manifest built from runtime evidence.

    ``evidence`` MUST supply at minimum the fields in
    :data:`RUN_MANIFEST_REQUIRED_FIELDS`; anything else is passed through as-is
    for callers that want to persist extra descriptive metadata.
    """

    manifest: Dict[str, Any] = dict(evidence)
    manifest["run_valid_for_main_results"] = compute_run_valid_for_main_results(evidence)
    for field in RUN_MANIFEST_REQUIRED_FIELDS:
        manifest.setdefault(field, None)
    return manifest


def persist_vdra_artifacts(
    output_dir: str | Path,
    tree: Mapping[str, Any],
    *,
    run_id: Optional[str] = None,
    tree_id: Optional[str] = None,
    queue_flushes: Optional[Iterable[Mapping[str, Any]]] = None,
    run_manifest: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist the canonical VDRA runtime records for one tree/run.

    ``run_manifest`` is written verbatim under ``run_manifest.json``. Callers
    that want the aggregated PLAN.md P1.3 flags should compute them with
    :func:`build_run_manifest` first — this helper deliberately never overwrites
    a supplied manifest with weaker per-tree defaults.
    """

    out = Path(output_dir)
    append_jsonl(out / "nodes.jsonl", allocation_node_records(tree, run_id=run_id, tree_id=tree_id))
    append_jsonl(
        out / "queue_flushes.jsonl",
        [queue_flush_record(r, run_id=run_id, tree_id=tree_id) for r in (queue_flushes or [])],
    )
    write_json(out / "compute_summary.json", build_compute_summary(tree))
    manifest_path = out / "run_manifest.json"
    # P1.3: never clobber a stronger run-level manifest with a weaker per-tree
    # default. If the existing file has run_valid_for_main_results=False, keep
    # it unless the new payload also says False (a run is invalid the moment
    # any per-tree evidence fails).
    if manifest_path.exists() and run_manifest is not None:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if isinstance(existing, dict):
            new_manifest = dict(run_manifest)
            existing_valid = existing.get("run_valid_for_main_results")
            new_valid = new_manifest.get("run_valid_for_main_results")
            # False sticks: once any tree taints the run, the manifest stays
            # invalid until the trainer explicitly rewrites it.
            if existing_valid is False:
                new_manifest["run_valid_for_main_results"] = False
            elif new_valid is False:
                pass  # honor the new False
            elif existing_valid is True and new_valid is None:
                new_manifest["run_valid_for_main_results"] = True
            write_json(manifest_path, new_manifest)
            return
    write_json(manifest_path, dict(run_manifest or {}))
