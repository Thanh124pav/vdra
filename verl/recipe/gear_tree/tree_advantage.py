"""Tree edge advantage glue for the GEAR/Tree recipe.

This module intentionally keeps the math in ``gear_core.tree_update_modes`` and
only owns the IO layer needed by verl: tree nodes -> edge rows -> token tensors.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Iterable

import torch

from recipe.gear_tree.gear_core.tree_update_modes import compute_tree_update_values


def _bok(value: float, bok: int = 4) -> float:
    return 1.0 - (1.0 - float(value)) ** bok


def _node_reward(node: dict[str, Any]) -> float:
    return float(node.get("reward", node.get("score", 0.0)) or 0.0)


def _node_reward_std(node: dict[str, Any]) -> float:
    return float(node.get("reward_std", 0.0) or 0.0)


def extract_edges_from_tree(
    tree: dict[str, Any],
    *,
    adv_method: str = "rloo",
    only_adv_greater_than_zero: bool = False,
    tree_update_mode: str = "spo",
    treepo_global_weight: float = 0.5,
    treerl_gamma: float = 0.9,
    emit_pruned_edges: bool = False,
    strict_fresh_iid: bool = False,
) -> list[dict[str, Any]]:
    """Extract treetune-compatible training edges from a generated tree.

    This ports ``TreeEpisodeUtils.extract_edges_from_tree`` while reading both
    ``reward`` and legacy ``score`` fields.

    Stage 1 zero-filter contract: advantages are computed for every realized
    child before filtering. When ``only_adv_greater_than_zero`` is true, the
    legacy name means "drop exact-zero advantages to save compute"; positive
    and negative advantages are retained. Administrative ``pruned=True``
    placeholder rows must NOT enter replay; ``emit_pruned_edges`` defaults to
    ``False`` and is only for diagnostic dumps.

    ``strict_fresh_iid`` enforces the fresh_iid invariants right at edge
    extraction:
      * every realized (non-pruned) child has ``sample_multiplicity == 1``;
      * for every parent, the count of realized children equals
        ``vdra_allocated_k``.
    """
    edges: list[dict[str, Any]] = []
    tree_copy = copy.deepcopy(tree)
    data_instance = tree_copy.get("_request_object", {})
    # P1.6: require a globally stable question id. `_treetune__idx` /
    # `uid` come from the dataset row; falling back to a per-batch index
    # would let the per-question replay cap combine different questions
    # that happen to share a batch-local index across steps.
    question_id = data_instance.get("_treetune__idx", data_instance.get("uid"))
    if question_id is None:
        raise ValueError(
            "Data instance has no stable question id (_treetune__idx / uid). "
            "Add a dataset UID (or a hash of the normalized problem) — a "
            "per-batch index is not acceptable (PLAN.md P1.6)."
        )
    root_reward = _node_reward(tree_copy)
    policy_snapshot_id = (
        tree_copy.get("policy_snapshot_id")
        or tree_copy.get("vdra_policy_snapshot_id")
        or data_instance.get("policy_snapshot_id")
        or data_instance.get("current_rollout_snapshot_id")
    )
    # PLAN.md P0.2/P0.H: every stochastic tree has one globally-unique
    # tree_instance_id stamped by the tree builder (make_tree_instance_id:
    # snapshot + rollout iteration + question + per-tree uuid/counter).
    # Prefer it; legacy tree_id fields keep old fixtures working. The
    # (snapshot, question) tuple must NEVER be used as a tree id in main runs
    # — two rollouts for the same prompt in the same iteration would collide.
    tree_instance_id = (
        tree_copy.get("tree_instance_id")
        or data_instance.get("tree_instance_id")
    )
    if strict_fresh_iid and not tree_instance_id:
        # PLAN.md P0.H / M3: strict main generation requires the explicit
        # tree_instance_id stamped by the tree builder (make_tree_instance_id:
        # snapshot + rollout iteration + question + per-tree uuid/counter).
        # A legacy tree_id alone is NOT a valid strict identity, and the
        # (snapshot, question) tuple must never be used as a tree id.
        raise ValueError(
            "Strict VDRA requires a unique tree_instance_id stamped by "
            "the tree builder (make_tree_instance_id); a legacy tree_id "
            "alone or the ambiguous snapshot:question fallback is not a "
            "valid strict identity (PLAN.md P0.H)."
        )
    tree_id = (
        tree_instance_id
        or tree_copy.get("tree_id")
        or data_instance.get("tree_id")
    )
    if tree_id is None:
        # Non-strict compatibility path only — strict already raised above.
        tree_id = f"{policy_snapshot_id}:{question_id}"

    # PLAN.md P0.N1: aggregate tree-level counts as we walk the tree so the
    # trainer can assert group integrity without a second pass.
    expanded_parent_group_ids: set[str] = set()
    trainable_child_count = 0
    queue_to_parent_group_counts: dict[Any, set[str]] = {}
    # Stage 1: pre-filter segment counts. Every realized non-pruned
    # segment increments ``tree_total_segment_count`` even if its advantage
    # is exactly zero and the row is dropped from replay to save compute.
    # Queue counts preserve the same pre-filter construction snapshot.
    tree_total_segment_count = 0
    queue_released_segment_count: dict[Any, int] = {}
    # PLAN.md P0.2: per-parent realized-vs-allocated snapshot. Fresh_iid
    # requires ``realized_child_count == allocated_k`` at construction time;
    # zero-advantage filtering may only shrink the retained set later.
    realized_by_parent: dict[str, dict[str, int]] = {}
    root_parent_group_id = f"tree:{tree_id}#root"

    def _parent_group_id(parent: dict[str, Any]) -> str:
        parent_seg = parent.get("gear_segment_id")
        if parent_seg is None or parent_seg == "root":
            return root_parent_group_id
        return f"tree:{tree_id}#pg:{parent_seg}"

    def _child_segment_id(node: dict[str, Any], parent_group_id: str, idx: int) -> str:
        seg = node.get("gear_segment_id")
        if seg:
            return str(seg)
        return f"{parent_group_id}/c{idx}"

    def visit(node: dict[str, Any], parent: dict[str, Any] | None = None) -> None:
        nonlocal trainable_child_count, tree_total_segment_count
        if parent is not None:
            parent_reward = _node_reward(parent)
            child_reward = _node_reward(node)
            update_values = compute_tree_update_values(
                child_reward=child_reward,
                parent_reward=parent_reward,
                root_reward=root_reward,
                parent_reward_std=_node_reward_std(parent),
                adv_method=adv_method,
                mode=tree_update_mode,
                treepo_global_weight=treepo_global_weight,
                treerl_gamma=treerl_gamma,
            )

            is_pruned = bool(node.get("pruned", node.get("is_pruned", False)))
            advantage = 0.0 if is_pruned else float(update_values["advantage"])
            value = float(update_values["value"])
            prover_advantage = _bok(child_reward) - _bok(parent_reward)
            pav_advantage = advantage + prover_advantage

            parent_group_id = _parent_group_id(parent)
            # child_index: prefer explicit stamp, else use position within
            # parent's realized children list. Position is deterministic in
            # DFS/BFS since siblings share one asyncio.gather.
            siblings = list(parent.get("children", []))
            try:
                idx = siblings.index(node)
            except ValueError:
                idx = int(node.get("child_index", 0) or 0)
            child_segment_id = _child_segment_id(node, parent_group_id, idx)
            parent_segment_id = parent.get("gear_segment_id") or "root"
            # PLAN.md P0.N1: allocated_k must equal the number of realized
            # trainable children for fresh_iid. The tree builder stamps
            # vdra_allocated_k on the parent; otherwise fall back to the
            # length of the child list (all realized siblings).
            allocated_k = int(
                parent.get("vdra_allocated_k", len(siblings)) or len(siblings)
            )
            # sample_multiplicity (P0.N2): must be separate from any
            # optimization coefficient. Under fresh_iid it is 1; under
            # weighted_reuse it is the cluster multiplicity.
            raw_multiplicity = node.get(
                "vdra_cluster_multiplicity",
                node.get("sample_multiplicity"),
            )
            if raw_multiplicity is None:
                sample_multiplicity = 1
            else:
                try:
                    sample_multiplicity = max(int(raw_multiplicity), 1)
                except (TypeError, ValueError):
                    sample_multiplicity = 1
            # queue_flush_id: stamped by the online-alloc path per queue
            # flush; defaults to 0 for DFS/batch paths where each parent is
            # its own "flush".
            queue_flush_id = node.get(
                "vdra_queue_flush_id",
                parent.get("vdra_queue_flush_id", 0),
            )

            edge = {
                "question_id": question_id,
                "policy_snapshot_id": policy_snapshot_id,
                "instance": data_instance,
                "query_text": parent.get("full_text", parent.get("text", "")),
                "response_text": node.get("text", ""),
                # Tokens for the training row: query = parent's accumulated
                # trajectory tokens (prompt + prior segments), response = this
                # segment's generated tokens. Both come straight from the
                # rollout so no re-tokenization mismatch can creep in.
                "query_token_ids": parent.get("full_token_ids"),
                "response_token_ids": node.get("response_token_ids", node.get("token_ids")),
                "actor_shifted_log_probs": node.get("actor_shifted_log_probs", node.get("old_log_probs")),
                "prover_advantage": prover_advantage,
                "advantage": advantage,
                "value": value,
                "leaf": bool(node.get("leaf", not node.get("children"))),
                "reward": child_reward,
                "pruned": is_pruned,
                # PLAN.md P0.N1/N2: canonical grouping metadata. These must
                # survive tree -> edge -> replay -> DataProto -> actor.
                # tree_id is kept for backward compat; on the strict path it
                # always equals tree_instance_id (stamped below, PLAN.md M3).
                "tree_id": str(tree_id),
                "parent_group_id": str(parent_group_id),
                "parent_segment_id": str(parent_segment_id),
                "child_segment_id": str(child_segment_id),
                "child_index": int(idx),
                "allocated_k": int(allocated_k),
                "sample_multiplicity": int(sample_multiplicity),
                "queue_flush_id": queue_flush_id,
                # P0.5 / P0.W3: representative-weight fields must survive
                # tree → edge → DataProto → actor for the weighted_reuse
                # ablation. edge_weight is a separate optimization-time
                # coefficient; sample_multiplicity is the sample count.
                "edge_weight": node.get(
                    "edge_weight", node.get("vdra_representative_weight")
                ),
                "vdra_cluster_id": node.get("vdra_cluster_id"),
                "vdra_cluster_multiplicity": node.get("vdra_cluster_multiplicity"),
                "vdra_original_pilot_indices": node.get("vdra_original_pilot_indices"),
                **update_values,
            }
            if tree_instance_id:
                # PLAN.md M3: the explicit instance identity survives the
                # tree -> edge boundary. Absent only on the legacy
                # non-strict path, which never had one to begin with.
                edge["tree_instance_id"] = str(tree_instance_id)
            edge["advantage"] = advantage
            # Stage 1: count every realized non-pruned segment BEFORE
            # zero-advantage filtering. ``tree_total_segment_count`` preserves
            # the construction count even when exact-zero rows are dropped
            # from replay to save compute. Pruned placeholders remain excluded.
            if not is_pruned:
                tree_total_segment_count += 1
                queue_released_segment_count[queue_flush_id] = (
                    queue_released_segment_count.get(queue_flush_id, 0) + 1
                )
                # PLAN.md P0.2: per-parent realized-child snapshot so
                # ``realized_child_count == allocated_k`` can be enforced by
                # ``_enforce_fresh_iid_invariants`` even after the retained
                # rows shrink under zero-advantage filtering.
                slot = realized_by_parent.setdefault(
                    str(parent_group_id),
                    {"realized": 0, "allocated_k": int(allocated_k)},
                )
                slot["realized"] += 1
                # allocated_k may only be stamped on some children when a
                # legacy fixture is used; prefer the max we saw.
                if int(allocated_k) > slot["allocated_k"]:
                    slot["allocated_k"] = int(allocated_k)
            # Stage 1: the zero filter uses the EXACT scalar that
            # token_fields_for_edges broadcasts into the policy `advantages`
            # tensor (edge["advantage"]), never diagnostic pav_advantage.
            # Despite the legacy flag name, true keeps positive and negative
            # advantages and removes only exact-zero rows.
            if (not is_pruned or emit_pruned_edges) and (
                not only_adv_greater_than_zero or advantage != 0
            ):
                edges.append(edge)
                if not is_pruned:
                    trainable_child_count += 1
                    expanded_parent_group_ids.add(str(parent_group_id))
                    queue_to_parent_group_counts.setdefault(
                        queue_flush_id, set()
                    ).add(str(parent_group_id))

        for child in node.get("children", []):
            visit(child, node)
        node.pop("children", None)

    visit(tree_copy)
    # PLAN.md P0.N1 + Stage 1: tree-level construction counts. Stamp them on
    # every retained edge so replay / tensorization / actor can index by tree
    # without another tree walk. ``tree_total_segment_count`` is the pre-filter
    # realized non-pruned segment count; queue_released_segment_count[q] is the
    # matching pre-filter queue count for logging / theoretical validation.
    tree_summary = {
        "tree_id": str(tree_id),
        "expanded_parent_group_count": len(expanded_parent_group_ids),
        "trainable_child_count": int(trainable_child_count),
        "tree_total_segment_count": int(tree_total_segment_count),
        "queue_to_parent_group_counts": {
            str(k): len(v) for k, v in queue_to_parent_group_counts.items()
        },
        "queue_released_segment_count": {
            str(k): int(v) for k, v in queue_released_segment_count.items()
        },
    }
    for edge in edges:
        edge.setdefault("tree_summary", tree_summary)
        # Preserve on every row so downstream splits do not lose it.
        edge["tree_total_segment_count"] = int(tree_total_segment_count)
        qid = edge.get("queue_flush_id", 0)
        edge["queue_released_segment_count"] = int(
            queue_released_segment_count.get(qid, 0)
        )
        # PLAN.md P0.5: persist the parent's pre-filter realized child count
        # on every edge. Downstream sparse-vs-dense parity checks compare
        # ``realized_child_count`` (unaffected by zero-adv filtering) against
        # ``allocated_k``.
        pgid = str(edge.get("parent_group_id", ""))
        slot = realized_by_parent.get(pgid)
        if slot is not None:
            edge["realized_child_count"] = int(slot["realized"])

    # PLAN.md P0.2: fresh_iid invariants — realized_child_count == allocated_k
    # is a construction invariant checked from the pre-filter snapshot; the
    # retained-row check is done inside ``_enforce_fresh_iid_invariants``.
    if strict_fresh_iid:
        _enforce_realized_child_count_equals_allocated_k(realized_by_parent)
        _enforce_fresh_iid_invariants(edges)

    return json.loads(json.dumps(edges))


def _enforce_realized_child_count_equals_allocated_k(
    realized_by_parent: dict[str, dict[str, int]],
) -> None:
    """PLAN.md P0.2 construction invariant: ``realized_child_count == allocated_k``.

    Independent of ``only_adv_greater_than_zero`` — this looks at the
    pre-filter tree walk snapshot, before any zero-advantage row is dropped.
    """
    failures: list[str] = []
    for pgid, slot in realized_by_parent.items():
        allocated = int(slot.get("allocated_k", 0))
        realized = int(slot.get("realized", 0))
        if allocated > 0 and realized != allocated:
            failures.append(
                f"fresh_iid parent_group_id={pgid!r} realized {realized} children "
                f"but allocated_k={allocated}"
            )
    if failures:
        raise ValueError(
            "fresh_iid invariants failed (PLAN.md P0.2):\n  " + "\n  ".join(failures)
        )


def _enforce_fresh_iid_invariants(edges: list[dict[str, Any]]) -> None:
    """PLAN.md P0.2: for every fresh_iid parent group, retained rows must
    obey ``retained_row_count <= realized_child_count == allocated_k`` after
    zero-advantage filtering, and every retained row must have
    ``sample_multiplicity == 1``.

    Pruned placeholders are excluded from ``edges`` upstream, so this check
    only looks at retained training rows. The realized-child count (before
    the advantage filter) comes from the per-tree summary and is guaranteed
    to equal ``allocated_k`` by tree construction.
    """
    from collections import defaultdict

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        by_parent[str(edge.get("parent_group_id", ""))].append(edge)
    failures: list[str] = []
    for pgid, group in by_parent.items():
        alloc_values = {int(e.get("allocated_k", 0) or 0) for e in group}
        if len(alloc_values) != 1:
            failures.append(
                f"parent_group_id={pgid!r} has inconsistent allocated_k={alloc_values}"
            )
            continue
        allocated_k = next(iter(alloc_values), 0)
        mults = [int(e.get("sample_multiplicity", 1) or 1) for e in group]
        if any(m != 1 for m in mults):
            failures.append(
                f"fresh_iid parent_group_id={pgid!r} has sample_multiplicity != 1: {mults}"
            )
        # PLAN.md P0.2: allow retained rows to be a subset of the realized
        # children (zero-advantage rows may have been filtered out). The
        # pre-filter equality lives in tree construction and is asserted by
        # the tree_summary.tree_total_segment_count. Here we just guard
        # against an over-count.
        if allocated_k and len(group) > allocated_k:
            failures.append(
                f"fresh_iid parent_group_id={pgid!r} retained {len(group)} rows "
                f"which exceeds allocated_k={allocated_k}"
            )
    if failures:
        raise ValueError(
            "fresh_iid invariants failed (PLAN.md P0.2):\n  " + "\n  ".join(failures)
        )


def token_fields_for_edges(
    edges: Iterable[dict[str, Any]],
    response_mask: torch.Tensor,
    *,
    include_old_log_probs: bool = True,
) -> dict[str, torch.Tensor]:
    """Broadcast edge-level scalars to verl token-level tensors.

    ``response_mask`` defines the valid response length per row. This mirrors
    treetune's tree generators, which repeat each edge scalar over every token in
    the child segment/response.
    """
    edge_list = list(edges)
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be 2D, got shape {tuple(response_mask.shape)}")
    if len(edge_list) != response_mask.shape[0]:
        raise ValueError(f"got {len(edge_list)} edges for batch size {response_mask.shape[0]}")

    dtype = torch.float32
    device = response_mask.device
    advantages = torch.zeros_like(response_mask, dtype=dtype, device=device)
    values = torch.zeros_like(response_mask, dtype=dtype, device=device)
    returns = torch.zeros_like(response_mask, dtype=dtype, device=device)
    rewards = torch.zeros_like(response_mask, dtype=dtype, device=device)
    old_log_probs = torch.zeros_like(response_mask, dtype=dtype, device=device)
    edge_weights = torch.zeros_like(response_mask, dtype=dtype, device=device)
    has_old_log_probs = False
    has_edge_weights = False

    for row, edge in enumerate(edge_list):
        valid_len = int(response_mask[row].sum().item())
        if valid_len <= 0:
            continue
        advantages[row, :valid_len] = float(edge["advantage"])
        values[row, :valid_len] = float(edge.get("value", edge.get("reward", 0.0)))
        returns[row, :valid_len] = float(edge.get("value", edge.get("reward", 0.0)))
        rewards[row, valid_len - 1] = float(edge.get("reward", 0.0))

        maybe_log_probs = edge.get("actor_shifted_log_probs")
        if include_old_log_probs and maybe_log_probs is not None:
            if len(maybe_log_probs) != valid_len:
                raise ValueError(
                    f"edge {row} has {len(maybe_log_probs)} old logprobs for valid length {valid_len}"
                )
            old_log_probs[row, :valid_len] = torch.as_tensor(maybe_log_probs, dtype=dtype, device=device)
            has_old_log_probs = True

        maybe_weight = edge.get(
            "edge_weight",
            edge.get("vdra_representative_weight", edge.get("vdra_edge_weight")),
        )
        weight = 1.0 if maybe_weight is None else float(maybe_weight)
        if maybe_weight is not None:
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"edge {row} has invalid edge weight {maybe_weight!r}")
            has_edge_weights = True
        edge_weights[row, :valid_len] = weight

    tensors = {
        "advantages": advantages,
        "values": values,
        "returns": returns,
        "token_level_rewards": rewards,
    }
    if has_old_log_probs:
        tensors["old_log_probs"] = old_log_probs
    if has_edge_weights:
        tensors["edge_weights"] = edge_weights
    return tensors


def add_tree_advantage_tensors(data: Any, edges: Iterable[dict[str, Any]], *, response_mask_key: str = "response_mask") -> Any:
    """Mutate and return a verl ``DataProto`` with precomputed tree tensors."""
    if response_mask_key not in data.batch.keys():
        raise KeyError(f"DataProto.batch is missing {response_mask_key!r}")
    for key, value in token_fields_for_edges(edges, data.batch[response_mask_key]).items():
        data.batch[key] = value
    return data