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
    only_adv_greater_than_zero: bool = True,
    tree_update_mode: str = "spo",
    treepo_global_weight: float = 0.5,
    treerl_gamma: float = 0.9,
    emit_pruned_edges: bool = True,
) -> list[dict[str, Any]]:
    """Extract treetune-compatible training edges from a generated tree.

    This ports ``TreeEpisodeUtils.extract_edges_from_tree`` while reading both
    ``reward`` and legacy ``score`` fields. Pruned GEAR edges may be emitted with
    zero advantage so downstream tensor shapes can remain aligned with rollout
    rows; set ``emit_pruned_edges=False`` to drop them.
    """
    edges: list[dict[str, Any]] = []
    tree_copy = copy.deepcopy(tree)
    data_instance = tree_copy.get("_request_object", {})
    question_id = data_instance.get("_treetune__idx", data_instance.get("uid"))
    root_reward = _node_reward(tree_copy)
    policy_snapshot_id = (
        tree_copy.get("policy_snapshot_id")
        or tree_copy.get("vdra_policy_snapshot_id")
        or data_instance.get("policy_snapshot_id")
        or data_instance.get("current_rollout_snapshot_id")
    )

    def visit(node: dict[str, Any], parent: dict[str, Any] | None = None) -> None:
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
                **update_values,
            }
            edge["advantage"] = advantage
            if (not is_pruned or emit_pruned_edges) and (
                not only_adv_greater_than_zero or pav_advantage != 0
            ):
                edges.append(edge)

        for child in node.get("children", []):
            visit(child, node)
        node.pop("children", None)

    visit(tree_copy)
    return json.loads(json.dumps(edges))


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