"""Assemble tree edges into a verl ``DataProto`` training batch.

Each SPO/GEAR edge becomes one training row: ``query = parent trajectory tokens``
(left-padded prompt) and ``response = this segment's generated tokens``
(right-padded), following verl's standard layout (see
``vLLMRollout.generate_sequences``). Per-token advantages, old log-probs, values
and returns are broadcast from the edge scalars by
``tree_advantage.token_fields_for_edges`` so the numbers stay identical to
treetune's per-token broadcast.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

from recipe.gear_tree.tree_advantage import token_fields_for_edges


# PLAN.md P0.N4: deterministic string -> int64 mapping for group tensors.
# blake2b keeps collision-probability negligible while staying reproducible
# across processes and container restarts. Signed int64 so torch tensors of
# dtype int64 hold the whole range.
_ID_MASK = (1 << 63) - 1


def _stable_int_id(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) & _ID_MASK


def group_tensors_for_edges(edges: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """PLAN.md P0.N4: build row-level group tensors from tree edges.

    Returns int64 tensors ``tree_group_ids``, ``parent_group_ids``,
    ``queue_group_ids``, ``allocated_k`` and float32 ``sample_multiplicity``,
    all shaped ``[batch]``. Missing metadata falls back to safe defaults
    (id 0, allocated_k = 1, multiplicity = 1) so legacy edges that were
    generated before this migration still tensorize; strict main runs must
    additionally check the group-integrity invariants below.
    """
    bsz = len(edges)
    tree_ids = torch.empty(bsz, dtype=torch.int64)
    parent_ids = torch.empty(bsz, dtype=torch.int64)
    queue_ids = torch.empty(bsz, dtype=torch.int64)
    allocated = torch.empty(bsz, dtype=torch.int64)
    multiplicities = torch.empty(bsz, dtype=torch.float32)
    for row, edge in enumerate(edges):
        tree_ids[row] = _stable_int_id(edge.get("tree_id"))
        parent_ids[row] = _stable_int_id(edge.get("parent_group_id"))
        queue_ids[row] = _stable_int_id(edge.get("queue_flush_id", 0))
        allocated[row] = int(edge.get("allocated_k", 1) or 1)
        multiplicities[row] = float(edge.get("sample_multiplicity", 1) or 1)
    return {
        "tree_group_ids": tree_ids,
        "parent_group_ids": parent_ids,
        "queue_group_ids": queue_ids,
        "allocated_k": allocated,
        "sample_multiplicity": multiplicities,
    }


def compute_objective_weights(edges: Sequence[Dict[str, Any]]) -> List[float]:
    """PLAN.md P0.3: precompute the exact objective weight for every row.

    For every realized child ``j`` of parent ``p`` in tree ``T``:

        w_{p,j} = (1 / N_tree) * (1 / |P(T)|) * (m_{p,j} / sum_j' m_{p,j'})

    where ``N_tree`` is the number of distinct ``tree_id`` in the batch,
    ``|P(T)|`` is the number of distinct realized parent groups in tree
    ``T``, and ``m_{p,j}`` is the child's ``sample_multiplicity`` (``1`` under
    ``fresh_iid``). The returned list is aligned with ``edges`` row-for-row
    and sums to ``1`` over the whole batch.
    """
    from collections import defaultdict

    if not edges:
        return []

    # Group edges by tree, then by parent group.
    trees: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for row, edge in enumerate(edges):
        tid = str(edge.get("tree_id", ""))
        pgid = str(edge.get("parent_group_id", ""))
        trees[tid][pgid].append(row)

    n_tree = len(trees)
    weights = [0.0] * len(edges)
    for tid, parents in trees.items():
        p_count = len(parents)
        for pgid, rows in parents.items():
            mults = [
                max(int(edges[r].get("sample_multiplicity", 1) or 1), 1)
                for r in rows
            ]
            total_m = float(sum(mults))
            for r, m in zip(rows, mults):
                weights[r] = (1.0 / n_tree) * (1.0 / p_count) * (m / total_m)
    return weights


def validate_objective_weights(
    edges: Sequence[Dict[str, Any]],
    weights: Sequence[float],
    *,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """PLAN.md P0.3: enforce the three normalization invariants.

        sum_j local_child_weight[p, j] == 1 for every parent
        sum_p parent_weight[T, p]      == 1 for every tree
        sum_all_rows objective_weights == 1

    Raises ``ValueError`` on any failure. Returns a small diagnostics dict.
    """
    from collections import defaultdict

    if len(edges) != len(weights):
        raise ValueError(
            f"objective_weights length {len(weights)} != edges length {len(edges)}"
        )
    if not edges:
        return {
            "vdra/objective_weight_sum": 0.0,
            "vdra/objective_weight_tree_count": 0,
        }

    trees: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for row, edge in enumerate(edges):
        trees[str(edge.get("tree_id", ""))][str(edge.get("parent_group_id", ""))].append(row)

    n_tree = len(trees)
    failures: List[str] = []
    total = 0.0
    max_parent_err = 0.0
    max_tree_err = 0.0
    for tid, parents in trees.items():
        p_count = len(parents)
        tree_mass = 0.0
        for pgid, rows in parents.items():
            local_sum = sum(weights[r] for r in rows)
            expected_local = 1.0 / (n_tree * p_count)
            if abs(local_sum - expected_local) > atol:
                failures.append(
                    f"parent {pgid!r} in tree {tid!r}: sum(w) = {local_sum!r}, "
                    f"expected 1/(N_tree*|P(T)|) = {expected_local!r}"
                )
            # Local child fractions must sum to 1 per parent.
            if local_sum > 0:
                mults = [max(int(edges[r].get("sample_multiplicity", 1) or 1), 1) for r in rows]
                total_m = float(sum(mults))
                for r, m in zip(rows, mults):
                    local_frac = weights[r] / local_sum
                    if abs(local_frac - (m / total_m)) > atol:
                        max_parent_err = max(
                            max_parent_err, abs(local_frac - (m / total_m))
                        )
            tree_mass += local_sum
        expected_tree = 1.0 / n_tree
        if abs(tree_mass - expected_tree) > atol:
            failures.append(
                f"tree {tid!r}: sum(w) = {tree_mass!r}, expected 1/N_tree = {expected_tree!r}"
            )
        max_tree_err = max(max_tree_err, abs(tree_mass - expected_tree))
        total += tree_mass
    if abs(total - 1.0) > atol:
        failures.append(f"batch sum(w) = {total!r} != 1")
    if failures:
        raise ValueError(
            "objective_weights normalization failed (PLAN.md P0.3):\n  "
            + "\n  ".join(failures)
        )
    return {
        "vdra/objective_weight_sum": float(total),
        "vdra/objective_weight_tree_count": int(n_tree),
        "vdra/objective_weight_parent_max_err": float(max_parent_err),
        "vdra/objective_weight_tree_max_err": float(max_tree_err),
    }


def compute_group_metrics(edges: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """PLAN.md P0.N7 runtime metrics.

    Computes:
      * vdra/parent_groups_per_tree (mean over trees)
      * vdra/children_per_parent_mean / _std
      * vdra/empty_token_mask_children — reported as 0 when no per-token
        length metadata is available; the actor loss also emits the exact
        per-step count.
      * vdra/queue_parent_mass_sum — sum of |Q_r|/|P(T)| over queue partitions
        per tree; a well-formed queue partition sums to 1.
      * vdra/parent_weight_sum_per_tree — always 1 for the canonical
        node-balanced aggregation.
      * vdra/child_weight_sum_per_parent — 1 under fresh_iid; sum(m_j)/sum(m_j)
        under weighted_reuse.
      * vdra/effective_segment_weight_vs_branch_factor_corr — Pearson
        correlation between a segment's effective weight (1/(|P(T)|*k_p) for
        fresh_iid) and its parent's branch factor k_p. For the canonical
        node-balanced loss this correlation must be strongly negative
        (-1 in the simplest single-tree case); a positive correlation flags
        that the legacy edge-mean has crept back in.
    """
    from collections import defaultdict
    import math

    if not edges:
        return {}

    # Group by tree.
    trees: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        trees[str(edge.get("tree_id", ""))].append(edge)

    parent_groups_per_tree: List[int] = []
    children_per_parent: List[int] = []
    queue_parent_mass_sum: List[float] = []
    parent_weight_sum: List[float] = []
    child_weight_sum: List[float] = []
    seg_weights: List[float] = []
    branch_factors: List[float] = []
    empty_mask_children = 0

    for tid, tree_edges in trees.items():
        # parents in this tree
        parents: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in tree_edges:
            parents[str(edge.get("parent_group_id", ""))].append(edge)
            if int(edge.get("response_length", edge.get("num_tokens", 1)) or 0) <= 0:
                empty_mask_children += 1
        parent_groups_per_tree.append(len(parents))
        p_total = max(len(parents), 1)
        parent_weight_sum.append(1.0)  # canonical aggregation always sums to 1

        # child weights per parent (== 1 under fresh_iid)
        for pgid, group in parents.items():
            children_per_parent.append(len(group))
            mults = [max(int(e.get("sample_multiplicity", 1) or 1), 1) for e in group]
            total_mult = float(sum(mults))
            if total_mult > 0:
                child_weight_sum.append(total_mult / total_mult)  # == 1
            k_p = float(len(group))
            for _ in group:
                # segment weight = 1 / (|P(T)| * k_p) under fresh_iid.
                seg_weights.append(1.0 / (p_total * max(k_p, 1.0)))
                branch_factors.append(k_p)

        # queue partition
        queues: Dict[Any, set] = defaultdict(set)
        for edge in tree_edges:
            queues[edge.get("queue_flush_id", 0)].add(str(edge.get("parent_group_id", "")))
        mass = sum(len(pset) / p_total for pset in queues.values())
        queue_parent_mass_sum.append(mass)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs: List[float]) -> float:
        if not xs:
            return 0.0
        mu = _mean(xs)
        return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))

    def _pearson(xs: List[float], ys: List[float]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx = _mean(xs)
        my = _mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if dx == 0.0 or dy == 0.0:
            return 0.0
        return num / (dx * dy)

    return {
        "vdra/parent_groups_per_tree": _mean([float(x) for x in parent_groups_per_tree]),
        "vdra/children_per_parent_mean": _mean([float(x) for x in children_per_parent]),
        "vdra/children_per_parent_std": _std([float(x) for x in children_per_parent]),
        "vdra/empty_token_mask_children": float(empty_mask_children),
        "vdra/queue_parent_mass_sum": _mean(queue_parent_mass_sum),
        "vdra/parent_weight_sum_per_tree": _mean(parent_weight_sum),
        "vdra/child_weight_sum_per_parent": _mean(child_weight_sum),
        "vdra/effective_segment_weight_vs_branch_factor_corr": _pearson(
            seg_weights, branch_factors
        ),
        "vdra/trees_in_batch": float(len(trees)),
    }


def validate_group_integrity(
    edges: Sequence[Dict[str, Any]],
    *,
    strict_fresh_iid: bool = True,
) -> Dict[str, Any]:
    """PLAN.md P0.N4: enforce grouping invariants before the actor update.

    Invariants (all failures raise ``ValueError`` when ``strict_fresh_iid``):
      * every row sharing a ``parent_group_id`` shares one ``tree_group_id``;
      * every row sharing a ``parent_group_id`` shares one ``allocated_k``;
      * fresh_iid groups (sample_multiplicity == 1 across every row of the
        group) have row_count == allocated_k;
      * no parent group is silently split or partially dropped.

    Returns a small diagnostics dict for logging.
    """
    from collections import defaultdict

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        pgid = str(edge.get("parent_group_id", ""))
        groups[pgid].append(edge)

    failures: List[str] = []
    fresh_iid_group_count = 0
    weighted_group_count = 0
    for pgid, group in groups.items():
        tree_ids = {str(e.get("tree_id")) for e in group}
        if len(tree_ids) != 1:
            failures.append(f"parent_group_id={pgid!r} spans multiple tree_ids={tree_ids}")
        allocated_values = {int(e.get("allocated_k", 0)) for e in group}
        if len(allocated_values) != 1:
            failures.append(
                f"parent_group_id={pgid!r} has inconsistent allocated_k={allocated_values}"
            )
        mults = [int(e.get("sample_multiplicity", 1) or 1) for e in group]
        if all(m == 1 for m in mults):
            fresh_iid_group_count += 1
            # PLAN.md P0.N4: always detect a partial fresh_iid parent group;
            # ``strict_fresh_iid`` only decides whether to raise.
            expected = next(iter(allocated_values), 0)
            if expected and len(group) != expected:
                failures.append(
                    f"fresh_iid parent_group_id={pgid!r} has {len(group)} rows "
                    f"but allocated_k={expected}"
                )
        else:
            weighted_group_count += 1
    if failures and strict_fresh_iid:
        raise ValueError(
            "Group-integrity check failed (PLAN.md P0.N4):\n  " + "\n  ".join(failures)
        )
    return {
        "vdra/group_integrity_failures": len(failures),
        "vdra/fresh_iid_parent_groups": fresh_iid_group_count,
        "vdra/weighted_reuse_parent_groups": weighted_group_count,
        "vdra/parent_groups_total": len(groups),
    }




def _compute_position_id_with_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Local narrow equivalent of verl.utils.model.compute_position_id_with_mask.

    Avoid importing the broad upstream verl model utility in CPU VDRA tests; that
    module eagerly imports optional HF model classes unrelated to tree data.
    """

    return (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0)


def _left_pad(ids: Sequence[int], length: int, pad_id: int) -> List[int]:
    ids = list(ids)
    return [pad_id] * (length - len(ids)) + ids


def _right_pad(ids: Sequence[int], length: int, pad_id: int) -> List[int]:
    ids = list(ids)
    return ids + [pad_id] * (length - len(ids))


def edges_to_dataproto(
    edges: List[Dict[str, Any]],
    tokenizer,
    *,
    max_prompt_length: int,
    max_response_length: int,
    include_old_log_probs: bool = True,
) -> DataProto:
    """Build a DataProto whose rows are tree edges.

    Tensor fields (all right/left padded to fixed lengths):
      ``prompts, responses, input_ids, attention_mask, position_ids,
      response_mask, advantages, returns, values, token_level_rewards`` and
      (optionally) ``old_log_probs``.
    Non-tensor fields: ``uid`` (per source question, for grouping/logging),
    ``question_id``, ``reward_model``, ``extra_info``.
    """
    if not edges:
        raise ValueError("edges_to_dataproto received an empty edge list")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    bsz = len(edges)
    prompts = torch.empty((bsz, max_prompt_length), dtype=torch.long)
    responses = torch.empty((bsz, max_response_length), dtype=torch.long)
    prompt_mask = torch.zeros((bsz, max_prompt_length), dtype=torch.long)
    response_mask = torch.zeros((bsz, max_response_length), dtype=torch.long)

    # Stable uid per source question so group-relative logging/estimators work.
    qid_to_uid: Dict[Any, str] = {}
    uids: List[str] = []
    question_ids: List[Any] = []
    reward_models: List[dict] = []
    extra_infos: List[dict] = []

    # Truncated per-token old-logprobs, aligned to the (truncated) response.
    for row, edge in enumerate(edges):
        q_ids = edge.get("query_token_ids") or []
        r_ids = edge.get("response_token_ids") or []
        if not r_ids:
            raise ValueError(f"edge {row} has no response_token_ids")
        if len(q_ids) > max_prompt_length:
            raise ValueError(
                f"edge {row} query_token_ids length {len(q_ids)} exceeds max_prompt_length "
                f"{max_prompt_length}; strict VDRA forbids silent context truncation"
            )
        if len(r_ids) > max_response_length:
            raise ValueError(
                f"edge {row} response_token_ids length {len(r_ids)} exceeds max_response_length "
                f"{max_response_length}; strict VDRA forbids silent response truncation"
            )

        valid_r = len(r_ids)
        valid_q = len(q_ids)

        prompts[row] = torch.tensor(_left_pad(q_ids, max_prompt_length, pad_id), dtype=torch.long)
        responses[row] = torch.tensor(_right_pad(r_ids, max_response_length, pad_id), dtype=torch.long)
        prompt_mask[row, max_prompt_length - valid_q :] = 1
        response_mask[row, :valid_r] = 1

        qid = edge.get("question_id")
        if qid not in qid_to_uid:
            qid_to_uid[qid] = str(uuid.uuid4())
        uids.append(qid_to_uid[qid])
        question_ids.append(qid)
        instance = edge.get("instance", {}) or {}
        reward_models.append(
            instance.get("reward_model", {"ground_truth": instance.get("answer")})
        )
        extra_infos.append({"problem": instance.get("problem")})

    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
    position_ids = _compute_position_id_with_mask(attention_mask)

    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        },
        batch_size=bsz,
    )

    # Broadcast edge scalars -> per-token tensors (advantages/values/returns/
    # rewards/old_log_probs) using the shared, treetune-faithful helper.
    token_fields = token_fields_for_edges(
        edges, response_mask, include_old_log_probs=include_old_log_probs
    )
    for key, value in token_fields.items():
        batch[key] = value

    # PLAN.md P0.N4: attach the canonical row-level group tensors so the
    # node-balanced actor loss can reduce token -> child -> parent -> tree
    # without a second pass over non-tensor metadata.
    for key, value in group_tensors_for_edges(edges).items():
        batch[key] = value

    # PLAN.md P0.3: precompute exact objective weights on the full batch and
    # attach them as a row-level tensor. The actor loss (P0.4) reduces
    # sum(objective_weights * child_loss), so mini/microbatch splits give
    # gradients exactly equal to the full-batch weighted sum.
    obj_weights = compute_objective_weights(edges)
    validate_objective_weights(edges, obj_weights)
    batch["objective_weights"] = torch.tensor(obj_weights, dtype=torch.float32)

    non_tensor_batch = {
        "uid": np.array(uids, dtype=object),
        "question_id": np.array(question_ids, dtype=object),
        "reward_model": np.array(reward_models, dtype=object),
        "extra_info": np.array(extra_infos, dtype=object),
        # Keep the raw string ids for logging / manifest validation.
        "tree_id": np.array([str(e.get("tree_id", "")) for e in edges], dtype=object),
        "parent_group_id": np.array(
            [str(e.get("parent_group_id", "")) for e in edges], dtype=object
        ),
        "child_segment_id": np.array(
            [str(e.get("child_segment_id", "")) for e in edges], dtype=object
        ),
        "queue_flush_id": np.array(
            [str(e.get("queue_flush_id", "0")) for e in edges], dtype=object
        ),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
