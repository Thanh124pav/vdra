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
            expected = next(iter(allocated_values), 0)
            if strict_fresh_iid and expected and len(group) != expected:
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
