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

import uuid
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

from recipe.gear_tree.tree_advantage import token_fields_for_edges




def _compute_position_id_with_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Local narrow equivalent of verl.utils.model.compute_position_id_with_mask.

    Avoid importing the broad upstream verl model utility in CPU VDRA tests; that
    module eagerly imports optional HF model classes unrelated to tree data.
    """

    return (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0)


def _left_pad(ids: Sequence[int], length: int, pad_id: int) -> List[int]:
    ids = list(ids)[-length:]  # left-truncate to keep the most recent context
    return [pad_id] * (length - len(ids)) + ids


def _right_pad(ids: Sequence[int], length: int, pad_id: int) -> List[int]:
    ids = list(ids)[:length]  # right-truncate over-long responses
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

        valid_r = min(len(r_ids), max_response_length)
        valid_q = min(len(q_ids), max_prompt_length)

        prompts[row] = torch.tensor(_left_pad(q_ids, max_prompt_length, pad_id), dtype=torch.long)
        responses[row] = torch.tensor(_right_pad(r_ids, max_response_length, pad_id), dtype=torch.long)
        prompt_mask[row, max_prompt_length - valid_q :] = 1
        response_mask[row, :valid_r] = 1

        # Keep old-logprobs consistent with the (possibly truncated) response.
        alp = edge.get("actor_shifted_log_probs")
        if alp is not None:
            edge["actor_shifted_log_probs"] = list(alp)[:valid_r]

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

    non_tensor_batch = {
        "uid": np.array(uids, dtype=object),
        "question_id": np.array(question_ids, dtype=object),
        "reward_model": np.array(reward_models, dtype=object),
        "extra_info": np.array(extra_infos, dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
