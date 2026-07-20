# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "edge_weights" in data.batch.keys():
            select_keys.append("edge_weights")
        # PLAN.md P0.N4/N5/P0.3: forward the row-level group tensors emitted
        # by ``tree_data.edges_to_dataproto`` to the node-balanced PPO loss.
        # Legacy losses ignore them; the vdra_node_balanced_ppo loss reduces
        # child -> parent -> tree through them, or uses objective_weights
        # directly when precomputed.
        for key in (
            "parent_group_ids",
            "tree_group_ids",
            "queue_group_ids",
            "allocated_k",
            "sample_multiplicity",
            "objective_weights",
            # PLAN.md P0.4: segment-average VDRA main loss forwards
            # per-row segment weights + tree_total_segment_count so
            # microbatch splits preserve the pre-filter denominator.
            "segment_objective_weights",
            "tree_total_segment_count",
            # PLAN.md §1.2/§1.3: trainer-stamped logical-batch structure for
            # the canonical paper aggregations (sparse-execution contract).
            "logical_batch_index",
            "is_dummy",
        ):
            if key in data.batch.keys():
                select_keys.append(key)

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # PLAN.md §1.3 (user decision 2026-07-21): the canonical paper
        # aggregations group mini-batches by the trainer-stamped LOGICAL
        # optimizer batch, whose pre-filter denominators M_B / T_B were fixed
        # at reservation time. Every other mode keeps verl's fixed-size split.
        _split_loss_mode = str(self.config.policy_loss.get("loss_mode", "vanilla"))
        _aggregation = str(
            self.config.policy_loss.get(
                "policy_aggregation", "tree_balanced_segment_mean"
            )
        ).strip().lower()
        vdra_canonical_aggregation = (
            _split_loss_mode == "vdra_segment_mean_ppo"
            and _aggregation in ("segment_mean", "token_mean")
        )
        logical_denominators = None
        vdra_dp_size = 1
        n_all_zero_logical_batches = 0
        if vdra_canonical_aggregation:
            if "logical_batch_index" not in data.batch.keys():
                raise ValueError(
                    "policy_aggregation="
                    f"{_aggregation!r} requires trainer-stamped logical "
                    "batches (logical_batch_index tensor from "
                    "build_logical_update_batch); refusing to fall back to a "
                    "retained-row split (PLAN.md §1.2/§1.3)."
                )
            seg_counts = data.meta_info.get("original_logical_segment_count")
            tok_counts = data.meta_info.get("original_logical_token_count")
            if not seg_counts or not tok_counts or len(seg_counts) != len(tok_counts):
                raise ValueError(
                    "canonical VDRA aggregation requires per-logical-batch "
                    "original_logical_segment_count / "
                    "original_logical_token_count lists in meta_info "
                    "(PLAN.md §1.3); got "
                    f"{seg_counts!r} / {tok_counts!r}."
                )
            if self.ulysses_sequence_parallel_size > 1:
                raise NotImplementedError(
                    "canonical VDRA aggregations are only specified for "
                    "ulysses_sequence_parallel_size == 1; the data-parallel "
                    "group for the dp-size reducer compensation is undefined "
                    "under sequence parallel (PLAN.md §1.3 — discuss before "
                    "enabling)."
                )
            _world = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
            _stamped_dp = int(data.meta_info.get("logical_dp_size", _world))
            if _stamped_dp != _world:
                raise ValueError(
                    "trainer padded logical batches for dp_size="
                    f"{_stamped_dp} but the actual data-parallel world size "
                    f"is {_world}; rank shares would diverge (PLAN.md §1.2)."
                )
            vdra_dp_size = _world
            _idx = data.batch["logical_batch_index"]
            mini_batches = []
            logical_denominators = []
            for _k in range(len(seg_counts)):
                _sel = torch.nonzero(_idx == _k, as_tuple=False).squeeze(-1)
                if _sel.numel() == 0:
                    # All-zero logical batch: no rows on ANY rank by
                    # construction, so every rank skips this optimizer step
                    # consistently (PLAN.md §1.3 approved skip).
                    n_all_zero_logical_batches += 1
                    continue
                mini_batches.append(data[_sel])
                logical_denominators.append(
                    (float(seg_counts[_k]), float(tok_counts[_k]))
                )
        else:
            # Split to make minibatch iterator for updating the actor
            # See PPO paper for details. https://arxiv.org/abs/1707.06347
            mini_batches = data.split(self.config.ppo_mini_batch_size)

        # P0.4: replay/tree edges carry stored generation-time old_log_probs
        # that must be preserved as the PPO denominator, even in the
        # single-minibatch/single-epoch shape that would otherwise look
        # "on-policy". Trainers set force_stored_old_log_probs=True to opt in.
        force_stored_old_log_probs = bool(
            data.meta_info.get("force_stored_old_log_probs", False)
        )
        on_policy = (
            len(mini_batches) == 1
            and self.config.ppo_epochs == 1
            and not force_stored_old_log_probs
        )

        metrics = {}
        # PLAN.md P0.3: count actual optimizer.step() calls so the trainer can
        # advance ``global_step`` by the correct amount for this update.
        num_optimizer_steps = 0
        # The original selected-slot count for this optimizer batch feeds the
        # labeled batch-slot ablation loss (N_B); the canonical VDRA loss uses
        # the original batch's unique-tree count (N_T) instead. Both are fixed
        # per mini_batch BEFORE micro/dynamic splitting so microbatch splits
        # preserve the loss weights w_s = 1 / (N_T * N_seg(T)).
        original_optimizer_batch_slot_count = len(mini_batches[0]) if mini_batches else 0
        original_optimizer_batch_tree_count = None
        logical_segment_count = None
        logical_token_count = None
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                original_optimizer_batch_slot_count = len(mini_batch)
                if logical_denominators is not None:
                    # PLAN.md §1.3: fixed pre-filter denominators of THIS
                    # logical batch, stamped at reservation time — reused by
                    # every micro-batch and identical on every DP rank.
                    logical_segment_count, logical_token_count = (
                        logical_denominators[batch_idx]
                    )
                if "tree_group_ids" in mini_batch.batch.keys():
                    original_optimizer_batch_tree_count = int(
                        torch.unique(mini_batch.batch["tree_group_ids"]).numel()
                    )
                else:
                    original_optimizer_batch_tree_count = None
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # PLAN.md P0.4/P0.5: the canonical VDRA losses
                    # (``vdra_segment_mean_ppo`` main, ``vdra_node_balanced_ppo``
                    # ablation) return a globally weighted sum
                    # (``sum_row w_row * L_row``), so summing across
                    # microbatches already reproduces the full-batch weighted
                    # objective. Re-scaling by 1/gradient_accumulation would
                    # divide the partial numerator a second time.
                    _loss_mode = str(self.config.policy_loss.get("loss_mode", "vanilla"))
                    vdra_mode = _loss_mode in (
                        "vdra_node_balanced_ppo",
                        "vdra_segment_mean_ppo",
                    )
                    if vdra_canonical_aggregation:
                        # PLAN.md §1.3: the FSDP/DDP reducer AVERAGES rank
                        # gradients (measured, docs/h1_fsdp_parity_report.md)
                        # while the canonical loss divides the LOCAL numerator
                        # by the GLOBAL logical denominator — multiply by the
                        # actual data-parallel size so the average reproduces
                        # the single-rank objective exactly.
                        loss_scale_factor = float(vdra_dp_size)
                    elif vdra_mode:
                        loss_scale_factor = 1.0
                    elif self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout importance sampling weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    edge_weights = model_inputs.get("edge_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (all functions return 4 values)
                    loss_kwargs = {
                        "old_log_prob": old_log_prob,
                        "log_prob": log_prob,
                        "advantages": advantages,
                        "response_mask": response_mask,
                        "loss_agg_mode": loss_agg_mode,
                        "config": self.config,
                        "rollout_is_weights": rollout_is_weights,
                    }
                    if edge_weights is not None and "edge_weights" in policy_loss_fn.__code__.co_varnames:
                        loss_kwargs["edge_weights"] = edge_weights
                    # PLAN.md P0.N5/P0.3: pass VDRA group tensors and the
                    # precomputed objective_weights to any loss that declares
                    # them (currently vdra_node_balanced_ppo).
                    for group_key in (
                        "parent_group_ids",
                        "tree_group_ids",
                        "queue_group_ids",
                        "allocated_k",
                        "sample_multiplicity",
                        "objective_weights",
                        "segment_objective_weights",
                        "tree_total_segment_count",
                    ):
                        maybe = model_inputs.get(group_key, None)
                        if maybe is not None and group_key in policy_loss_fn.__code__.co_varnames:
                            loss_kwargs[group_key] = maybe
                    # The labeled batch-slot ablation reads N_B from the
                    # original optimizer batch's slot count; the canonical
                    # tree-segment-mean loss reads N_T from the original
                    # batch's unique-tree count. Both are fixed before the
                    # micro split so the denominators never shrink.
                    if (
                        "original_optimizer_batch_slot_count"
                        in policy_loss_fn.__code__.co_varnames
                    ):
                        loss_kwargs["original_optimizer_batch_slot_count"] = (
                            original_optimizer_batch_slot_count
                        )
                    if (
                        original_optimizer_batch_tree_count is not None
                        and "original_optimizer_batch_tree_count"
                        in policy_loss_fn.__code__.co_varnames
                    ):
                        loss_kwargs["original_optimizer_batch_tree_count"] = (
                            original_optimizer_batch_tree_count
                        )
                    # PLAN.md §1.3: canonical paper aggregations receive the
                    # trainer-stamped pre-filter logical denominators.
                    if (
                        logical_denominators is not None
                        and "original_logical_segment_count"
                        in policy_loss_fn.__code__.co_varnames
                    ):
                        loss_kwargs["original_logical_segment_count"] = (
                            logical_segment_count
                        )
                        loss_kwargs["original_logical_token_count"] = (
                            logical_token_count
                        )
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(**loss_kwargs)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                # PLAN.md M1: each call to _optimizer_step is one internal PPO
                # optimizer-batch update. The trainer accumulates the returned
                # count into the diagnostic num_optimizer_steps_total;
                # global_step still advances by 1 per successful update_actor.
                num_optimizer_steps += 1
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        # PLAN.md M1: expose the true optimizer-step count for this update so
        # the trainer can log the diagnostic num_optimizer_steps_total and
        # optimizer_steps_this_iteration. Stored under the standard verl
        # `metrics` key so it is emitted through `reduce_metrics` unchanged.
        metrics.setdefault("actor/num_optimizer_steps", []).append(int(num_optimizer_steps))
        if vdra_canonical_aggregation:
            # PLAN.md §1.3: all-zero logical batches are skipped consistently
            # on every rank (no forward/backward/optimizer.step) and reported.
            metrics.setdefault("actor/all_zero_logical_batches", []).append(
                int(n_all_zero_logical_batches)
            )
        # PLAN.md P0.J: OBSERVED fact for the manifest — 1.0 only when the
        # stored generation-time old_log_probs were actually kept as the PPO
        # ratio denominator (i.e. this update was not treated as on-policy
        # with old_log_prob overwritten by the current policy's log_prob).
        used_stored = (not on_policy) and ("old_log_probs" in data.batch.keys())
        metrics.setdefault("actor/used_stored_old_log_probs", []).append(
            1.0 if used_stored else 0.0
        )
        return metrics
