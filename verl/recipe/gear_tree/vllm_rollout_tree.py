"""Tree/segment rollout on verl's synchronous vLLM SPMD engine (Step 2 binding).

Mirrors TreePO's approach of extending
``verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py``: ``vLLMTreeRollout``
subclasses ``vLLMRollout`` and adds ``build_trees`` — for every prompt it drives
``build_tree`` segment-by-segment against ``self.inference_engine``, grades leaves
with the vendored ``MathRewardFunction``, extracts SPO/GEAR edges, and packs them
into one flat ``DataProto`` (one row per edge) via ``edges_to_dataproto``.

The heavy lifting lives in ``build_edge_batch`` — a plain function that takes an
``inference_engine`` + ``tokenizer``, so it is unit-testable with a fake engine.
Advantages are computed here (in generation), matching treetune; the downstream
verl actor update consumes ``batch["advantages"]`` directly.

NOTE: the ``build_trees`` worker path requires a GPU + a live vLLM engine and has
to be validated end-to-end there. The pure tree/advantage/data-assembly core it
calls is covered by CPU tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from recipe.gear_tree.gear_core.reward_function import MathRewardFunction
from recipe.gear_tree.tree_rollout import VLLMTreeRollout, build_tree
from recipe.gear_tree.tree_advantage import extract_edges_from_tree
from recipe.gear_tree.tree_data import edges_to_dataproto


def _strip_left_pad(token_ids: Sequence[int], pad_id: int) -> List[int]:
    ids = list(token_ids)
    i = 0
    while i < len(ids) and ids[i] == pad_id:
        i += 1
    return ids[i:]


def build_edge_batch(
    prompts,
    *,
    inference_engine,
    tokenizer,
    tree_shape: Sequence[int],
    M: int,
    reward_fn: Optional[MathRewardFunction] = None,
    gear_gate: Any = None,
    max_prompt_length: int,
    max_response_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: Optional[int] = None,
    tree_update_mode: str = "spo",
    adv_method: str = "rloo",
    treepo_global_weight: float = 0.5,
    treerl_gamma: float = 0.9,
    only_adv_greater_than_zero: bool = False,
    vineppo_K: int = 0,
    unfinished_penalty: float = 0.0,
    demo_logger: Any = None,
    strict_fresh_iid: bool = False,
    loss_mode: str = "vdra_segment_mean_ppo",
):
    """Build the flat edge-batch ``DataProto`` for a batch of prompts.

    ``prompts`` is a verl ``DataProto`` with ``input_ids`` (left-padded prompts)
    and ``non_tensor_batch`` carrying ``reward_model``/``extra_info``. Returns a
    new ``DataProto`` whose rows are tree edges with precomputed advantages.
    """
    reward_fn = reward_fn or MathRewardFunction()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    binder = VLLMTreeRollout(
        inference_engine=inference_engine,
        tokenizer=tokenizer,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        free_max_tokens=max_response_length,
        logprobs=1,
        seed=seed,
    )

    input_ids = prompts.batch["input_ids"]
    ntb = prompts.non_tensor_batch
    all_edges: List[Dict[str, Any]] = []

    for row in range(input_ids.shape[0]):
        prompt_ids = _strip_left_pad(input_ids[row].tolist(), pad_id)
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        reward_model = ntb.get("reward_model", [{}] * input_ids.shape[0])[row] or {}
        extra_info = ntb.get("extra_info", [{}] * input_ids.shape[0])[row] or {}
        data_instance = {
            "problem": extra_info.get("problem") or reward_model.get("problem"),
            "answer": reward_model.get("ground_truth"),
            "reward_model": reward_model,
            "_treetune__idx": row,
        }

        def grade_fn(query, response, inst):
            reward, _ = reward_fn(query, response, inst)
            return float(reward)

        tree = build_tree(
            prompt_text,
            prompt_ids,
            data_instance,
            tree_shape=tree_shape,
            M=M,
            segment_fn=binder.segment_fn,
            grade_fn=grade_fn,
            gear_gate=gear_gate,
        )

        # GEAR-VinePPO: replace internal-node values with MC estimates from K
        # independent rollouts per cut-point, so the SPO edge advantage becomes
        # the VinePPO TD residual value(child)-value(parent).
        if vineppo_K > 0:
            from recipe.gear_tree.vineppo_advantage import annotate_tree_with_mc_values

            def rollout_fn(prefix_ids, K):
                return binder.segment_fn(prefix_ids, K, None)

            annotate_tree_with_mc_values(
                tree,
                data_instance,
                rollout_fn=rollout_fn,
                grade_fn=grade_fn,
                K=vineppo_K,
                unfinished_penalty=unfinished_penalty,
            )

        # Log tree stats + one full-tree example (offline, treetune-style).
        if demo_logger is not None:
            try:
                demo_logger.log_tree(tree, data_instance.get("_treetune__idx", row))
            except Exception:
                pass

        # Stage 1: extract after computing advantages for realized children.
        # The legacy flag removes only exact-zero advantages; positive and
        # negative advantages remain trainable. Pruned placeholders stay out.
        edges = extract_edges_from_tree(
            tree,
            adv_method=adv_method,
            only_adv_greater_than_zero=only_adv_greater_than_zero,
            tree_update_mode=tree_update_mode,
            treepo_global_weight=treepo_global_weight,
            treerl_gamma=treerl_gamma,
            emit_pruned_edges=False,
            strict_fresh_iid=strict_fresh_iid,
        )
        all_edges.extend(edges)

    if not all_edges:
        raise RuntimeError("tree rollout produced no training edges for this batch")

    # PLAN.md P0.C: the configured loss mode decides whether float objective
    # weights are attached (node-balanced ablation only).
    return edges_to_dataproto(
        all_edges,
        tokenizer,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        loss_mode=loss_mode,
    )


# --------------------------------------------------------------------------- #
# vLLM rollout subclass (GPU worker path).
# --------------------------------------------------------------------------- #
try:  # keep the module importable on CPU (vllm not installed)
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

    class vLLMTreeRollout(vLLMRollout):  # pragma: no cover - requires GPU/vLLM
        """``vLLMRollout`` + segment-tree ``build_trees``."""

        def set_tree_config(self, *, tokenizer, tree_shape, M, gear_gate=None, reward_fn=None, **kw):
            self._tree_tokenizer = tokenizer
            self._tree_shape = list(tree_shape)
            self._tree_M = int(M)
            self._tree_gate = gear_gate
            self._tree_reward_fn = reward_fn or MathRewardFunction()
            self._tree_kw = kw
            return self

        def build_trees(self, prompts, **kwargs):
            return build_edge_batch(
                prompts,
                inference_engine=self.inference_engine,
                tokenizer=self._tree_tokenizer,
                tree_shape=self._tree_shape,
                M=self._tree_M,
                reward_fn=self._tree_reward_fn,
                gear_gate=self._tree_gate,
                max_prompt_length=self.config.prompt_length,
                max_response_length=self.config.response_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.get("top_k", -1),
                **self._tree_kw,
            )

except Exception:  # pragma: no cover
    vLLMTreeRollout = None  # type: ignore
