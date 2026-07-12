"""Actor/rollout worker exposing the dispatched ``build_trees`` method.

Subclasses verl's FSDP ``ActorRolloutRefWorker`` and adds one dispatched method,
``build_trees``, mirroring the mode-switch + sharding wrapper of
``generate_sequences`` (fsdp_workers.py) but delegating to the segment-tree
rollout. The ``gear_tree`` config block lives at the top level of the run config
(not in the worker's ``actor_rollout_ref`` slice), so the trainer passes it via
``prompts.meta_info["gear_tree_config"]`` and the rollout is configured lazily on
the first call.

GPU/Ray-only. Imported lazily by ``main_gear_tree`` so CPU imports don't require
vLLM/FSDP.
"""

from __future__ import annotations

import asyncio

from verl import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_id, get_torch_device
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def _build_gate(gt: dict):
    if not gt.get("gear", {}).get("enabled", False):
        return None
    from recipe.gear_tree.gear_gate import GearGate

    g = gt["gear"]
    return GearGate(
        epsilon=g.get("epsilon", 0.02),
        r_max=g.get("r_max", 1.0),
        gamma=g.get("gamma", 0.9),
        alpha=g.get("alpha", 0.05),
        k_algorithm=g.get("k_algorithm", "simple"),
        n_min=g.get("n_min", 0),
        budget_lambda=g.get("budget_lambda", 0.0),
        n_tv_estimates=g.get("n_tv_estimates", None),
        root_allocation=g.get("root_allocation", True),
        skip_near_leaf_expand=g.get("skip_near_leaf_expand", True),
        max_depth=len(gt.get("tree_shape", [])) or None,
        enable_share=g.get("enable_share", False),
        eps_tail=g.get("eps_tail", 0.0),
        eps_tail_by_depth=g.get("eps_tail_by_depth", None),
        bound_form=g.get("bound_form", "linear"),
        tv_estimator=g.get("tv_estimator", "tanh"),
        tv_first_phase_tokens=g.get("tv_first_phase_tokens", 120),
        tv_second_phase_tokens=g.get("tv_second_phase_tokens", 60),
        queue_count=g.get("queue_count", 1),
        queue_timeout_seconds=g.get("queue_timeout_seconds", 0.0),
        use_residual_budget=g.get("use_residual_budget", True),
    )


class GearTreeActorRolloutWorker(ActorRolloutRefWorker):  # pragma: no cover - GPU only
    def _build_rollout(self, trust_remote_code=False):
        rollout, sharding_manager = super()._build_rollout(trust_remote_code=trust_remote_code)
        from recipe.gear_tree.vllm_rollout_tree import vLLMTreeRollout

        if vLLMTreeRollout is not None and type(rollout).__name__ == "vLLMRollout":
            rollout.__class__ = vLLMTreeRollout
        self._tree_configured = False
        return rollout, sharding_manager

    def _configure_tree_rollout(self, gt: dict):
        from recipe.gear_tree.gear_core.reward_function import MathRewardFunction
        from recipe.gear_tree.tree_logging import TreeDemoLogger

        gate = _build_gate(gt)
        if gate is not None and gate.enable_share:
            from recipe.gear_tree.engine_scorer import EngineLPScorer

            gate.scorer = EngineLPScorer(self.rollout.inference_engine, self.tokenizer)

        reward_fn = MathRewardFunction(
            answer_prefix=gt.get("answer_prefix", "# Answer\n"),
            use_minerva_few_shot_prompt=gt.get("use_minerva_few_shot_prompt", False),
        )
        demo_logger = TreeDemoLogger(
            gt.get("demos_dir"),
            demo_examples_per_tree=gt.get("demo_examples_per_tree", 4),
            full_tree_every_n_trees=gt.get("full_tree_every_n_trees", 25),
            full_tree_max_trees=gt.get("full_tree_max_trees", 5),
        )
        self.rollout.set_tree_config(
            tokenizer=self.tokenizer,
            tree_shape=gt.get("tree_shape", [6, 6, 6]),
            M=gt.get("segment_length", 100),
            gear_gate=gate,
            reward_fn=reward_fn,
            tree_update_mode=gt.get("tree_update_mode", "spo"),
            adv_method=gt.get("adv_method", "rloo"),
            treepo_global_weight=gt.get("treepo_global_weight", 0.5),
            treerl_gamma=gt.get("treerl_gamma", 0.9),
            only_adv_greater_than_zero=gt.get("only_adv_greater_than_zero", True),
            vineppo_K=gt.get("vineppo_K", 0),
            unfinished_penalty=gt.get("unfinished_penalty", 0.0),
            demo_logger=demo_logger,
        )
        self._tree_configured = True

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def build_trees(self, prompts: DataProto):
        assert self._is_rollout
        if not getattr(self, "_tree_configured", False):
            self._configure_tree_rollout(dict(prompts.meta_info.get("gear_tree_config", {})))

        prompts = prompts.to(get_device_id())
        prompts.meta_info.update(
            {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id}
        )

        loop = None
        if self._is_actor:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(self.rollout_mode())

        output = self.rollout.build_trees(prompts=prompts)

        if self._is_actor and loop is not None:
            loop.run_until_complete(self.trainer_mode())

        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output
