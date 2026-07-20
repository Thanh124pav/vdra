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


def _build_gate(gt: dict, scorer=None):
    if not gt.get("gear", {}).get("enabled", False):
        return None
    from recipe.gear_tree.gear_gate import GearGate

    from recipe.gear_tree.calibration import resolve_gear_calibration

    g = resolve_gear_calibration(dict(gt["gear"]))
    # Defaults here must match the GearGate signature so a missing config key
    # behaves identically no matter which entry point built the gate.
    return GearGate(
        epsilon=g.get("epsilon", 0.02),
        r_max=g.get("r_max", 1.0),
        gamma=g.get("gamma", 0.9),
        alpha=g.get("alpha", 0.05),
        k_algorithm=g.get("k_algorithm", "budget_allocation"),
        n_min=g.get("n_min", 1), pilot_branch_factor=g.get("pilot_branch_factor", None), likelihood_samples_per_distribution=g.get("likelihood_samples_per_distribution", 2),
        root_allocation=g.get("root_allocation", False),
        skip_near_leaf_expand=g.get("skip_near_leaf_expand", True),
        max_depth=len(gt.get("tree_shape", [])) or None,
        enable_share=g.get("enable_share", False),
        scorer=scorer,
        eps_tail=g.get("eps_tail", 0.0),
        eps_tail_by_depth=g.get("eps_tail_by_depth", None),
        bound_form=g.get("bound_form", "linear"),
        tv_estimator=g.get("tv_estimator", "tanh"),
        tv_first_phase_tokens=g.get("tv_first_phase_tokens", 60),
        tv_second_phase_tokens=g.get("tv_second_phase_tokens", 60),
        queue_count=g.get("queue_count", 4), queue_capacity=g.get("queue_capacity", 8),
        queue_timeout_seconds=g.get("queue_timeout_seconds", 1.0),
        use_residual_budget=g.get("use_residual_budget", True), strict_vdra=g.get("strict_vdra", True), invalid_support_policy=g.get("invalid_support_policy", "error"), budget_mode=g.get("budget_mode", "fixed_main"),
        allocation_proxy=g.get("allocation_proxy", "vdra"),
        allocation_runtime=g.get("allocation_runtime", "online_timeout"),
        artifact_dir=g.get("artifact_dir"),
        eps_tail_calibration_path=g.get("eps_tail_source"),
        eps_tail_calibration_metadata=g.get("eps_tail_calibration_metadata"),
        pilot_execution_mode=g.get("pilot_execution_mode", "fresh_iid"),
        weighted_reuse_fallback=g.get("weighted_reuse_fallback", "fresh_iid"),
        representative_weight_mode=g.get("representative_weight_mode", "cluster_multiplicity"),
        terminal_pilot_handling=g.get("terminal_pilot_handling", "include_in_dispersion"),
        rollout_temperature=g.get("rollout_temperature", 1.0),
        rollout_top_p=g.get("rollout_top_p", 1.0),
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

        scorer = None
        if gt.get("gear", {}).get("enabled", False) and (
            gt.get("gear", {}).get("k_algorithm", "budget_allocation") == "budget_allocation"
            or gt.get("gear", {}).get("enable_share", False)
        ):
            from recipe.gear_tree.engine_scorer import EngineLPScorer

            scorer = EngineLPScorer(self.rollout.inference_engine, self.tokenizer)
        gate = _build_gate(gt, scorer=scorer)
        # The SPMD path drives the synchronous build_tree, which has no
        # online/batch allocation branch: running VDRA there would silently
        # degrade to uniform SPO expansion. Fail loudly instead.
        if gate is not None and gate.k_algorithm == "budget_allocation":
            raise ValueError(
                "VDRA budget_allocation is not supported on the SPMD rollout "
                "backend (sync build_tree has no allocation path). Set "
                "gear_tree.rollout_backend: async, or use k_algorithm: simple."
            )

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
            # PLAN.md P0.G: canonical default is DENSE (keep zero-advantage
            # rows) — must match the main config and every other call site.
            only_adv_greater_than_zero=gt.get("only_adv_greater_than_zero", False),
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
