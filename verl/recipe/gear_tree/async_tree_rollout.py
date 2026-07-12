"""Async (agent-loop) tree rollout for verl >= 0.7 + vLLM >= 0.20.

verl 0.7+ removed the synchronous SPMD ``vLLMRollout``; generation now goes
through an async OpenAI-compatible server driven by ``AsyncLLMServerManager`` and
the ``experimental.agent_loop`` framework. This module re-targets the tree
rollout onto that stack **without touching the algorithm core**:

  * ``AsyncServerSegmentGenerator`` — an async ``segment_fn`` that fires
    ``branch_factor`` concurrent ``server_manager.generate`` calls per node and
    converts each ``TokenOutput`` into a ``SegmentSample``. This is the only new
    engine-coupled code; it is CPU-testable with a mock server manager.
  * ``TreeAgentLoop`` — an ``AgentLoopBase`` (registered ``gear_tree_agent``)
    whose ``run`` builds the whole SPO/GEAR tree via ``async_build_tree``,
    extracts edges (``tree_advantage``), and stashes them in
    ``AgentLoopOutput.extra_fields["gear_tree_edges"]`` so they land in the
    rollout DataProto's ``non_tensor_batch``. The trainer then flattens the
    per-prompt edges into the training batch (``tree_data.edges_to_dataproto``).

Everything downstream — advantages (``compute_tree_update_values``), GEAR gate,
VinePPO MC values, ``treetune_ppo`` loss, reward grading, logging — is unchanged
and shared with the SPMD path.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from recipe.gear_tree.tree_rollout import SegmentSample, async_build_tree
from recipe.gear_tree.tree_advantage import extract_edges_from_tree
from recipe.gear_tree.gear_core.reward_function import MathRewardFunction


class AsyncServerSegmentGenerator:
    """Async ``segment_fn`` backed by ``AsyncLLMServerManager.generate``."""

    def __init__(
        self,
        server_manager: Any,
        tokenizer: Any,
        *,
        base_sampling_params: Optional[Dict[str, Any]] = None,
        free_max_tokens: int = 1024,
    ) -> None:
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.base_sampling_params = dict(base_sampling_params or {})
        self.free_max_tokens = free_max_tokens

    async def _one(self, prompt_ids: Sequence[int], sp: Dict[str, Any]) -> SegmentSample:
        out = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=list(prompt_ids),
            sampling_params=sp,
        )
        tids = list(out.token_ids)
        logps = list(out.log_probs) if getattr(out, "log_probs", None) else None
        cap = sp.get("max_tokens", self.free_max_tokens)
        # Truncated => expandable. Use the server's stop_reason when present,
        # else fall back to "hit the token cap".
        stop_reason = getattr(out, "stop_reason", None)
        truncated = (stop_reason == "length") or (len(tids) >= cap)
        text = self.tokenizer.decode(tids, skip_special_tokens=True)
        return SegmentSample(
            token_ids=tids,
            text=text,
            finish_reason="length" if truncated else "stop",
            logprobs=logps,
        )

    async def segment_fn(
        self, prompt_token_ids: Sequence[int], branch_factor: int, max_tokens: Optional[int]
    ) -> List[SegmentSample]:
        sp = {
            **self.base_sampling_params,
            "n": 1,
            "logprobs": 1,
            "max_tokens": max_tokens if max_tokens is not None else self.free_max_tokens,
        }
        return await asyncio.gather(
            *[self._one(prompt_token_ids, dict(sp)) for _ in range(int(branch_factor))]
        )


class SegmentNodeExpander:
    """Adapter exposing ``segment_fn`` as the ``node_expander`` interface used
    by ``gear_core.gear.tv_estimators.ConditionalTVEstimator`` (pilot short
    continuations for the VDRA budget-allocation path)."""

    def __init__(self, segment_generator: AsyncServerSegmentGenerator, tokenizer: Any) -> None:
        self.segment_generator = segment_generator
        self.tokenizer = tokenizer

    async def expand(
        self,
        *,
        current_node: Dict[str, Any],
        prefix: str,
        depth: int,
        max_tokens: int,
        branch_factor: int,
    ) -> List[Dict[str, Any]]:
        prompt_ids = current_node.get("full_token_ids")
        if prompt_ids is None:
            prompt_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        samples = await self.segment_generator.segment_fn(
            prompt_ids, int(branch_factor), int(max_tokens)
        )
        nodes: List[Dict[str, Any]] = []
        for s in samples:
            nodes.append(
                {
                    "text": s.text,
                    "full_text": prefix + s.text,
                    "depth": depth + 1,
                    "finish_reason": s.finish_reason,
                    "sum_logprobs": s.sum_logprobs,
                    "num_tokens": s.num_tokens,
                    "full_token_ids": list(prompt_ids) + list(s.token_ids),
                    "response_token_ids": list(s.token_ids),
                    "actor_shifted_log_probs": list(s.logprobs) if s.logprobs is not None else None,
                }
            )
        return nodes


async def build_tree_edges_async(
    prompt_text: str,
    prompt_token_ids: Sequence[int],
    data_instance: Dict[str, Any],
    *,
    segment_generator: AsyncServerSegmentGenerator,
    reward_fn: MathRewardFunction,
    tree_shape: Sequence[int],
    M: int,
    gear_gate: Any = None,
    tree_update_mode: str = "spo",
    adv_method: str = "rloo",
    treepo_global_weight: float = 0.5,
    treerl_gamma: float = 0.9,
    only_adv_greater_than_zero: bool = True,
    vineppo_K: int = 0,
    unfinished_penalty: float = 0.0,
    demo_logger: Any = None,
    gear_node_expander: Any = None,
) -> List[Dict[str, Any]]:
    """Build one tree via the async server and return its SPO/GEAR edges."""

    def grade_fn(query, response, inst):
        return float(reward_fn(query, response, inst)[0])

    tree = await async_build_tree(
        prompt_text, prompt_token_ids, data_instance,
        tree_shape=tree_shape, M=M,
        segment_fn=segment_generator.segment_fn, grade_fn=grade_fn, gear_gate=gear_gate,
        gear_node_expander=gear_node_expander,
    )

    if vineppo_K > 0:
        from recipe.gear_tree.vineppo_advantage import annotate_tree_with_mc_values

        async def _mc(prefix_ids, K):
            return await segment_generator.segment_fn(prefix_ids, K, None)

        # MC values need async rollouts; run them synchronously within this coroutine.
        await _annotate_mc_async(tree, data_instance, _mc, grade_fn, vineppo_K, unfinished_penalty)

    if demo_logger is not None:
        try:
            demo_logger.log_tree(tree, data_instance.get("_treetune__idx"))
        except Exception:
            pass

    return extract_edges_from_tree(
        tree, adv_method=adv_method, only_adv_greater_than_zero=only_adv_greater_than_zero,
        tree_update_mode=tree_update_mode, treepo_global_weight=treepo_global_weight,
        treerl_gamma=treerl_gamma,
    )


async def _annotate_mc_async(tree, data_instance, mc_rollout, grade_fn, K, unfinished_penalty):
    """Async version of vineppo_advantage.annotate_tree_with_mc_values."""
    children = tree.get("children") or []
    for child in children:
        await _annotate_mc_async(child, data_instance, mc_rollout, grade_fn, K, unfinished_penalty)
    if children:
        samples = await mc_rollout(tree.get("full_token_ids", []), K)
        rewards = [
            grade_fn(tree.get("full_text", ""), tree.get("full_text", "") + s.text, data_instance)
            if s.finish_reason != "length" else float(unfinished_penalty)
            for s in samples
        ]
        tree["reward"] = sum(rewards) / len(rewards) if rewards else 0.0


# --------------------------------------------------------------------------- #
# TreeAgentLoop — registered agent for verl's agent-loop framework (GPU path).
# --------------------------------------------------------------------------- #
try:  # keep CPU-importable when agent_loop isn't installed
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register

    @register("gear_tree_agent")  # pragma: no cover - requires async server + GPU
    class TreeAgentLoop(AgentLoopBase):
        """Build a full SPO/GEAR tree per prompt; emit edges via extra_fields."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            gt = dict(self.config.get("gear_tree", {}))
            self._gt = gt
            self._gen = AsyncServerSegmentGenerator(
                self.server_manager, self.tokenizer,
                base_sampling_params={
                    "temperature": self.rollout_config.temperature,
                    "top_p": self.rollout_config.top_p,
                },
                free_max_tokens=self.rollout_config.response_length,
            )
            self._reward_fn = MathRewardFunction(
                answer_prefix=gt.get("answer_prefix", "# Answer\n"),
                use_minerva_few_shot_prompt=gt.get("use_minerva_few_shot_prompt", False),
            )
            self._gate = _build_gate(gt, tokenizer=self.tokenizer)
            self._node_expander = SegmentNodeExpander(self._gen, self.tokenizer)

        async def run(self, sampling_params: dict, **kwargs) -> "AgentLoopOutput":
            messages = list(kwargs["raw_prompt"])
            prompt_ids = await self.apply_chat_template(messages)
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)

            rm = kwargs.get("reward_model", {}) or {}
            extra = kwargs.get("extra_info", {}) or {}
            data_instance = {
                "problem": extra.get("problem") or rm.get("problem"),
                "answer": rm.get("ground_truth"),
                "reward_model": rm,
                "_treetune__idx": kwargs.get("index"),
            }

            edges = await build_tree_edges_async(
                prompt_text, prompt_ids, data_instance,
                segment_generator=self._gen, reward_fn=self._reward_fn,
                tree_shape=self._gt.get("tree_shape", [6, 6, 6]),
                M=self._gt.get("segment_length", 100), gear_gate=self._gate,
                tree_update_mode=self._gt.get("tree_update_mode", "spo"),
                adv_method=self._gt.get("adv_method", "rloo"),
                treepo_global_weight=self._gt.get("treepo_global_weight", 0.5),
                treerl_gamma=self._gt.get("treerl_gamma", 0.9),
                only_adv_greater_than_zero=self._gt.get("only_adv_greater_than_zero", True),
                vineppo_K=self._gt.get("vineppo_K", 0),
                gear_node_expander=self._node_expander,
            )

            # Placeholder response (first edge) keeps the AgentLoopOutput schema
            # valid; the real training rows are the edges in extra_fields.
            resp = (edges[0].get("response_token_ids") if edges else [self.tokenizer.eos_token_id]) or [
                self.tokenizer.eos_token_id
            ]
            resp = resp[: self.rollout_config.response_length]
            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=resp,
                response_mask=[1] * len(resp),
                num_turns=2,
                metrics=AgentLoopMetrics(),
                extra_fields={"gear_tree_edges": edges},
            )

    def _build_gate(gt: dict, tokenizer: Any = None):
        if not gt.get("gear", {}).get("enabled", False):
            return None
        from recipe.gear_tree.gear_gate import GearGate

        g = gt["gear"]
        scorer = _build_scorer(g, tokenizer)
        return GearGate(
            epsilon=g.get("epsilon", 0.02), r_max=g.get("r_max", 1.0), gamma=g.get("gamma", 0.9),
            alpha=g.get("alpha", 0.05), k_algorithm=g.get("k_algorithm", "budget_allocation"),
            n_min=g.get("n_min", 1), pilot_branch_factor=g.get("pilot_branch_factor", None), likelihood_samples_per_distribution=g.get("likelihood_samples_per_distribution", 2), root_allocation=g.get("root_allocation", True),
            skip_near_leaf_expand=g.get("skip_near_leaf_expand", True),
            max_depth=len(gt.get("tree_shape", [])) or None, enable_share=g.get("enable_share", False),
            scorer=scorer,
            eps_tail=g.get("eps_tail", 0.0),
            eps_tail_by_depth=g.get("eps_tail_by_depth", None),
            bound_form=g.get("bound_form", "linear"),
            tv_estimator=g.get("tv_estimator", "tanh"),
            tv_first_phase_tokens=g.get("tv_first_phase_tokens", 120),
            tv_second_phase_tokens=g.get("tv_second_phase_tokens", 60),
            queue_count=g.get("queue_count", 1), queue_capacity=g.get("queue_capacity", 8),
            queue_timeout_seconds=g.get("queue_timeout_seconds", 0.0),
            use_residual_budget=g.get("use_residual_budget", True), strict_vdra=g.get("strict_vdra", True), invalid_support_policy=g.get("invalid_support_policy", "error"), budget_mode=g.get("budget_mode", "fixed_main"),
        )

    def _build_scorer(g: dict, tokenizer: Any):
        """Build the log-prob scorer for share / budget-allocation paths.

        ``scorer_api_base`` points at an OpenAI-compatible vLLM server (the
        agent-loop stack already runs one); without it the gate runs with
        """
        api_base = g.get("scorer_api_base")
        if not api_base or tokenizer is None:
            return None
        from recipe.gear_tree.gear_core.gear.vllm_scorer import (
            VLLMLogprobClient,
            make_lp_scorer,
        )

        client = VLLMLogprobClient(
            api_base=str(api_base),
            model=str(g.get("scorer_model", "")),
            max_concurrency=int(g.get("scorer_max_concurrency", 64)),
        )
        return make_lp_scorer(
            client, lambda text: tokenizer.encode(text, add_special_tokens=False)
        )

except Exception:  # pragma: no cover
    TreeAgentLoop = None  # type: ignore

    def _build_gate(gt, tokenizer=None):  # type: ignore
        return None


def collect_tree_edges(dataproto) -> List[Dict[str, Any]]:
    """Flatten per-prompt ``gear_tree_edges`` (non_tensor_batch) into one list."""
    edges: List[Dict[str, Any]] = []
    per_prompt = dataproto.non_tensor_batch.get("gear_tree_edges")
    if per_prompt is None:
        raise KeyError("rollout output has no 'gear_tree_edges' (is agent.name=gear_tree_agent?)")
    for item in per_prompt:
        if item:
            edges.extend(item)
    return edges
