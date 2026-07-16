"""GEAR inference strategy: SPO-tree with budget allocation and local TV gates.

The current production path is `budget_allocation`: TV probes estimate reward
variance for frontier nodes and branch budget is assigned across those nodes.
The legacy reference-solution generation path has been removed from this strategy;
any SHARE/PRUNE mode that remains here uses only sibling-local TV comparisons.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import math
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.inference_strategies.hybrid_inference_strategy import (
    HybridInferenceStrategy,
)
from treetune.inference_strategies.tree_inference import Node
from treetune.logging_utils import get_logger

from treetune.gear.logging_helpers import aggregate_tree_stats
from treetune.gear.thresholds import ThresholdConfig
from treetune.gear.budget_allocation import allocate_branch_factors
from treetune.gear.budget_scheduler import FlexibleBudgetScheduler
from treetune.gear.online_budget import (
    OnlineQueueItem,
    RootQueueManager,
    SharedReservePool,
)
from treetune.gear.tv_estimators import ConditionalTVEstimator, select_reuse_candidates
from treetune.gear.triggers import Action
from treetune.gear.vllm_scorer import VLLMLogprobClient, make_lp_scorer
from vdra_core.logging_schema import validate_node_accounting, write_node_accounting

logger = get_logger(__name__)


def _finalize_empty_expansion_node(
    node: Node,
    *,
    initial_prompt: str,
    data_instance: Optional[Dict[str, Any]],
    reward_function,
) -> None:
    """Turn a final-depth node into a scored leaf when generation is empty."""

    node["children"] = []
    node["reward"], _ = reward_function(
        query=initial_prompt,
        response=node.get("full_text", node.get("text", "")),
        dataset_instance=data_instance,
    )
    node["reward_std"] = 0.0
    node["leaf"] = True


@InferenceStrategy.register("gear", exist_ok=True)
class GEARInferenceStrategy(HybridInferenceStrategy):
    def __init__(
        self,
        # GEAR-specific knobs ------------------------------------------------
        gear_epsilon: float = 0.02,
        gear_r_max: float = 1.0,
        gear_gamma: float = 0.9,
        gear_score_concurrency: int = 64,
        gear_score_timeout_seconds: float = 120.0,
        gear_score_retry_attempts: int = 3,
        gear_score_retry_backoff_seconds: float = 0.5,
        gear_k_algorithm: str = "hierarchical",
        gear_generation_mode: str = "single_request",
        gear_pilot_branch_factor: int = 8,
        gear_likelihood_samples_per_distribution: int = 2,
        gear_tv_subnode_max_tokens: int = 60,
        gear_tv_second_phase_tokens: int = 60,
        gear_tv_includes_half_factor: bool = True,
        gear_budget_queue_capacity: int = 8,
        gear_strict_vdra: bool = True,
        gear_invalid_support_policy: str = "error",
        gear_budget_mode: str = "fixed_main",
        gear_allocation_proxy: str = "vdra",
        gear_rounding_strategy: str = "largest_remainder",
        gear_rounding_seed: int = 0,
        gear_n_min: int = 1,
        gear_budget_overhead_mode: str = "flexible",
        gear_budget_queue_count: int = 4,
        gear_budget_queue_timeout_seconds: float = 1.0,
        gear_skip_near_leaf_expand: bool = True,
        gear_root_allocation: bool = True,
        gear_use_residual_budget: bool = True,
        gear_allocation_mode: str = "budget_allocation",
        # VDRA scoring / bound knobs ------------------------------------------
        gear_eps_tail: float = 0.0,
        gear_eps_tail_calibration_path: Optional[str] = None,
        gear_eps_tail_by_depth: Optional[Dict[int, float]] = None,
        gear_bound_form: str = "linear",
        gear_tv_estimator: str = "tanh",
        # Inherited ----------------------------------------------------------
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if gear_strict_vdra:
            if not gear_eps_tail_calibration_path:
                raise ValueError("strict VDRA requires gear_eps_tail_calibration_path")
            from vdra_core.calibration import load_tail_calibration
            calibration = load_tail_calibration(
                gear_eps_tail_calibration_path,
                pilot_branch_factor=gear_pilot_branch_factor,
                likelihood_samples_per_distribution=gear_likelihood_samples_per_distribution,
                short_horizon=gear_tv_second_phase_tokens,
                strict_metadata=gear_strict_vdra,
            )
            gear_eps_tail = calibration["eps_tail"]
            gear_eps_tail_by_depth = calibration["eps_tail_by_depth"]
        self.cfg_thresholds = ThresholdConfig(
            epsilon=gear_epsilon,
            r_max=gear_r_max,
            gamma=gear_gamma,
            K=max(int(gear_pilot_branch_factor * gear_likelihood_samples_per_distribution), 1),
            eps_tail=float(gear_eps_tail),
            eps_tail_by_depth=(
                {int(k): float(v) for k, v in gear_eps_tail_by_depth.items()}
                if gear_eps_tail_by_depth
                else None
            ),
            bound_form=gear_bound_form,
        )
        self.gear_tv_estimator = gear_tv_estimator
        self.gear_score_concurrency = int(gear_score_concurrency)
        self.gear_score_timeout_seconds = float(gear_score_timeout_seconds)
        self.gear_score_retry_attempts = int(gear_score_retry_attempts)
        self.gear_score_retry_backoff_seconds = float(
            gear_score_retry_backoff_seconds
        )
        if gear_budget_overhead_mode not in {"flexible", "none"}:
            raise ValueError(
                f"Unsupported gear_budget_overhead_mode: {gear_budget_overhead_mode}"
            )
        if gear_k_algorithm not in {"hierarchical", "perplexity", "simple", "entropy_guided"}:
            raise ValueError(f"Unsupported gear_k_algorithm: {gear_k_algorithm}")
        if gear_generation_mode not in {"single_request", "replicated_requests"}:
            raise ValueError(f"Unsupported gear_generation_mode: {gear_generation_mode}")
        if gear_allocation_mode not in {"budget_allocation", "prune_only"}:
            raise ValueError(f"Unsupported gear_allocation_mode: {gear_allocation_mode}")
        self.gear_algorithm_mode = "budget_allocation"
        self.gear_k_algorithm = gear_k_algorithm
        self.gear_generation_mode = gear_generation_mode
        self.gear_pilot_branch_factor = max(int(gear_pilot_branch_factor), 2)
        self.gear_likelihood_samples_per_distribution = max(
            int(gear_likelihood_samples_per_distribution), 1
        )
        self.gear_tv_subnode_max_tokens = int(gear_tv_subnode_max_tokens)
        self.gear_tv_second_phase_tokens = int(gear_tv_second_phase_tokens)
        self.gear_tv_includes_half_factor = bool(gear_tv_includes_half_factor)
        self.gear_n_min = max(int(gear_n_min), 0)
        self.gear_budget_overhead_mode = gear_budget_overhead_mode
        self.gear_budget_queue_count = max(int(gear_budget_queue_count), 1)
        self.gear_budget_queue_capacity = max(int(gear_budget_queue_capacity), 1)
        self.gear_strict_vdra = bool(gear_strict_vdra)
        self.gear_invalid_support_policy = str(gear_invalid_support_policy)
        if gear_budget_mode not in {"fixed_main", "fixed_total_generated"}:
            raise ValueError(f"Unsupported gear_budget_mode: {gear_budget_mode}")
        if gear_budget_mode == "fixed_total_generated":
            # The shared generated-token cap is enforced only by the verl
            # online builder (async_build_tree_online_alloc); accepting the
            # mode here would silently run fixed_main under a false label.
            raise NotImplementedError(
                "gear_budget_mode='fixed_total_generated' is implemented on "
                "the verl online runtime only (recipe.gear_tree)."
            )
        self.gear_budget_mode = gear_budget_mode
        if gear_allocation_proxy not in {
            "vdra", "uniform", "random", "direct_tv", "empirical_variance",
            "external_score", "oracle",
        }:
            raise ValueError(f"Unsupported gear_allocation_proxy: {gear_allocation_proxy}")
        self.gear_allocation_proxy = gear_allocation_proxy
        if gear_rounding_strategy not in {"largest_remainder", "nearest_repair", "stochastic"}:
            raise ValueError(f"Unsupported gear_rounding_strategy: {gear_rounding_strategy}")
        self.gear_rounding_strategy = gear_rounding_strategy
        self.gear_rounding_seed = int(gear_rounding_seed)
        self.gear_budget_queue_timeout_seconds = float(
            gear_budget_queue_timeout_seconds
        )
        self.gear_skip_near_leaf_expand = bool(gear_skip_near_leaf_expand)
        self.gear_root_allocation = bool(gear_root_allocation)
        self.gear_use_residual_budget = bool(gear_use_residual_budget)
        self.gear_allocation_mode = gear_allocation_mode
        self._lp_client: Optional[VLLMLogprobClient] = None
        self._validate_gear_generation_contract()


    def _tv_mode_for_k_algorithm(self) -> str:
        if self.gear_k_algorithm == "perplexity":
            return "perplexity"
        if self.gear_k_algorithm in {"hierarchical", "entropy_guided"}:
            return "hierarchical"
        return "subnode"

    # ------------------------------------------------------------------
    # GEAR contract checks
    # ------------------------------------------------------------------

    def _validate_gear_generation_contract(self) -> None:
        expand = getattr(self.node_expander, "expand", None)
        if expand is None:
            raise RuntimeError("GEAR requires a node_expander with an expand method")
        signature = inspect.signature(expand)
        has_branch_factor = "branch_factor" in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if not has_branch_factor:
            raise RuntimeError(
                "GEAR requires a node_expander whose expand(...) accepts "
                "branch_factor overrides. Use an efficient_iid-style expander."
            )
        program_kwargs = getattr(self.node_expander, "program_kwargs", None)
        if not isinstance(program_kwargs, dict) or program_kwargs.get("logprobs") != 1:
            raise RuntimeError(
                "GEAR requires node_expander.program_kwargs.logprobs: 1 so "
                "every generated node has sum_logprobs and num_tokens."
            )

    @staticmethod
    def _require_generated_logprobs(node: Node, *, context: str) -> None:
        if node.get("sum_logprobs") is None or node.get("num_tokens") is None:
            raise RuntimeError(
                "GEAR requires every generated non-root node to include "
                f"sum_logprobs and num_tokens; missing at {context}."
            )

    # ------------------------------------------------------------------
    # vLLM scoring client
    # ------------------------------------------------------------------

    def _ensure_lp_client(self) -> VLLMLogprobClient:
        if self._lp_client is not None:
            return self._lp_client
        guidance_kwargs = self.guidance_llm_lazy._params  # type: ignore[attr-defined]
        api_base = guidance_kwargs.get("api_base")
        if api_base in (None, "none"):
            api_base = os.environ.get("APP_OPENAI_VLLM_API_BASE")
        if api_base is None:
            raise RuntimeError("GEAR needs api_base to call vLLM /completions")
        model = guidance_kwargs.get("model")
        api_key = guidance_kwargs.get("api_key", "EMPTY")
        self._lp_client = VLLMLogprobClient(
            api_base=api_base,
            model=model,
            api_key=api_key,
            timeout=self.gear_score_timeout_seconds,
            max_concurrency=self.gear_score_concurrency,
            retry_attempts=self.gear_score_retry_attempts,
            retry_backoff_seconds=self.gear_score_retry_backoff_seconds,
        )
        return self._lp_client

    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("GEAR requires a tokenizer to count suffix tokens")
        return self.tokenizer(text).input_ids

    async def _prepare_tree_construction_context(
        self,
        dataset,
        question_format_keys: Sequence[str],
    ) -> Dict[str, Any]:
        shared_context = {
            "shared_reserve_pool": SharedReservePool(
                queue_count=self.gear_budget_queue_count
            )
        }
        if (
            not self.gear_root_allocation
            or self.gear_allocation_mode == "prune_only"
            or len(dataset) == 0
        ):
            return shared_context

        client = self._ensure_lp_client()
        scorer = make_lp_scorer(client, self._tokenize)
        tv_estimator = ConditionalTVEstimator(
            scorer=scorer,
            node_expander=self.node_expander,
            gamma=self.cfg_thresholds.gamma,
            mode=self._tv_mode_for_k_algorithm(),
            pilot_branch_factor=self.gear_pilot_branch_factor,
            likelihood_samples_per_distribution=self.gear_likelihood_samples_per_distribution,
            invalid_support_policy=getattr(self, "gear_invalid_support_policy", "error"),
            strict_vdra=getattr(self, "gear_strict_vdra", True),
            first_phase_tokens=self.gear_tv_subnode_max_tokens,
            second_phase_tokens=self.gear_tv_second_phase_tokens,
            tv_includes_half_factor=self.gear_tv_includes_half_factor,
            tv_estimator=getattr(self, "gear_tv_estimator", "tanh"),
            r_max=self.cfg_thresholds.r_max,
            eps_tail=self.cfg_thresholds.eps_tail,
            bound_form=self.cfg_thresholds.bound_form,
        )

        root_nodes: List[Node] = []
        root_ids: List[Any] = []
        for data_instance in dataset:
            instance_idx = data_instance["_treetune__idx"]
            format_kwargs = {key: data_instance[key] for key in question_format_keys}
            initial_prompt = self.question_template.format(**format_kwargs)
            root_ids.append(instance_idx)
            root_nodes.append(
                {
                    "text": initial_prompt,
                    "depth": 0,
                    "full_text": initial_prompt,
                    "stop_text": "aaa",
                    "_request_object": data_instance,
                    "leaf": False,
                    "gear_action": Action.EXPAND.value,
                    "gear_segment_id": str(instance_idx),
                    "gear_algorithm_mode": "budget_allocation",
                    "gear_k_algorithm": self.gear_k_algorithm,
                    "gear_allocation_mode": self.gear_allocation_mode,
                    "gear_use_residual_budget": self.gear_use_residual_budget,
                    "gear_n_min": self.gear_n_min,
                    "M": self.M,
                }
            )

        try:
            base_branch_factor = int(
                self.node_expander.branch_factor_strategy({"depth": 0})
            )
        except Exception as exc:
            if getattr(self, "gear_strict_vdra", True):
                raise RuntimeError("VDRA root branch-factor strategy failed") from exc
            base_branch_factor = 1
        total_root_budget = base_branch_factor * len(root_nodes)

        t_var = time.time()
        estimate_results = await asyncio.gather(
            *(tv_estimator.estimate_k_for_parent(node, depth=0, duplicate_tv_threshold=self.cfg_thresholds.epsilon) for node in root_nodes)
        )
        variance_seconds = time.time() - t_var

        for node, result in zip(root_nodes, estimate_results):
            from vdra_core.proxies import select_dispersion_proxy
            node["vdra_dispersion_C"] = select_dispersion_proxy(
                getattr(self, "gear_allocation_proxy", "vdra"),
                vdra_dispersion_C=getattr(result, "dispersion_C", getattr(result, "reward_variance", 0.0)),
                pair_tvs=result.pair_tvs,
                pilot_count=len(getattr(result, "candidates", getattr(result, "unique_candidates", []))),
                node=node,
            )
            node["vdra_default_k"] = base_branch_factor
            node["vdra_predicted_k"] = max(result.predicted_k, self.gear_n_min)

        t_alloc = time.time()
        if self.gear_budget_overhead_mode == "flexible":
            scheduler = FlexibleBudgetScheduler(
                queue_count=self.gear_budget_queue_count,
                n_min=self.gear_n_min,
            )
            summaries = scheduler.allocate(
                root_nodes, total_depth_budget=total_root_budget
            )
            allocations: Dict[str, int] = {}
            weights: Dict[str, float] = {}
            for summary in summaries:
                allocations.update(summary.allocations)
                weights.update(summary.weights)
        else:
            summary = allocate_branch_factors(
                root_nodes,
                total_budget=total_root_budget,
                n_min=self.gear_n_min,
            )
            allocations = summary.allocations
            weights = summary.weights
        allocation_seconds = time.time() - t_alloc
        allocated_root_budget = sum(int(value) for value in allocations.values())
        logger.info(
            "GEAR root_allocation allocated depth-0 budget across %d roots: requested=%d allocated=%d",
            len(root_nodes),
            total_root_budget,
            allocated_root_budget,
        )

        root_allocations: Dict[Any, Dict[str, Any]] = {}
        per_root_variance_seconds = variance_seconds / max(len(root_nodes), 1)
        per_root_allocation_seconds = allocation_seconds / max(len(root_nodes), 1)
        for instance_idx, node, result in zip(root_ids, root_nodes, estimate_results):
            node_id = str(instance_idx)
            root_allocations[instance_idx] = {
                "allocated_branch_factor": int(allocations.get(node_id, 0)),
                "budget_weight": float(weights.get(node_id, 0.0)),
                "dispersion_C": float(result.dispersion_C),
                "tv_pair_count": len(result.pair_tvs),
                "tv_support_size": len(result.samples),
                # Post-pruning survivors only; reuse selection among them is a
                # seeded uniform draw at expansion time.
                "budget_candidates": list(result.unique_candidates),
                "shortcut_candidates": list(getattr(result, "shortcut_candidates", [])),
                "tv_logp_matrix": result.logp_matrix,
                "variance_seconds": per_root_variance_seconds,
                "allocation_seconds": per_root_allocation_seconds,
                "total_root_budget": total_root_budget,
                "allocated_root_budget": allocated_root_budget,
            }

        shared_context["root_allocations"] = root_allocations
        return shared_context

    def _get_tree_construction_kwargs(
        self,
        tree_construction_context: Dict[str, Any],
        instance_idx,
        data_instance: Dict[str, Any],
        initial_prompt: str,
    ) -> Dict[str, Any]:
        root_allocations = tree_construction_context.get("root_allocations") or {}
        root_allocation_info = root_allocations.get(instance_idx)
        out = {}
        if root_allocation_info is not None:
            out["root_allocation_info"] = root_allocation_info
        shared_reserve_pool = tree_construction_context.get("shared_reserve_pool")
        if shared_reserve_pool is not None:
            out["shared_reserve_pool"] = shared_reserve_pool
        return out

    async def _construct_tree(
        self,
        initial_prompt: str,
        max_depth: int = 2,
        data_instance: Optional[Dict[str, Any]] = None,
        root_allocation_info: Optional[Dict[str, Any]] = None,
        shared_reserve_pool: Optional[SharedReservePool] = None,
    ):
        return await self._construct_budget_allocated_tree(
            initial_prompt=initial_prompt,
            max_depth=max_depth,
            data_instance=data_instance,
            root_allocation_info=root_allocation_info,
            shared_reserve_pool=shared_reserve_pool,
        )

    async def _construct_budget_allocated_tree(
        self,
        initial_prompt: str,
        max_depth: int = 2,
        data_instance: Optional[Dict[str, Any]] = None,
        root_allocation_info: Optional[Dict[str, Any]] = None,
        shared_reserve_pool: Optional[SharedReservePool] = None,
    ):
        """Construct an online GEAR tree with per-node k prediction.

        Each generated non-leaf node receives the SPO default branch budget N.
        Its k-predictor either spends fewer branches immediately, contributing
        N-k to the minibatch reserve, or sends the node to a same-root queue
        that can draw from the shared reserve on timeout/final drain.
        """

        self._validate_gear_generation_contract()
        t0_tree = time.time()
        client = self._ensure_lp_client()
        scorer = make_lp_scorer(client, self._tokenize)
        problem_id = (
            str(data_instance.get("_treetune__idx", uuid.uuid4()))
            if data_instance
            else str(uuid.uuid4())
        )
        reserve_pool = shared_reserve_pool or SharedReservePool(
            queue_count=self.gear_budget_queue_count
        )
        queue_manager = RootQueueManager(
            queue_count=self.gear_budget_queue_count,
            queue_capacity=getattr(self, "gear_budget_queue_capacity", 8),
            timeout_seconds=self.gear_budget_queue_timeout_seconds,
            reserve_pool=reserve_pool,
            n_min=self.gear_n_min,
            use_residual_budget=self.gear_use_residual_budget,
            policy_snapshot_id=problem_id,
            strict_vdra=getattr(self, "gear_strict_vdra", True),
            rounding_strategy=getattr(self, "gear_rounding_strategy", "largest_remainder"),
            rounding_seed=getattr(self, "gear_rounding_seed", 0),
        )

        tree: Node = {
            "text": initial_prompt,
            "depth": 0,
            "gear_depth": 0,
            "full_text": initial_prompt,
            "stop_text": "aaa",
            "_request_object": data_instance,
            "leaf": False,
            "gear_action": Action.EXPAND.value,
            "gear_segment_id": "root",
            "gear_algorithm_mode": "budget_allocation",
            "gear_k_algorithm": self.gear_k_algorithm,
            "gear_generation_mode": self.gear_generation_mode,
            "gear_allocation_mode": self.gear_allocation_mode,
            "gear_use_residual_budget": self.gear_use_residual_budget,
            "gear_n_min": self.gear_n_min,
            "M": self.M,
        }

        def _default_branch_factor(depth: int) -> int:
            try:
                return max(
                    int(self.node_expander.branch_factor_strategy({"depth": depth})),
                    0,
                )
            except Exception as exc:
                if getattr(self, "gear_strict_vdra", True):
                    raise RuntimeError("VDRA branch-factor strategy failed") from exc
                return 1

        def _max_tokens_for_expansion(depth: int) -> Optional[int]:
            return None if depth >= max_depth - 1 else self.M

        def _node_id(node: Node, fallback: str) -> str:
            return str(node.get("gear_segment_id") or fallback)

        requested_by_depth: Dict[int, int] = {}
        allocated_by_depth: Dict[int, int] = {}
        built_by_depth: Dict[int, int] = {}
        underallocated_by_depth: Dict[int, int] = {}
        variance_seconds_by_depth: Dict[int, float] = {}
        allocation_seconds_by_depth: Dict[int, float] = {}
        queue_flush_records: List[Dict[str, Any]] = []
        expansion_seconds_by_depth: Dict[int, float] = {}
        branch_factor_by_depth: Dict[int, int] = {}
        generation_request_count = 0
        generation_rollout_count = 0
        generation_seconds = 0.0
        k_prediction_count = 0
        direct_expand_count = 0
        queued_node_count = 0
        early_leaf_reserve_count = 0

        tv_mode = self._tv_mode_for_k_algorithm()
        tv_estimator = ConditionalTVEstimator(
            scorer=scorer,
            node_expander=self.node_expander,
            gamma=self.cfg_thresholds.gamma,
            mode=tv_mode,
            pilot_branch_factor=self.gear_pilot_branch_factor,
            likelihood_samples_per_distribution=self.gear_likelihood_samples_per_distribution,
            invalid_support_policy=getattr(self, "gear_invalid_support_policy", "error"),
            strict_vdra=getattr(self, "gear_strict_vdra", True),
            first_phase_tokens=self.gear_tv_subnode_max_tokens,
            second_phase_tokens=self.gear_tv_second_phase_tokens,
            tv_includes_half_factor=self.gear_tv_includes_half_factor,
            tv_estimator=getattr(self, "gear_tv_estimator", "tanh"),
            r_max=self.cfg_thresholds.r_max,
            eps_tail=self.cfg_thresholds.eps_tail,
            bound_form=self.cfg_thresholds.bound_form,
        )

        async def _expand_raw(
            *,
            current_node: Node,
            prefix: str,
            depth: int,
            max_tokens: Optional[int],
            branch_factor: int,
        ) -> List[Node]:
            nonlocal generation_request_count, generation_rollout_count, generation_seconds
            branch_factor = max(int(branch_factor), 0)
            if branch_factor <= 0:
                return []
            t_expand = time.time()
            if self.gear_generation_mode == "single_request":
                generation_request_count += 1
                nodes = await self.node_expander.expand(
                    current_node=current_node,
                    prefix=prefix,
                    depth=depth,
                    max_tokens=max_tokens,
                    branch_factor=branch_factor,
                )
            elif self.gear_generation_mode == "replicated_requests":
                generation_request_count += branch_factor
                batches = await asyncio.gather(
                    *(
                        self.node_expander.expand(
                            current_node=current_node,
                            prefix=prefix,
                            depth=depth,
                            max_tokens=max_tokens,
                            branch_factor=1,
                        )
                        for _ in range(branch_factor)
                    )
                )
                nodes = [node for batch in batches for node in batch]
            else:
                raise ValueError(f"Unsupported gear_generation_mode: {self.gear_generation_mode}")
            generation_seconds += time.time() - t_expand
            nodes = list(nodes)[:branch_factor]
            generation_rollout_count += len(nodes)
            for idx, node in enumerate(nodes):
                self._require_generated_logprobs(
                    node,
                    context=(
                        f"parent={_node_id(current_node, 'unknown')} "
                        f"depth={depth} rollout_idx={idx}"
                    ),
                )
            return nodes

        async def _complete_candidate(
            parent: Node, candidate: Node, depth: int, child_idx: int
        ) -> Node:
            child = dict(candidate)
            child["gear_segment_id"] = f"{_node_id(parent, 'root')}/{depth}/{child_idx}"
            child["gear_parent_segment_id"] = _node_id(parent, "root")
            child["gear_depth"] = depth + 1
            child["gear_action"] = Action.EXPAND.value
            child["depth"] = depth + 1
            child["leaf"] = False
            self._require_generated_logprobs(
                child,
                context=f"candidate={child['gear_segment_id']}",
            )
            if child.get("finish_reason") != "length" or depth + 1 >= max_depth:
                child["reward"], _ = self.reward_function(
                    query=parent.get("full_text", ""),
                    response=child.get("full_text", child.get("text", "")),
                    dataset_instance=data_instance,
                )
                child["leaf"] = True
                return child

            continuation_budget = self.M
            if self.gear_tv_subnode_max_tokens > 0 and self.M is not None:
                continuation_budget = max(int(self.M) - self.gear_tv_subnode_max_tokens, 1)
            continuations = await _expand_raw(
                current_node=child,
                prefix=child.get("full_text", ""),
                depth=depth + 1,
                max_tokens=continuation_budget,
                branch_factor=1,
            )
            if not continuations:
                _finalize_empty_expansion_node(
                    child,
                    initial_prompt=initial_prompt,
                    data_instance=data_instance,
                    reward_function=self.reward_function,
                )
                return child

            cont = continuations[0]
            child["text"] = child.get("text", "") + cont.get("text", "")
            child["full_text"] = cont.get("full_text", child.get("full_text", ""))
            child["finish_reason"] = cont.get("finish_reason", child.get("finish_reason"))
            child["stop_text"] = cont.get("stop_text", child.get("stop_text"))
            child["sum_logprobs"] = float(child.get("sum_logprobs", 0.0)) + float(
                cont.get("sum_logprobs", 0.0)
            )
            child["num_tokens"] = int(child.get("num_tokens", 0)) + int(
                cont.get("num_tokens", 0)
            )
            if child.get("finish_reason") != "length":
                child["reward"], _ = self.reward_function(
                    query=parent.get("full_text", ""),
                    response=child.get("full_text", child.get("text", "")),
                    dataset_instance=data_instance,
                )
                child["leaf"] = True
            return child

        async def _expand_with_candidates(
            parent: Node,
            *,
            depth: int,
            branch_factor: int,
            candidates: Optional[Sequence[Node]] = None,
            shortcut: Optional[Sequence[Node]] = None,
        ) -> List[Node]:
            branch_factor = max(int(branch_factor), 0)
            shortcut_candidates = list(shortcut or [])
            if branch_factor <= 0 and not shortcut_candidates:
                return []
            # Terminal phase-1 pilots are complete answers: keep every one of
            # them (dropping generated full trajectories wastes compute) and
            # count them against the branch budget.
            parent["vdra_shortcut_overage"] = max(
                len(shortcut_candidates) - branch_factor, 0
            )
            target = max(branch_factor, len(shortcut_candidates))
            reuse_budget = max(branch_factor - len(shortcut_candidates), 0)
            # Reuse selection is a seeded uniform draw over the post-pruning
            # survivors — never likelihood-ranked (that would bias the child
            # sample the node value is estimated from).
            selected_candidates = select_reuse_candidates(
                list(candidates or []),
                reuse_budget,
                seed=f"vdra-reuse:{_node_id(parent, 'root')}",
            )
            tasks = [
                asyncio.create_task(_complete_candidate(parent, cand, depth, idx))
                for idx, cand in enumerate(shortcut_candidates + selected_candidates)
            ]
            completed_candidates = await asyncio.gather(*tasks) if tasks else []
            children = list(completed_candidates)
            if len(children) < target:
                children.extend(
                    await _expand_raw(
                        current_node=parent,
                        prefix=parent.get("full_text", ""),
                        depth=depth,
                        max_tokens=_max_tokens_for_expansion(depth),
                        branch_factor=target - len(children),
                    )
                )
            return children[:target]

        def _set_reward_summary(node: Node) -> None:
            rewards = []
            for child in node.get("children", []) or []:
                reward = child.get("reward")
                if reward is None or (isinstance(reward, float) and math.isnan(reward)):
                    continue
                rewards.append(float(reward))
            if not rewards:
                node["reward"] = 0.0
                node["reward_std"] = 0.0
                return
            avg = sum(rewards) / len(rewards)
            node["reward"] = float(avg)
            node["reward_std"] = float(
                (sum((value - avg) ** 2 for value in rewards) / len(rewards)) ** 0.5
            )

        async def _predict_k(
            node: Node,
            depth: int,
            default_branch_factor: int,
            *,
            require_generated_logprobs: bool = True,
        ) -> Dict[str, Any]:
            nonlocal k_prediction_count
            k_prediction_count += 1
            if require_generated_logprobs:
                self._require_generated_logprobs(
                    node,
                    context=f"k-predictor node={_node_id(node, 'unknown')} depth={depth}",
                )
            if self.gear_k_algorithm == "simple":
                if not require_generated_logprobs:
                    raise RuntimeError(
                        "GEAR simple k predictor cannot run on the root before "
                        "generation because root has no generated logprobs."
                    )
                tokens = int(node.get("num_tokens") or 0)
                if tokens <= 0:
                    raise RuntimeError("GEAR simple k predictor requires num_tokens > 0")
                ppl = math.exp(-float(node["sum_logprobs"]) / tokens)
                k = max(int(math.ceil(ppl)), 0)
                node["gear_predicted_k"] = k
                node["gear_perplexity"] = float(ppl)
                node["gear_allocation_weight_override"] = float(k)
                return {"k": k, "candidates": [], "shortcut": [], "weight_key": "gear_allocation_weight_override"}

            entropy_anchor_text = ""
            estimator_parent = node
            if self.gear_k_algorithm == "entropy_guided":
                generated_text, entropies, entropy_tokens = await client.completion_with_token_entropies(
                    node.get("full_text", ""),
                    max_tokens=self.gear_tv_subnode_max_tokens,
                )
                anchor_idx = max(range(len(entropies)), key=entropies.__getitem__) if entropies else 0
                entropy_anchor_text = (
                    "".join(entropy_tokens[: anchor_idx + 1])
                    if entropy_tokens
                    else generated_text
                )
                node["gear_entropy_anchor_index"] = int(anchor_idx)
                node["gear_entropy_max"] = float(entropies[anchor_idx]) if entropies else 0.0
                node["gear_entropy_probe_text"] = generated_text
                node["gear_entropy_anchor_text"] = entropy_anchor_text
                estimator_parent = dict(node)
                estimator_parent["text"] = node.get("text", "") + entropy_anchor_text
                estimator_parent["full_text"] = node.get("full_text", "") + entropy_anchor_text

            mode = self._tv_mode_for_k_algorithm()
            estimator = tv_estimator
            if estimator.mode != mode:
                estimator = ConditionalTVEstimator(
                    scorer=scorer,
                    node_expander=self.node_expander,
                    gamma=self.cfg_thresholds.gamma,
                    mode=mode,
                    pilot_branch_factor=self.gear_pilot_branch_factor,
            likelihood_samples_per_distribution=self.gear_likelihood_samples_per_distribution,
            invalid_support_policy=getattr(self, "gear_invalid_support_policy", "error"),
            strict_vdra=getattr(self, "gear_strict_vdra", True),
                    first_phase_tokens=self.gear_tv_subnode_max_tokens,
                    second_phase_tokens=self.gear_tv_second_phase_tokens,
                    tv_includes_half_factor=self.gear_tv_includes_half_factor,
            tv_estimator=getattr(self, "gear_tv_estimator", "tanh"),
            r_max=self.cfg_thresholds.r_max,
            eps_tail=self.cfg_thresholds.eps_tail,
            bound_form=self.cfg_thresholds.bound_form,
                )
            t_var = time.time()
            result = await estimator.estimate_k_for_parent(
                estimator_parent,
                depth=depth,
                duplicate_tv_threshold=self.cfg_thresholds.epsilon,
            )
            variance_seconds_by_depth[depth] = variance_seconds_by_depth.get(depth, 0.0) + (time.time() - t_var)
            if entropy_anchor_text:
                adjusted_candidates = []
                for candidate in result.unique_candidates:
                    adjusted = dict(candidate)
                    adjusted["text"] = entropy_anchor_text + adjusted.get("text", "")
                    adjusted["gear_entropy_anchor_text"] = entropy_anchor_text
                    adjusted["sum_logprobs"] = await scorer.score_one(
                        node.get("full_text", ""), adjusted.get("text", "")
                    )
                    adjusted["num_tokens"] = max(
                        len(self._tokenize(node.get("full_text", "") + adjusted.get("text", "")))
                        - len(self._tokenize(node.get("full_text", ""))),
                        1,
                    )
                    adjusted_candidates.append(adjusted)
                result.unique_candidates = adjusted_candidates

            k = max(int(result.predicted_k), 0)
            node["gear_predicted_k"] = k
            from vdra_core.proxies import select_dispersion_proxy
            node["vdra_dispersion_C"] = select_dispersion_proxy(
                getattr(self, "gear_allocation_proxy", "vdra"),
                vdra_dispersion_C=getattr(result, "dispersion_C", getattr(result, "reward_variance", 0.0)),
                pair_tvs=result.pair_tvs,
                pilot_count=len(getattr(result, "candidates", getattr(result, "unique_candidates", []))),
                node=node,
            )
            node["gear_tv_pair_count"] = len(result.pair_tvs)
            node["gear_tv_support_size"] = len(result.samples)
            node["gear_duplicate_prefix_pairs"] = len(result.duplicate_pairs)
            node["gear_budget_candidates"] = list(result.unique_candidates)
            node["gear_shortcut_candidates"] = list(getattr(result, "shortcut_candidates", []))
            node["vdra_pilot_children_shortcut"] = len(node["gear_shortcut_candidates"])
            node["gear_tv_logp_matrix"] = result.logp_matrix
            return {
                "k": k,
                "candidates": result.unique_candidates,
                "shortcut": node["gear_shortcut_candidates"],
                "weight_key": None,
            }

        async def _handle_queue_flush(result) -> None:
            nonlocal allocation_seconds_by_depth
            queue_flush_records.append(result.to_record())
            allocation_seconds_by_depth[-1] = allocation_seconds_by_depth.get(-1, 0.0) + result.allocation_seconds
            for item in result.items:
                node = item.node
                node_id = _node_id(node, "unknown")
                allocated = int(result.summary.allocations.get(node_id, 0))
                write_node_accounting(
                    node,
                    default_k=item.default_branch_factor,
                    predicted_k=int(node["vdra_predicted_k"]),
                    allocated_k=allocated,
                    k_min=self.gear_n_min,
                    dispersion_C=float(node.get("vdra_dispersion_C", 0.0) or 0.0),
                    allocation_weight=result.summary.weights[node_id],
                )
                if self.gear_strict_vdra:
                    validate_node_accounting(node, k_min=self.gear_n_min)
                node["gear_queue_base_budget"] = item.default_branch_factor
                node["gear_queue_total_budget"] = result.total_budget
                node["gear_queue_reserve_draw"] = result.reserve_draw
                await _expand_parent(
                    node,
                    depth=item.depth,
                    allocated_branch_factor=allocated,
                    default_branch_factor=item.default_branch_factor,
                    candidates=node.get("gear_budget_candidates") or [],
                    shortcut=node.get("gear_shortcut_candidates") or [],
                )

        async def _flush_ready_queues() -> None:
            for result in await queue_manager.flush_ready():
                await _handle_queue_flush(result)

        async def _drain_queues() -> None:
            while True:
                results = await queue_manager.drain()
                if not results:
                    break
                for result in results:
                    await _handle_queue_flush(result)

        async def _handle_generated_child(child: Node, parent: Node, parent_depth: int, child_idx: int) -> None:
            nonlocal direct_expand_count, queued_node_count, early_leaf_reserve_count
            child_depth = parent_depth + 1
            child["gear_segment_id"] = f"{_node_id(parent, 'root')}/{parent_depth}/{child_idx}"
            child["gear_parent_segment_id"] = _node_id(parent, "root")
            child["gear_depth"] = child_depth
            child["depth"] = child_depth
            child["gear_action"] = Action.EXPAND.value
            self._require_generated_logprobs(
                child,
                context=f"generated child={child['gear_segment_id']}",
            )
            if child.get("finish_reason") != "length" or child_depth >= max_depth:
                child["reward"], _ = self.reward_function(
                    query=parent.get("full_text", ""),
                    response=child.get("full_text", child.get("text", "")),
                    dataset_instance=data_instance,
                )
                child["leaf"] = True
                if child_depth < max_depth:
                    default_n = _default_branch_factor(child_depth)
                    requested_by_depth[child_depth] = requested_by_depth.get(child_depth, 0) + default_n
                    underallocated_by_depth[child_depth] = underallocated_by_depth.get(child_depth, 0) + default_n
                    if self.gear_use_residual_budget:
                        await reserve_pool.add(default_n)
                        early_leaf_reserve_count += default_n
                return

            child["leaf"] = False
            default_n = _default_branch_factor(child_depth)
            branch_factor_by_depth[child_depth] = default_n
            if self.gear_skip_near_leaf_expand and child_depth >= max_depth - 1:
                child["gear_near_leaf_skip"] = True
                child["gear_predicted_k"] = default_n
                direct_expand_count += 1
                await _expand_parent(
                    child,
                    depth=child_depth,
                    allocated_branch_factor=default_n,
                    default_branch_factor=default_n,
                    candidates=[],
                )
                return

            prediction = await _predict_k(child, child_depth, default_n)
            k = int(prediction["k"])
            if self.gear_allocation_mode == "prune_only":
                direct_expand_count += 1
                await _expand_parent(
                    child,
                    depth=child_depth,
                    allocated_branch_factor=min(max(k, 0), default_n),
                    default_branch_factor=default_n,
                    candidates=prediction.get("candidates") or [],
                    shortcut=prediction.get("shortcut") or [],
                )
            elif k < default_n:
                reserve_delta = default_n - max(k, 0)
                if self.gear_use_residual_budget:
                    await reserve_pool.add(reserve_delta)
                direct_expand_count += 1
                await _expand_parent(
                    child,
                    depth=child_depth,
                    allocated_branch_factor=max(k, 0),
                    default_branch_factor=default_n,
                    candidates=prediction.get("candidates") or [],
                    shortcut=prediction.get("shortcut") or [],
                )
            else:
                queued_node_count += 1
                queue_manager.enqueue(
                    OnlineQueueItem(
                        node=child,
                        default_branch_factor=default_n,
                        depth=child_depth,
                        weight_key=prediction.get("weight_key"),
                    )
                )
                await _flush_ready_queues()

        async def _expand_parent(
            node: Node,
            *,
            depth: int,
            allocated_branch_factor: int,
            default_branch_factor: int,
            candidates: Optional[Sequence[Node]] = None,
            shortcut: Optional[Sequence[Node]] = None,
        ) -> None:
            branch_factor_by_depth[depth] = default_branch_factor
            requested_by_depth[depth] = requested_by_depth.get(depth, 0) + max(default_branch_factor, 0)
            allocated_by_depth[depth] = allocated_by_depth.get(depth, 0) + max(allocated_branch_factor, 0)
            if allocated_branch_factor <= 0 and not (shortcut or []):
                underallocated_by_depth[depth] = underallocated_by_depth.get(depth, 0) + max(default_branch_factor, 0)
                _finalize_empty_expansion_node(
                    node,
                    initial_prompt=initial_prompt,
                    data_instance=data_instance,
                    reward_function=self.reward_function,
                )
                return

            node["gear_allocated_branch_factor"] = int(allocated_branch_factor)
            node["gear_default_branch_factor"] = int(default_branch_factor)
            t_expand = time.time()
            children = await _expand_with_candidates(
                node,
                depth=depth,
                branch_factor=allocated_branch_factor,
                candidates=candidates,
                shortcut=shortcut,
            )
            expansion_seconds_by_depth[depth] = expansion_seconds_by_depth.get(depth, 0.0) + (time.time() - t_expand)
            built_by_depth[depth] = built_by_depth.get(depth, 0) + len(children)
            underallocated_by_depth[depth] = underallocated_by_depth.get(depth, 0) + max(default_branch_factor - len(children), 0)
            node["children"] = children
            for idx, child in enumerate(children):
                await _handle_generated_child(child, node, depth, idx)
            _set_reward_summary(node)
            await _flush_ready_queues()

        root_default = _default_branch_factor(0)
        branch_factor_by_depth[0] = root_default
        if (
            max_depth <= 1
            and self.gear_k_algorithm != "simple"
            and root_allocation_info is None
        ):
            prediction = await _predict_k(
                tree,
                0,
                root_default,
                require_generated_logprobs=False,
            )
            root_k = int(prediction["k"])
            if self.gear_allocation_mode == "prune_only":
                direct_expand_count += 1
                await _expand_parent(
                    tree,
                    depth=0,
                    allocated_branch_factor=min(max(root_k, 0), root_default),
                    default_branch_factor=root_default,
                    candidates=prediction.get("candidates") or [],
                    shortcut=prediction.get("shortcut") or [],
                )
            elif root_k < root_default:
                reserve_delta = root_default - max(root_k, 0)
                if self.gear_use_residual_budget:
                    await reserve_pool.add(reserve_delta)
                direct_expand_count += 1
                await _expand_parent(
                    tree,
                    depth=0,
                    allocated_branch_factor=max(root_k, 0),
                    default_branch_factor=root_default,
                    candidates=prediction.get("candidates") or [],
                    shortcut=prediction.get("shortcut") or [],
                )
            else:
                queued_node_count += 1
                queue_manager.enqueue(
                    OnlineQueueItem(
                        node=tree,
                        default_branch_factor=root_default,
                        depth=0,
                        weight_key=prediction.get("weight_key"),
                    )
                )
        elif root_allocation_info is not None and self.gear_allocation_mode != "prune_only":
            root_allocated = int(root_allocation_info.get("allocated_branch_factor", root_default))
            tree["vdra_dispersion_C"] = float(root_allocation_info.get("dispersion_C", 0.0))
            tree["gear_budget_candidates"] = list(root_allocation_info.get("budget_candidates", []))
            tree["gear_shortcut_candidates"] = list(root_allocation_info.get("shortcut_candidates", []))
            await _expand_parent(
                tree,
                depth=0,
                allocated_branch_factor=root_allocated,
                default_branch_factor=root_default,
                candidates=tree.get("gear_budget_candidates") or [],
                shortcut=tree.get("gear_shortcut_candidates") or [],
            )
        else:
            root_allocated = root_default
            await _expand_parent(
                tree,
                depth=0,
                allocated_branch_factor=root_allocated,
                default_branch_factor=root_default,
                candidates=tree.get("gear_budget_candidates") or [],
                shortcut=tree.get("gear_shortcut_candidates") or [],
            )
        await _drain_queues()

        # Queued children get their rewards only at flush time, after their
        # parent's inline reward summary already ran — recompute every internal
        # node bottom-up so ancestor values (advantage baselines) are correct.
        def _backprop_rewards(node: Node) -> None:
            for child in node.get("children", []) or []:
                _backprop_rewards(child)
            if node.get("children"):
                _set_reward_summary(node)

        _backprop_rewards(tree)

        tree["gear_max_depth"] = int(max_depth)
        tree["gear_branch_factor_by_depth"] = dict(branch_factor_by_depth)
        tree["gear_requested_node_budget_by_depth"] = dict(requested_by_depth)
        tree["gear_allocated_branch_factor_by_depth"] = dict(allocated_by_depth)
        tree["gear_built_nodes_by_depth"] = dict(built_by_depth)
        tree["gear_underallocated_rollouts_by_depth"] = dict(underallocated_by_depth)
        tree["gear_variance_seconds_by_depth"] = dict(variance_seconds_by_depth)
        tree["gear_allocation_seconds_by_depth"] = dict(allocation_seconds_by_depth)
        tree["vdra_queue_flush_records"] = queue_flush_records
        tree["gear_expansion_seconds_by_depth"] = dict(expansion_seconds_by_depth)
        tree["gear_budget_overhead_mode"] = self.gear_budget_overhead_mode
        tree["gear_allocation_mode"] = self.gear_allocation_mode
        tree["gear_use_residual_budget"] = self.gear_use_residual_budget
        tree["gear_skip_near_leaf_expand"] = self.gear_skip_near_leaf_expand
        tree["gear_root_allocation"] = self.gear_root_allocation
        tree["gear_n_min"] = self.gear_n_min
        tree["M"] = self.M
        tree["gear_generation_request_count"] = int(generation_request_count)
        tree["gear_generation_rollout_count"] = int(generation_rollout_count)
        tree["gear_generation_seconds"] = float(generation_seconds)
        tree["gear_generation_rollouts_per_second"] = (
            float(generation_rollout_count) / generation_seconds if generation_seconds > 0.0 else 0.0
        )
        tree["gear_queue_flush_count"] = int(queue_manager.flush_count)
        tree["gear_queue_timeout_flush_count"] = int(queue_manager.timeout_flush_count)
        tree["vdra_queue_capacity_flush_count"] = int(queue_manager.capacity_flush_count)
        tree["vdra_queue_final_drain_count"] = int(queue_manager.final_drain_count)
        tree["vdra_budget_mode"] = getattr(self, "gear_budget_mode", "fixed_main")
        tree["gear_reserve_contributed"] = int(reserve_pool.contributed)
        tree["gear_reserve_consumed"] = int(reserve_pool.consumed)
        tree["gear_reserve_remaining"] = int(reserve_pool.value)
        tree["gear_k_prediction_count"] = int(k_prediction_count)
        tree["gear_direct_expand_count"] = int(direct_expand_count)
        tree["gear_queued_node_count"] = int(queued_node_count)
        tree["gear_early_leaf_reserve_count"] = int(early_leaf_reserve_count)
        tree_construction_seconds = time.time() - t0_tree
        tree["tree_construction_seconds"] = tree_construction_seconds
        tree["gear_tree_construction_seconds"] = tree_construction_seconds
        tree["gear_problem_id"] = problem_id
        tree["gear_stats"] = {
            **aggregate_tree_stats(tree),
            "gear/budget/underallocated_node_budget": float(sum(underallocated_by_depth.values())),
            "gear/variance_estimation_seconds": float(sum(variance_seconds_by_depth.values())),
            "gear/budget_allocation_seconds": float(sum(allocation_seconds_by_depth.values())),
            "gear/expansion_seconds": float(sum(expansion_seconds_by_depth.values())),
            "gear/generation_request_count": float(generation_request_count),
            "gear/generation_rollout_count": float(generation_rollout_count),
            "gear/generation_rollouts_per_second": float(tree["gear_generation_rollouts_per_second"]),
            "gear/queue_flush_count": float(queue_manager.flush_count),
            "gear/queue_timeout_flush_count": float(queue_manager.timeout_flush_count),
            "gear/reserve_contributed": float(reserve_pool.contributed),
            "gear/reserve_consumed": float(reserve_pool.consumed),
        }
        return tree
