"""GEAR online gate (Step 3): prune / share / budget allocation.

Wraps the vendored ``gear_core`` and hooks into the tree builders through:

  * ``branch_factor(parent, depth, default_bf)`` — online **prune** for the
    ``simple`` perplexity predictor (``k = ceil(exp(-sum_logprobs/num_tokens))``,
    byte-exact port of ``GEARInferenceStrategy._predict_k``).
  * ``allocate_batch_async(nodes, depth, default_bf, node_expander)`` — the
    VDRA **budget-allocation** path (``k_algorithm='budget_allocation'``):
    pilot short continuations per node, pairwise TV via the §9 tanh estimator,
    value-dispersion bound C_s (tail-corrected, ``value_gap_bound``), then a
    queue-batched ``allocate_branch_factors`` solve (``k_s ∝ sqrt(C_s)`` with
    the ``n_min`` floor).  Results are written to
    ``node['vdra_allocated_k']`` (with legacy allocation aliases).
  * ``filter_children`` / ``filter_children_async`` — sibling-local **share**
    via ``gear_core.local_value_share``.  The async variant awaits async
    scorers (``lp_scorer.LPScorer``); the sync variant drives them with
    ``asyncio.run`` (SPMD path).  Scoring failures are logged and counted,
    never silently swallowed.

Thresholds come from ``gear_core.gear.thresholds`` (Lemma 2.4 + Summary.md §7
tail correction), so ``eta``/``tau``/``value_gap_bound`` are shared with the
offline analysis code.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from typing import Any, Dict, List, Mapping, Optional

from vdra_core.logging_schema import node_allocated_k, write_node_accounting

from recipe.gear_tree.gear_core.gear.thresholds import (
    ThresholdConfig,
    eps_tail_for_depth,
)
from recipe.gear_tree.gear_core.gear import local_value_share as lvs

logger = logging.getLogger(__name__)


class GearGate:
    """Online GEAR gate driven by the recipe ``gear`` config block."""

    def __init__(
        self,
        *,
        epsilon: float = 0.02,
        r_max: float = 1.0,
        gamma: float = 0.9,
        alpha: float = 0.05,
        k_algorithm: str = "budget_allocation",
        n_min: int = 1,
        pilot_branch_factor: Optional[int] = None,
        likelihood_samples_per_distribution: int = 2,
        root_allocation: bool = False,
        skip_near_leaf_expand: bool = True,
        max_depth: Optional[int] = None,
        enable_share: bool = False,
        scorer: Any = None,
        # --- VDRA additions -------------------------------------------------
        eps_tail: float = 0.0,
        eps_tail_by_depth: Optional[Mapping[int, float]] = None,
        bound_form: str = "linear",
        tv_estimator: str = "tanh",
        tv_first_phase_tokens: int = 60,
        tv_second_phase_tokens: int = 60,
        queue_count: int = 4,
        queue_capacity: int = 8,
        queue_timeout_seconds: float = 1.0,
        use_residual_budget: bool = True,
        strict_vdra: bool = True,
        invalid_support_policy: str = "error",
        budget_mode: str = "fixed_main",
        allocation_proxy: str = "vdra",
        allocation_runtime: str = "online_timeout",
        artifact_dir: Optional[str] = None,
        eps_tail_calibration_path: Optional[str] = None,
        eps_tail_calibration_metadata: Optional[Mapping[str, Any]] = None,
        oracle_rollouts_per_node: int = 16,
        external_score_fn: Optional[Any] = None,
        rounding_strategy: str = "largest_remainder",
        rounding_seed: int = 0,
    ) -> None:
        self.cfg = ThresholdConfig(
            epsilon=epsilon,
            r_max=r_max,
            gamma=gamma,
            alpha=alpha,
            eps_tail=float(eps_tail),
            eps_tail_by_depth=(
                {int(k): float(v) for k, v in eps_tail_by_depth.items()}
                if eps_tail_by_depth
                else None
            ),
            bound_form=bound_form,
        )
        self.k_algorithm = k_algorithm
        if allocation_runtime not in {"online_timeout", "depth_batch"}:
            raise ValueError(f"Unsupported VDRA allocation_runtime: {allocation_runtime}")
        self.allocation_runtime = allocation_runtime
        self.artifact_dir = artifact_dir
        self.n_min = max(int(n_min), 0)
        self.pilot_branch_factor = pilot_branch_factor
        self.likelihood_samples_per_distribution = max(
            int(likelihood_samples_per_distribution), 1
        )
        self.root_allocation = bool(root_allocation)
        self.skip_near_leaf_expand = bool(skip_near_leaf_expand)
        self.max_depth = max_depth
        self.enable_share = bool(enable_share)
        self.scorer = scorer  # lp_scorer.LPScorer (async), EngineLPScorer (sync), or None
        self.tv_estimator = tv_estimator
        self.tv_first_phase_tokens = int(tv_first_phase_tokens)
        self.tv_second_phase_tokens = int(tv_second_phase_tokens)
        self.queue_count = max(int(queue_count), 1)
        self.queue_capacity = max(int(queue_capacity), 1)
        self.queue_timeout_seconds = max(float(queue_timeout_seconds), 0.0)
        self.use_residual_budget = bool(use_residual_budget)
        self.strict_vdra = bool(strict_vdra)
        self.invalid_support_policy = str(invalid_support_policy)
        if budget_mode not in {"fixed_main", "fixed_total_generated"}:
            raise ValueError(f"Unsupported VDRA budget_mode: {budget_mode}")
        self.budget_mode = budget_mode
        if allocation_proxy not in {
            "vdra", "uniform", "random", "direct_tv", "empirical_variance",
            "external_score", "oracle",
        }:
            raise ValueError(f"Unsupported allocation_proxy: {allocation_proxy}")
        self.allocation_proxy = allocation_proxy
        self.oracle_rollouts_per_node = max(int(oracle_rollouts_per_node), 2)
        self.external_score_fn = external_score_fn
        if rounding_strategy not in {"largest_remainder", "nearest_repair", "stochastic"}:
            raise ValueError(f"Unsupported VDRA rounding_strategy: {rounding_strategy}")
        self.rounding_strategy = rounding_strategy
        self.rounding_seed = int(rounding_seed)
        self.eps_tail_calibration_path = eps_tail_calibration_path
        self.eps_tail_calibration_metadata = (
            dict(eps_tail_calibration_metadata) if eps_tail_calibration_metadata else None
        )
        if self.k_algorithm == "budget_allocation" and self.scorer is None:
            raise ValueError("VDRA budget_allocation requires a likelihood scorer")
        self.share_error_count = 0
        self.allocation_error_count = 0

    # --- capabilities -------------------------------------------------------- #
    @property
    def use_batch_allocation(self) -> bool:
        """True when the legacy depth-batched allocation path should run."""
        return self.k_algorithm == "budget_allocation" and self.allocation_runtime == "depth_batch"

    @property
    def use_online_allocation(self) -> bool:
        """True when the main online timeout allocation path should run."""
        return self.k_algorithm == "budget_allocation" and self.allocation_runtime == "online_timeout"

    def validate_main_config(self, *, max_default_branch_factor: int, segment_length: Optional[int]) -> None:
        if not self.strict_vdra:
            return
        if self.tv_first_phase_tokens > int(segment_length or self.tv_first_phase_tokens):
            raise ValueError(
                "VDRA pilot length cannot exceed the main segment length "
                "when pilots are reused as tree children."
            )
        if self.use_online_allocation and self.queue_timeout_seconds <= 0.0:
            raise ValueError("VDRA online_timeout requires queue_timeout_seconds > 0 in strict mode")
        if (
            self.k_algorithm == "budget_allocation"
            and self.use_residual_budget
            and self.pilot_branch_factor_for(max_default_branch_factor) <= int(max_default_branch_factor)
        ):
            raise ValueError(
                "The current cluster-count k predictor requires "
                "pilot_branch_factor > max default branch factor "
                "for positive unmet demand and residual redistribution."
            )

    def pilot_branch_factor_for(self, branch_factor: int) -> int:
        """Return k0, defaulting to the configured branch factor."""
        if self.pilot_branch_factor is not None:
            return max(int(self.pilot_branch_factor), 2)
        return max(int(branch_factor), 2)

    # --- online prune: predict k, shrink branch factor ---------------------- #
    def _predict_k_simple(self, node: Dict[str, Any]) -> Optional[int]:
        """Byte-exact port of the GEAR ``simple`` k-predictor."""
        tokens = int(node.get("num_tokens") or 0)
        if tokens <= 0 or "sum_logprobs" not in node:
            return None
        ppl = math.exp(-float(node["sum_logprobs"]) / tokens)
        k = max(int(math.ceil(ppl)), 0)
        node["gear_predicted_k"] = k
        node["gear_perplexity"] = float(ppl)
        return k

    def branch_factor(self, parent: Dict[str, Any], depth: int, default_bf: int) -> int:
        # Root has no generated logprobs; keep the configured width.
        if depth == 0:
            return default_bf
        # Near-leaf: skip TV/budget, expand uniformly (gear_defaults semantics).
        if self.skip_near_leaf_expand and self.max_depth is not None and depth == self.max_depth - 1:
            return default_bf
        # Batch allocation writes vdra_allocated_k via allocate_batch_async.
        allocated = node_allocated_k(parent)
        if self.use_batch_allocation and allocated is not None:
            return max(int(allocated), 0)
        if self.k_algorithm != "simple":
            return default_bf
        k = self._predict_k_simple(parent)
        if k is None:
            return default_bf
        # Clamp to [n_min, default_bf]: never widen beyond the SPO width.
        return max(self.n_min, min(int(k), int(default_bf)))

    def make_queue_manager(self, *, policy_snapshot_id: str, reserve_pool: Any = None):
        from recipe.gear_tree.gear_core.gear.online_budget import (
            RootQueueManager,
            SharedReservePool,
        )

        if reserve_pool is None:
            reserve_pool = SharedReservePool(queue_count=self.queue_count)
        return RootQueueManager(
            queue_count=self.queue_count,
            queue_capacity=self.queue_capacity,
            timeout_seconds=self.queue_timeout_seconds,
            reserve_pool=reserve_pool,
            n_min=self.n_min,
            use_residual_budget=self.use_residual_budget,
            policy_snapshot_id=policy_snapshot_id,
            strict_vdra=self.strict_vdra,
            rounding_strategy=self.rounding_strategy,
            rounding_seed=self.rounding_seed,
        )

    def _record_estimate(
        self,
        node: Dict[str, Any],
        result: Any,
        estimator: Any,
        score_keys_before: set,
    ) -> None:
        """Write one TV-estimate result onto ``node`` (shared by both paths).

        Terminal (EOS-in-phase-1) pilots become ``vdra_shortcut_children``:
        complete trajectories the expansion step attaches as graded leaves
        counted against the branch budget. Reuse candidates are the
        post-pruning survivors only — selection among them is a seeded uniform
        draw at expansion time, never likelihood-ranked.
        """

        from vdra_core.proxies import select_dispersion_proxy

        node["vdra_dispersion_C"] = select_dispersion_proxy(
            self.allocation_proxy,
            vdra_dispersion_C=result.dispersion_C,
            pair_tvs=result.pair_tvs,
            pilot_count=len(result.candidates),
            node=node,
        )
        all_pilots = list(result.candidates)
        shortcut_pilots = list(result.shortcut_candidates)
        reusable_pilots = list(result.unique_candidates)
        support_nodes = list(result.support_nodes)
        node["vdra_all_pilot_children"] = all_pilots
        node["vdra_shortcut_children"] = shortcut_pilots
        node["vdra_reusable_pilot_children"] = reusable_pilots
        node["vdra_pilot_children"] = reusable_pilots  # legacy compatibility alias
        node["vdra_pilot_children_generated"] = len(all_pilots)
        node["vdra_pilot_children_shortcut"] = len(shortcut_pilots)
        node["vdra_pilot_children_reused"] = len(reusable_pilots) + len(shortcut_pilots)
        node["vdra_pilot_children_discarded"] = max(
            len(all_pilots) - len(reusable_pilots) - len(shortcut_pilots), 0
        )
        node["vdra_pilot_support_children_generated"] = len(support_nodes)
        node["vdra_pilot_support_generated_tokens"] = sum(
            int(support.get("num_tokens") or len(support.get("response_token_ids") or []))
            for support in support_nodes
        )
        node["vdra_generation_request_count"] = (
            node.get("vdra_generation_request_count", 0)
            + len(all_pilots)
            + len(support_nodes)
        )
        node["gear_predicted_k"] = int(result.predicted_k)
        node["vdra_predicted_k"] = int(result.predicted_k)
        node["gear_pair_tvs"] = {
            f"{i},{j}": float(tv) for (i, j), tv in result.pair_tvs.items()
        }
        node["gear_prob_matrix"] = result.prob_matrix
        score_keys = list(set(estimator._score_cache) - score_keys_before)
        node["vdra_likelihood_scoring_requests"] = len(score_keys)
        tokenize = getattr(self.scorer, "tokenize_fn", None)
        if tokenize is not None:
            node["vdra_likelihood_scored_prompt_tokens"] = sum(
                len(tokenize(prefix)) for prefix, _ in score_keys
            )
            node["vdra_likelihood_scored_continuation_tokens"] = sum(
                len(tokenize(text)) for _, text in score_keys
            )
            node["vdra_total_scored_tokens"] = (
                node.get("vdra_likelihood_scored_prompt_tokens", 0)
                + node.get("vdra_likelihood_scored_continuation_tokens", 0)
            )

    async def _prepare_proxy_fields(
        self,
        node: Dict[str, Any],
        *,
        default_bf: int,
        proxy_rollout_fn: Optional[Any],
    ) -> None:
        """Populate the node field a non-VDRA allocation proxy reads.

        ``empirical_variance`` / ``oracle`` need graded full rollouts, which
        only the online tree builder can provide (``proxy_rollout_fn``).
        ``oracle`` runs are evaluation-only and flagged in the run manifest.
        """

        if self.allocation_proxy == "external_score":
            if self.external_score_fn is None:
                raise ValueError(
                    "allocation_proxy='external_score' requires "
                    "gear.external_score_module (import path of a "
                    "callable(node) -> float)."
                )
            node["vdra_external_dispersion_C"] = max(float(self.external_score_fn(node)), 0.0)
            return
        if self.allocation_proxy not in {"empirical_variance", "oracle"}:
            return
        if proxy_rollout_fn is None:
            raise ValueError(
                f"allocation_proxy={self.allocation_proxy!r} needs graded full "
                "rollouts; it is only supported by the online allocation runtime."
            )
        rollouts = (
            self.oracle_rollouts_per_node
            if self.allocation_proxy == "oracle"
            else self.pilot_branch_factor_for(default_bf)
        )
        rewards = [float(r) for r in await proxy_rollout_fn(node, rollouts)]
        if rewards:
            mean = sum(rewards) / len(rewards)
            variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        else:
            variance = 0.0
        field = (
            "vdra_oracle_value_dispersion"
            if self.allocation_proxy == "oracle"
            else "vdra_empirical_reward_variance"
        )
        node[field] = variance

    async def estimate_node_async(
        self,
        node: Dict[str, Any],
        *,
        depth: int,
        default_bf: int,
        node_expander: Any,
        proxy_rollout_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        from recipe.gear_tree.gear_core.gear.tv_estimators import (
            ConditionalTVEstimator,
        )

        estimator = ConditionalTVEstimator(
            scorer=self.scorer,
            node_expander=node_expander,
            gamma=self.cfg.gamma,
            mode="subnode",
            pilot_branch_factor=self.pilot_branch_factor_for(default_bf),
            likelihood_samples_per_distribution=self.likelihood_samples_per_distribution,
            invalid_support_policy=self.invalid_support_policy,
            strict_vdra=self.strict_vdra,
            first_phase_tokens=self.tv_first_phase_tokens,
            second_phase_tokens=self.tv_second_phase_tokens,
            tv_estimator=self.tv_estimator,
            r_max=self.cfg.r_max,
            eps_tail=eps_tail_for_depth(self.cfg, depth),
            bound_form=self.cfg.bound_form,
        )
        score_keys_before = set(estimator._score_cache)
        result = await estimator.estimate_k_for_parent(
            node, depth=depth, duplicate_tv_threshold=self.cfg.epsilon
        )
        await self._prepare_proxy_fields(
            node, default_bf=default_bf, proxy_rollout_fn=proxy_rollout_fn
        )
        self._record_estimate(node, result, estimator, score_keys_before)
        predicted_k = max(int(result.predicted_k), self.n_min)
        write_node_accounting(
            node,
            default_k=int(default_bf),
            predicted_k=predicted_k,
            allocated_k=min(int(default_bf), predicted_k),
            k_min=self.n_min,
            dispersion_C=float(node.get("vdra_dispersion_C", 0.0) or 0.0),
        )
        return {
            "predicted_k": predicted_k,
            "candidates": list(result.unique_candidates),
            "shortcut": list(result.shortcut_candidates),
            "weight_key": None,
        }

    # --- VDRA batch allocation (Summary.md §10-§11) -------------------------- #
    async def allocate_batch_async(
        self,
        nodes: List[Dict[str, Any]],
        depth: int,
        default_bf: int,
        node_expander: Any,
    ) -> None:
        """Score dispersion bounds and allocate the depth budget across nodes.

        Writes ``gear_branch_allocation`` (int) plus scoring evidence
        (``vdra_dispersion_C``, ``gear_pair_tvs``, ``gear_predicted_k``)
        onto each node. Scorer or estimator failure aborts VDRA explicitly.
        """

        from recipe.gear_tree.gear_core.gear.online_budget import (
            OnlineQueueItem,
            RootQueueManager,
            SharedReservePool,
        )
        from recipe.gear_tree.gear_core.gear.tv_estimators import (
            ConditionalTVEstimator,
        )

        if not nodes:
            return
        if not self.use_batch_allocation:
            for node in nodes:
                write_node_accounting(
                    node,
                    default_k=int(default_bf),
                    predicted_k=int(default_bf),
                    allocated_k=int(default_bf),
                    k_min=self.n_min,
                )
            return

        def _make_estimator() -> ConditionalTVEstimator:
            # One estimator (and score cache) per node: sharing a cache across
            # nodes silently attributes cache-hit scoring cost to the wrong
            # node's token accounting.
            return ConditionalTVEstimator(
                scorer=self.scorer,
                node_expander=node_expander,
                gamma=self.cfg.gamma,
                mode="subnode",
                pilot_branch_factor=self.pilot_branch_factor_for(default_bf),
                likelihood_samples_per_distribution=self.likelihood_samples_per_distribution,
                invalid_support_policy=self.invalid_support_policy,
                strict_vdra=self.strict_vdra,
                first_phase_tokens=self.tv_first_phase_tokens,
                second_phase_tokens=self.tv_second_phase_tokens,
                tv_estimator=self.tv_estimator,
                r_max=self.cfg.r_max,
                eps_tail=eps_tail_for_depth(self.cfg, depth),
                bound_form=self.cfg.bound_form,
            )

        try:
            for idx, node in enumerate(nodes):
                node.setdefault("gear_segment_id", f"batch/{depth}/{idx}")
                estimator = _make_estimator()
                result = await estimator.estimate_k_for_parent(
                    node, depth=depth, duplicate_tv_threshold=self.cfg.epsilon
                )
                # depth_batch has no graded-rollout hook: rollout-based
                # proxies raise here with a pointer to the online runtime.
                await self._prepare_proxy_fields(
                    node, default_bf=default_bf, proxy_rollout_fn=None
                )
                self._record_estimate(node, result, estimator, set())
        except Exception as exc:
            self.allocation_error_count += 1
            raise RuntimeError(
                f"VDRA pilot/scoring failed at depth {depth}; no fallback is allowed"
            ) from exc

        manager = RootQueueManager(
            queue_count=self.queue_count,
            queue_capacity=self.queue_capacity,
            timeout_seconds=self.queue_timeout_seconds,
            reserve_pool=SharedReservePool(queue_count=self.queue_count),
            n_min=self.n_min,
            use_residual_budget=self.use_residual_budget,
            policy_snapshot_id=f"depth-{depth}",
            strict_vdra=self.strict_vdra,
            rounding_strategy=self.rounding_strategy,
            rounding_seed=self.rounding_seed,
        )
        for node in nodes:
            predicted_k = max(int(node["gear_predicted_k"]), self.n_min)
            base_k = min(int(default_bf), predicted_k)
            saved_k = max(int(default_bf) - base_k, 0)
            write_node_accounting(
                node,
                default_k=int(default_bf),
                predicted_k=predicted_k,
                allocated_k=base_k,
                k_min=self.n_min,
            )
            if saved_k and self.use_residual_budget:
                await manager.reserve_pool.add(saved_k)
            if node["vdra_unmet_demand"] <= 0:
                node["vdra_additional_children_generated"] = max(base_k - int(node.get("vdra_pilot_children_reused", 0) or 0), 0)
                continue
            manager.enqueue(
                OnlineQueueItem(
                    node=node,
                    default_branch_factor=int(default_bf),
                    depth=depth,
                    policy_snapshot_id=f"depth-{depth}",
                )
            )
        for flush in await manager.drain():
            for item in flush.items:
                node = item.node
                node_id = str(node.get("gear_segment_id"))
                allocated = flush.summary.allocations.get(node_id)
                if allocated is None:
                    raise RuntimeError(f"VDRA allocation missing node {node_id}")
                write_node_accounting(
                    node,
                    default_k=int(default_bf),
                    predicted_k=int(node["vdra_predicted_k"]),
                    allocated_k=int(allocated),
                    k_min=self.n_min,
                    allocation_weight=flush.summary.weights[node_id],
                )
                node["vdra_allocation_seconds"] = flush.allocation_seconds
                node["vdra_additional_children_generated"] = max(
                    int(allocated) - int(node.get("vdra_pilot_children_reused", 0) or 0), 0
                )

    # --- sibling-local value share ------------------------------------------ #
    def _annotate_children(
        self, parent: Dict[str, Any], depth: int, children: List[Dict[str, Any]]
    ) -> None:
        for idx, child in enumerate(children):
            child.setdefault("gear_segment_id", f"{_seg_id(parent)}/{depth}/{idx}")
            child.setdefault("gear_action", "expand")

    def _share_enabled(self, children: List[Dict[str, Any]]) -> bool:
        return self.enable_share and self.scorer is not None and len(children) >= 2

    def filter_children(
        self, parent: Dict[str, Any], depth: int, default_bf: int, children: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sync entry point (SPMD path).  Async scorers run in a fresh loop."""

        self._annotate_children(parent, depth, children)
        if not self._share_enabled(children):
            return children
        try:
            asyncio.run(self._apply_local_share(children))
        except RuntimeError as exc:
            # Called from a running event loop: the caller must use
            # filter_children_async instead of the sync wrapper.
            self.share_error_count += 1
            logger.warning(
                "GEAR local share skipped (use filter_children_async inside an "
                "event loop): %r",
                exc,
            )
        except Exception as exc:
            self.share_error_count += 1
            logger.warning("GEAR local share failed: %r", exc)
        return children

    async def filter_children_async(
        self, parent: Dict[str, Any], depth: int, default_bf: int, children: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Async entry point (agent-loop path); awaits async scorers."""

        self._annotate_children(parent, depth, children)
        if not self._share_enabled(children):
            return children
        try:
            await self._apply_local_share(children)
        except Exception as exc:
            self.share_error_count += 1
            logger.warning("GEAR local share failed: %r", exc)
        return children

    async def _score_one(self, prefix: str, y: str) -> float:
        """Score ``log pi(y | prefix)`` with either a sync or an async scorer."""

        result = self.scorer.score_one(prefix, y)
        if inspect.isawaitable(result):
            return float(await result)
        return float(result)

    async def _apply_local_share(self, children: List[Dict[str, Any]]) -> None:
        # Build a small shared continuation set from the siblings' own texts.
        continuations = [c.get("text", "") for c in children]
        alpha = self.cfg.alpha
        n = len(continuations)
        radius = lvs.confidence_radius(n, alpha)
        logps_cache: Dict[int, List[float]] = {}

        async def logps_for(idx: int) -> List[float]:
            if idx not in logps_cache:
                prefix = children[idx].get("full_text", "")
                logps_cache[idx] = [
                    await self._score_one(prefix, y) for y in continuations
                ]
            return logps_cache[idx]

        for i, child in enumerate(children):
            if child.get("gear_action") != "expand":
                continue
            logps_i = await logps_for(i)
            for j in range(i):
                target = children[j]
                if target.get("gear_action") != "expand":
                    continue
                logps_j = await logps_for(j)
                tv = lvs.sampled_tv_from_logps(logps_i, logps_j)
                eta = self.cfg.epsilon / max(self.cfg.r_max, 1e-8)
                if tv + radius <= eta:
                    child["gear_action"] = "share"
                    child["gear_share_target"] = target.get("gear_segment_id")
                    child["gear_tv_m"] = float(tv)
                    break


def _seg_id(node: Dict[str, Any]) -> str:
    return str(node.get("gear_segment_id", "root"))
