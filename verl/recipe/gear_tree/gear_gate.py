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
    ``node['gear_branch_allocation']``.
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
        k_algorithm: str = "simple",
        n_min: int = 0,
        budget_lambda: float = 0.0,
        n_tv_estimates: Optional[int] = None,
        root_allocation: bool = True,
        skip_near_leaf_expand: bool = True,
        max_depth: Optional[int] = None,
        enable_share: bool = True,
        scorer: Any = None,
        # --- VDRA additions -------------------------------------------------
        eps_tail: float = 0.0,
        eps_tail_by_depth: Optional[Mapping[int, float]] = None,
        bound_form: str = "linear",
        tv_estimator: str = "tanh",
        tv_first_phase_tokens: int = 120,
        tv_second_phase_tokens: int = 60,
        queue_count: int = 1,
        queue_timeout_seconds: float = 0.0,
        use_residual_budget: bool = True,
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
        self.n_min = int(n_min)
        # Default lambda = 0 (see budget_allocation: sqrt(sigma^2 - lambda)).
        self.budget_lambda = float(budget_lambda)
        # None => auto = branch_factor**2 (set per-node in n_tv_estimates_for).
        self.n_tv_estimates = n_tv_estimates
        self.root_allocation = bool(root_allocation)
        self.skip_near_leaf_expand = bool(skip_near_leaf_expand)
        self.max_depth = max_depth
        self.enable_share = bool(enable_share)
        self.scorer = scorer  # lp_scorer.LPScorer (async), EngineLPScorer (sync), or None
        self.tv_estimator = tv_estimator
        self.tv_first_phase_tokens = int(tv_first_phase_tokens)
        self.tv_second_phase_tokens = int(tv_second_phase_tokens)
        self.queue_count = max(int(queue_count), 1)
        self.queue_timeout_seconds = max(float(queue_timeout_seconds), 0.0)
        self.use_residual_budget = bool(use_residual_budget)
        self.share_error_count = 0
        self.allocation_error_count = 0

    # --- capabilities -------------------------------------------------------- #
    @property
    def use_batch_allocation(self) -> bool:
        """True when the VDRA depth-batched allocation path should run."""
        return self.k_algorithm == "budget_allocation" and self.scorer is not None

    def n_tv_estimates_for(self, branch_factor: int) -> int:
        """Number of pilot children per node: explicit override, else B."""
        if self.n_tv_estimates is not None:
            return int(self.n_tv_estimates)
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
        # Batch allocation writes gear_branch_allocation via allocate_batch_async.
        allocated = parent.get("gear_branch_allocation")
        if self.use_batch_allocation and allocated is not None:
            return max(int(allocated), 0)
        if self.k_algorithm != "simple":
            return default_bf
        k = self._predict_k_simple(parent)
        if k is None:
            return default_bf
        # Clamp to [n_min, default_bf]: never widen beyond the SPO width.
        return max(self.n_min, min(int(k), int(default_bf)))

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
        (``gear_reward_variance``, ``gear_pair_tvs``, ``gear_predicted_k``)
        onto each node.  On scorer failure the nodes keep the uniform width.
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
                node["gear_branch_allocation"] = int(default_bf)
            return

        estimator = ConditionalTVEstimator(
            scorer=self.scorer,
            node_expander=node_expander,
            gamma=self.cfg.gamma,
            mode="subnode",
            n_tv_estimates=self.n_tv_estimates_for(default_bf),
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
                result = await estimator.estimate_k_for_parent(
                    node, depth=depth, duplicate_tv_threshold=self.cfg.epsilon
                )
                node["gear_reward_variance"] = float(result.reward_variance)
                node["gear_predicted_k"] = int(result.predicted_k)
                node["gear_pair_tvs"] = {
                    f"{i},{j}": float(tv) for (i, j), tv in result.pair_tvs.items()
                }
                node["gear_prob_matrix"] = result.prob_matrix
        except Exception as exc:  # pilot scoring failed: keep uniform widths
            self.allocation_error_count += 1
            logger.warning(
                "GEAR batch allocation scoring failed at depth %d "
                "(falling back to uniform width %d): %r",
                depth,
                default_bf,
                exc,
            )
            for node in nodes:
                node["gear_branch_allocation"] = int(default_bf)
            return

        manager = RootQueueManager(
            queue_count=self.queue_count,
            timeout_seconds=self.queue_timeout_seconds,
            reserve_pool=SharedReservePool(queue_count=self.queue_count),
            lambda_=self.budget_lambda,
            n_min=self.n_min,
            use_residual_budget=self.use_residual_budget,
        )
        for node in nodes:
            manager.enqueue(
                OnlineQueueItem(
                    node=node, default_branch_factor=int(default_bf), depth=depth
                )
            )
        for flush in await manager.drain():
            for item in flush.items:
                node = item.node
                node_id = str(node.get("gear_segment_id"))
                allocated = flush.summary.allocations.get(node_id)
                node["gear_branch_allocation"] = (
                    int(allocated) if allocated is not None else int(default_bf)
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
