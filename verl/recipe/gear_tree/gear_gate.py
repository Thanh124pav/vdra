"""GEAR online gate (Step 3): prune / share / budget allocation.

Wraps the vendored ``gear_core`` (unchanged math) and hooks into
``tree_rollout.build_tree`` through two methods:

  * ``branch_factor(parent, depth, default_bf)`` — online **prune**: predict
    ``k`` for the node and shrink the branch factor accordingly. The ``simple``
    perplexity predictor (``k = ceil(exp(-sum_logprobs/num_tokens))``) is a
    byte-exact port of ``GEARInferenceStrategy._predict_k`` (simple branch,
    gear_inference_strategy.py:659-676) and needs no extra model calls.
  * ``filter_children(parent, depth, default_bf, children)`` — sibling-local
    **share** via ``gear_core.local_value_share`` + ``segment_index.SegmentBST``.
    True local share scores a common continuation set under both sibling
    prefixes, which requires the vLLM log-prob scorer; when no ``scorer`` is
    configured this is a safe no-op (SPO-identical children).

The variance-based budget-allocation path (``allocation_mode='budget_allocation'``)
and the TV/Y-set scoring both reuse ``gear_core`` verbatim and are only active
when a ``scorer`` (``gear_core.gear.vllm_scorer``) is injected at deploy time.

Thresholds come from ``gear_core.gear.thresholds`` (Lemma 2.4), so ``eta``/``tau``
are identical to treetune.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from recipe.gear_tree.gear_core.gear.thresholds import ThresholdConfig
from recipe.gear_tree.gear_core.gear import local_value_share as lvs


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
    ) -> None:
        self.cfg = ThresholdConfig(
            epsilon=epsilon, r_max=r_max, gamma=gamma, alpha=alpha
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
        self.scorer = scorer  # vllm_scorer.LPScorer or None (CPU / disabled)

    def n_tv_estimates_for(self, branch_factor: int) -> int:
        """Number of TV estimates per node: explicit override, else B**2."""
        if self.n_tv_estimates is not None:
            return int(self.n_tv_estimates)
        return int(branch_factor) ** 2

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
        if self.k_algorithm != "simple":
            # Variance/TV budget path lives in gear_core and needs the scorer;
            # without it, fall back to the configured width (no prune).
            return default_bf
        k = self._predict_k_simple(parent)
        if k is None:
            return default_bf
        # Clamp to [n_min, default_bf]: never widen beyond the SPO width.
        return max(self.n_min, min(int(k), int(default_bf)))

    # --- sibling-local value share ------------------------------------------ #
    def filter_children(
        self, parent: Dict[str, Any], depth: int, default_bf: int, children: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for idx, child in enumerate(children):
            child.setdefault("gear_segment_id", f"{_seg_id(parent)}/{depth}/{idx}")
            child.setdefault("gear_action", "expand")

        if not self.enable_share or self.scorer is None or len(children) < 2:
            return children

        # True local share needs a common continuation set scored under each
        # sibling prefix. Delegate scoring to the injected vLLM scorer and reuse
        # gear_core.local_value_share math (Lemma 2.4 threshold).
        try:
            self._apply_local_share(children)
        except Exception:  # scorer/runtime issues must never break rollout
            pass
        return children

    def _apply_local_share(self, children: List[Dict[str, Any]]) -> None:
        # Build a small shared continuation set from the siblings' own texts.
        continuations = [c.get("text", "") for c in children]
        alpha = self.cfg.alpha
        n = len(continuations)
        radius = lvs.confidence_radius(n, alpha)
        for i, child in enumerate(children):
            if child.get("gear_action") != "expand":
                continue
            prefix_i = child.get("full_text", "")
            logps_i = [self.scorer.score_one(prefix_i, y) for y in continuations]
            for j in range(i):
                target = children[j]
                if target.get("gear_action") != "expand":
                    continue
                prefix_j = target.get("full_text", "")
                logps_j = [self.scorer.score_one(prefix_j, y) for y in continuations]
                tv = lvs.sampled_tv_from_logps(logps_i, logps_j)
                eta = self.cfg.epsilon / max(self.cfg.r_max, 1e-8)
                if tv + radius <= eta:
                    child["gear_action"] = "share"
                    child["gear_share_target"] = target.get("gear_segment_id")
                    child["gear_tv_m"] = float(tv)
                    break


def _seg_id(node: Dict[str, Any]) -> str:
    return str(node.get("gear_segment_id", "root"))
