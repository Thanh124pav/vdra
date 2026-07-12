"""Threshold derivation per PLAN Lemma 2.4.

    eta = epsilon / R_max - exp(delta_avg)
    tau(K, eta) = eta + sqrt( log(2/alpha) / (2K) )    # DKW band

`tau` is reused for both the Share check and the Prune check.  Setting
`use_dkw=False` falls back to plain eta (useful for ablations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from treetune.gear.budget_allocation import value_gap_bound


@dataclass
class ThresholdConfig:
    epsilon: float = 0.02  # acceptable value error
    r_max: float = 1.0  # max reward magnitude
    gamma: float = 0.9  # discount/simulation-lemma factor for TV->value bounds
    alpha: float = 0.05  # DKW confidence level
    K: int = 10  # fast subset size
    use_dkw: bool = True
    eta_override: Optional[float] = None  # for ablation: bypass Lemma 2.4
    # Tail-divergence correction (Summary.md §7): D_L <= D_m + (1-D_m)*eps_tail.
    # eps_tail_by_depth overrides the global value per tree depth when set.
    eps_tail: float = 0.0
    eps_tail_by_depth: Optional[Dict[int, float]] = None
    # 'linear' (R_max * TV bound, default) or 'simulation_lemma' (legacy gamma form).
    bound_form: str = "linear"


def eps_tail_for_depth(cfg: ThresholdConfig, depth: Optional[int] = None) -> float:
    """Resolve the tail-correction coefficient for a node depth."""

    if depth is not None and cfg.eps_tail_by_depth:
        by_depth = cfg.eps_tail_by_depth
        if depth in by_depth:
            return float(by_depth[depth])
        # Fall back to the deepest configured level below `depth`, else global.
        known = [d for d in by_depth if d <= depth]
        if known:
            return float(by_depth[max(known)])
    return float(cfg.eps_tail)


def compute_eta(cfg: ThresholdConfig, delta_avg: float = 0.0) -> float:
    """eta from Lemma 2.4. delta_avg = exp-mean of per-segment delta."""

    if cfg.eta_override is not None:
        return float(cfg.eta_override)
    eta = cfg.epsilon / max(cfg.r_max, 1e-8) - delta_avg
    return max(eta, 1e-6)


def compute_tau(cfg: ThresholdConfig, eta: float) -> float:
    if not cfg.use_dkw:
        return eta
    band = math.sqrt(math.log(2.0 / cfg.alpha) / (2.0 * max(cfg.K, 1)))
    return eta + band


def tv_to_value_bound(
    tv: float, cfg: ThresholdConfig, depth: Optional[int] = None
) -> float:
    """Convert a TV estimate into a conservative value-difference bound.

    Delegates to ``budget_allocation.value_gap_bound`` so the pruning path and
    the budget-allocation path share one bound (tail correction, R_max scaling
    and the [0, R_max] clamp included).  ``depth`` selects a depth-dependent
    ``eps_tail`` when ``cfg.eps_tail_by_depth`` is configured.
    """

    return value_gap_bound(
        tv,
        gamma=cfg.gamma,
        r_max=cfg.r_max,
        eps_tail=eps_tail_for_depth(cfg, depth),
        bound_form=cfg.bound_form,
    )
