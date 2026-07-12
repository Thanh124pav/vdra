"""TV distance and AvgLP utilities.

Default estimator (Summary.md §9, ``estimator='tanh'``):

    TV_m(a,b) = mean_i | tanh( (LP[i][a] - LP[i][b]) / 2 ) |

which depends only on log-probability ratios and stays numerically meaningful
for full-sequence log-probs (exp of which underflow to 0).

The legacy PLAN.md Def 2.3 form (``estimator='legacy_abs'``) is kept for
ablations:

    TV_m(a,b)  = 0.5 * sum_{i=1..m} | exp(LP[i][a]) - exp(LP[i][b]) |
                 + 0.5 * ( exp(delta_a) + exp(delta_b) )

Its +0.5*(exp(delta_a)+exp(delta_b)) tail accounts for the residual mass
outside the sampled support, but with sequence-level LPs the residual is ≈ 1,
so the bound saturates near 1 and carries no signal.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from treetune.gear.log_prob_matrix import SegmentLP


def avg_lp_K(row: SegmentLP) -> float:
    return row.avg_lp_K


def avg_lp_m(row: SegmentLP) -> float:
    return row.avg_lp_m


def tv_m(row_a: SegmentLP, row_b: SegmentLP, *, estimator: str = "tanh") -> float:
    if row_a.full is None or row_b.full is None:
        raise RuntimeError(
            "tv_m requires both rows to have their full m-length LP vector. "
            "Call LogProbMatrix.fill_full(...) first."
        )
    if row_a.m != row_b.m:
        raise ValueError(f"Row m mismatch: {row_a.m} vs {row_b.m}")

    if estimator == "tanh":
        diffs = np.asarray(row_a.full, dtype=np.float64) - np.asarray(
            row_b.full, dtype=np.float64
        )
        return float(np.mean(np.abs(np.tanh(diffs / 2.0))))
    if estimator != "legacy_abs":
        raise ValueError(f"Unsupported tv_m estimator: {estimator}")

    pa = np.exp(row_a.full)
    pb = np.exp(row_b.full)
    body = 0.5 * float(np.sum(np.abs(pa - pb)))
    tail = 0.5 * (math.exp(row_a.delta()) + math.exp(row_b.delta()))
    return body + tail


def avg_lp_diff_K(row_a: SegmentLP, row_b: SegmentLP) -> float:
    return abs(row_a.avg_lp_K - row_b.avg_lp_K)


def conditional_ig_lower_bound(
    row_s: SegmentLP, row_pa: SegmentLP
) -> Tuple[float, float]:
    """Pinsker-style lower bound on I(A*; Y_s | Y_pa) via TV.

    Returns (lower_bound, raw_tv). Used in PLAN Lemma 2.4 sketch:
        I(A*; Y_s | Y_pa) >= 2 * TV(s, pa)^2 - O(exp(delta_s) + exp(delta_pa))

    Keeps the legacy partial-sum TV bound because the delta-based correction
    term is only meaningful for that estimator.
    """

    tv = tv_m(row_s, row_pa, estimator="legacy_abs")
    correction = math.exp(row_s.delta()) + math.exp(row_pa.delta())
    return 2.0 * tv * tv - correction, tv
