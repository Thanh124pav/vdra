"""Sibling-local ValueShare utilities.

This module compares local rollout distributions directly: for two
sibling nodes x and y, collect rollout continuations generated from both nodes,
score the same continuation set under both prefixes, normalize with softmax,
and compute total variation on that sampled support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class LocalShareDecision:
    source_id: str
    target_id: str
    tv: float
    value_bound: float
    n_continuations: int
    confidence_radius: float
    eta_used: float


def stable_softmax(logps: Sequence[float]) -> np.ndarray:
    arr = np.asarray(logps, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    max_lp = float(np.max(arr[finite]))
    shifted = np.where(finite, arr - max_lp, -math.inf)
    exp_shifted = np.exp(shifted)
    denom = float(np.sum(exp_shifted))
    if denom <= 0.0 or not math.isfinite(denom):
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    return exp_shifted / denom


def sampled_tv_from_logps(logps_a: Sequence[float], logps_b: Sequence[float]) -> float:
    if len(logps_a) != len(logps_b):
        raise ValueError(f"Logprob length mismatch: {len(logps_a)} vs {len(logps_b)}")
    if len(logps_a) == 0:
        return 1.0
    pa = stable_softmax(logps_a)
    pb = stable_softmax(logps_b)
    return 0.5 * float(np.sum(np.abs(pa - pb)))


def confidence_radius(n: int, alpha: float) -> float:
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / max(alpha, 1e-12)) / (2.0 * n))


def pair_budget(width: int, fraction: float = 0.25) -> int:
    if width < 2:
        return 0
    total = width * (width - 1) // 2
    target = max(1, int(round((width * width) * fraction)))
    return min(total, target)


def select_candidate_pairs(
    sibling_ids: Sequence[str],
    cheap_scores: Sequence[float],
    budget: int,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[float, int, int]] = []
    for i in range(len(sibling_ids)):
        for j in range(i + 1, len(sibling_ids)):
            gap = abs(float(cheap_scores[i]) - float(cheap_scores[j]))
            pairs.append((gap, i, j))
    pairs.sort(key=lambda item: item[0])
    return [(i, j) for _, i, j in pairs[:budget]]
