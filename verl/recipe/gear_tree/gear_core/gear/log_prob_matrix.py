"""Log-probability matrix used by GEAR triggers.

Implements PLAN.md Def 2.1 / 2.2:

    LP[i][s] = log pi_theta(y_i | traj(s))
    delta_s  = log( 1 - sum_i exp(LP[i][s]) )

`LP` is stored only for the K fast indices first; the remaining m-K columns
are filled lazily when a Share or Prune trigger needs the full vector.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SegmentLP:
    """Per-segment log-prob row."""

    segment_id: str
    K: int
    m: int
    fast: np.ndarray
    full: Optional[np.ndarray] = None
    prefix: Optional[str] = None  # traj(s) used to score this row

    @property
    def avg_lp_K(self) -> float:
        return float(np.mean(self.fast))

    @property
    def has_full(self) -> bool:
        return self.full is not None

    @property
    def avg_lp_m(self) -> float:
        if self.full is None:
            raise RuntimeError(f"Full LP vector not computed for {self.segment_id}")
        return float(np.mean(self.full))

    def delta(self) -> float:
        """log( 1 - sum_i exp(LP[i]) ) using the full vector if available."""

        vec = self.full if self.full is not None else self.fast
        log_sum = _logsumexp(vec)
        # Underflow guard: if sum_i exp(LP[i]) >= 1, treat residual as ~0.
        x = math.exp(min(log_sum, 0.0))
        residual = max(1.0 - x, 1e-12)
        return math.log(residual)


def _logsumexp(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return -math.inf
    m = float(np.max(arr))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(arr - m))))


class LogProbMatrix:
    """Thread-safe registry of SegmentLP rows.

    Designed as a global per-problem store: the inference strategy creates one
    instance for each tree and discards it when the tree is finished.
    """

    def __init__(self, K: int, m: int):
        if not (0 < K <= m):
            raise ValueError(f"Require 0 < K <= m, got K={K} m={m}")
        self.K = K
        self.m = m
        self._rows: Dict[str, SegmentLP] = {}
        self._lock = threading.Lock()

    def add_row(
        self,
        segment_id: str,
        fast: Sequence[float],
        prefix: Optional[str] = None,
    ) -> SegmentLP:
        if len(fast) != self.K:
            raise ValueError(f"Expected fast vector of length K={self.K}, got {len(fast)}")
        row = SegmentLP(
            segment_id=segment_id,
            K=self.K,
            m=self.m,
            fast=np.asarray(fast, dtype=np.float64),
            prefix=prefix,
        )
        with self._lock:
            self._rows[segment_id] = row
        return row

    def get(self, segment_id: str) -> SegmentLP:
        with self._lock:
            return self._rows[segment_id]

    def has(self, segment_id: str) -> bool:
        with self._lock:
            return segment_id in self._rows

    def fill_full(self, segment_id: str, tail: Sequence[float]) -> SegmentLP:
        """Append the K+1 .. m logprobs into the row's full vector.

        `tail` must have length m - K and represent indices K..m-1.
        Idempotent: if the row already has a full vector, do nothing.
        """

        expected = self.m - self.K
        if len(tail) != expected:
            raise ValueError(f"Expected tail length {expected}, got {len(tail)}")
        with self._lock:
            row = self._rows[segment_id]
            if row.full is None:
                row.full = np.concatenate([row.fast, np.asarray(tail, dtype=np.float64)])
        return row

    def avg_delta(self) -> float:
        """exp-mean of delta over all rows that have a full vector."""

        with self._lock:
            rows = [r for r in self._rows.values() if r.has_full]
        if not rows:
            return 0.0
        return float(np.mean([math.exp(r.delta()) for r in rows]))

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)
