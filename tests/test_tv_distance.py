import math

import pytest

from treetune.gear.log_prob_matrix import LogProbMatrix
from treetune.gear.tv_distance import (
    avg_lp_diff_K,
    conditional_ig_lower_bound,
    tv_m,
)


def make_row(M: LogProbMatrix, sid: str, full):
    M.add_row(sid, full[: M.K])
    M.fill_full(sid, full[M.K :])
    return M.get(sid)


def test_tv_m_identical_rows_zero():
    M = LogProbMatrix(K=2, m=4)
    r = [math.log(0.3), math.log(0.3), math.log(0.2), math.log(0.1)]
    a = make_row(M, "a", r)
    b = make_row(M, "b", r)
    # tanh estimator: identical log-prob rows give exactly zero TV.
    assert tv_m(a, b) == pytest.approx(0.0, abs=1e-12)
    # Legacy partial-sum bound keeps its residual-mass tail.
    assert tv_m(a, b, estimator="legacy_abs") == pytest.approx(
        2 * 0.5 * math.exp(a.delta()), abs=1e-9
    )


def test_tv_m_tanh_stays_informative_for_sequence_level_logprobs():
    # Full-sequence log-probs (~-60) underflow exp(); the legacy estimator's
    # body degenerates to ~0 while its tail saturates to ~1.  The tanh
    # estimator still separates near-identical from different rows.
    M = LogProbMatrix(K=1, m=3)
    close_a = make_row(M, "ca", [-60.0, -61.0, -59.5])
    close_b = make_row(M, "cb", [-60.1, -60.9, -59.4])
    far_a = make_row(M, "fa", [-60.0, -61.0, -59.5])
    far_b = make_row(M, "fb", [-80.0, -40.0, -75.0])
    tv_close = tv_m(close_a, close_b)
    tv_far = tv_m(far_a, far_b)
    assert 0.0 <= tv_close < 0.1
    assert tv_far > 0.9
    # Legacy form cannot distinguish the two cases (both ≈ 1 via the tail).
    legacy_close = tv_m(close_a, close_b, estimator="legacy_abs")
    legacy_far = tv_m(far_a, far_b, estimator="legacy_abs")
    assert legacy_close == pytest.approx(1.0, abs=1e-6)
    assert legacy_far == pytest.approx(1.0, abs=1e-6)


def test_tv_m_disjoint_rows_close_to_one():
    M = LogProbMatrix(K=2, m=4)
    a = make_row(M, "a", [math.log(0.95), math.log(0.04), -50.0, -50.0])
    b = make_row(M, "b", [-50.0, -50.0, math.log(0.95), math.log(0.04)])
    assert tv_m(a, b) > 0.85


def test_avg_lp_diff_K():
    M = LogProbMatrix(K=2, m=2)
    a = M.add_row("a", [-1.0, -2.0])
    b = M.add_row("b", [-1.5, -1.5])
    assert avg_lp_diff_K(a, b) == pytest.approx(0.0)


def test_ig_lower_bound_nonnegative_when_tv_large():
    M = LogProbMatrix(K=2, m=4)
    a = make_row(M, "a", [math.log(0.9), math.log(0.04), math.log(0.03), math.log(0.02)])
    b = make_row(M, "b", [math.log(0.02), math.log(0.03), math.log(0.04), math.log(0.9)])
    bound, tv = conditional_ig_lower_bound(a, b)
    assert tv > 0.5
    assert bound > 0
