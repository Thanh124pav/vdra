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
    assert tv_m(a, b) == pytest.approx(2 * 0.5 * math.exp(a.delta()), abs=1e-9)


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
