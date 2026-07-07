import math

import numpy as np
import pytest

from treetune.gear.log_prob_matrix import LogProbMatrix


def test_add_and_full_fill():
    M = LogProbMatrix(K=3, m=5)
    fast = [-1.0, -2.0, -3.0]
    row = M.add_row("s0", fast, prefix="prefix")
    assert row.avg_lp_K == pytest.approx(np.mean(fast))

    M.fill_full("s0", [-4.0, -5.0])
    row = M.get("s0")
    assert row.full is not None
    assert row.avg_lp_m == pytest.approx(np.mean([-1, -2, -3, -4, -5]))


def test_delta_residual():
    M = LogProbMatrix(K=2, m=2)
    # Heavy mass on first answer, residual = small.
    M.add_row("s0", [math.log(0.6), math.log(0.3)])
    row = M.get("s0")
    # delta = log(1 - (0.6 + 0.3)) = log(0.1)
    assert row.delta() == pytest.approx(math.log(0.1), abs=1e-6)


def test_validates_dimensions():
    M = LogProbMatrix(K=2, m=4)
    with pytest.raises(ValueError):
        M.add_row("s0", [-1.0, -2.0, -3.0])
    M.add_row("s0", [-1.0, -2.0])
    with pytest.raises(ValueError):
        M.fill_full("s0", [-3.0])
