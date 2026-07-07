import math

import pytest

from treetune.gear.thresholds import ThresholdConfig, compute_eta, compute_tau


def test_eta_from_lemma_2_4():
    cfg = ThresholdConfig(epsilon=0.02, r_max=1.0, K=10)
    assert compute_eta(cfg, delta_avg=0.0) == pytest.approx(0.02)
    # Higher delta_avg should shrink eta.
    assert compute_eta(cfg, delta_avg=0.005) == pytest.approx(0.015)


def test_eta_override_bypasses_formula():
    cfg = ThresholdConfig(epsilon=0.02, K=10, eta_override=0.07)
    assert compute_eta(cfg) == 0.07


def test_tau_dkw_band():
    cfg = ThresholdConfig(K=10, alpha=0.05)
    eta = 0.02
    expected_band = math.sqrt(math.log(2 / 0.05) / (2 * 10))
    assert compute_tau(cfg, eta) == pytest.approx(eta + expected_band)


def test_tau_no_dkw():
    cfg = ThresholdConfig(K=10, use_dkw=False)
    assert compute_tau(cfg, 0.02) == 0.02
