import math

import pytest

from treetune.gear.thresholds import (
    ThresholdConfig,
    compute_eta,
    compute_tau,
    eps_tail_for_depth,
    tv_to_value_bound,
)


def test_tv_to_value_bound_linear_default_is_clamped_tv():
    cfg = ThresholdConfig(r_max=1.0)
    assert tv_to_value_bound(0.3, cfg) == pytest.approx(0.3)
    # A value-difference bound can never exceed r_max.
    assert tv_to_value_bound(0.9, ThresholdConfig(r_max=0.5)) == pytest.approx(0.45)
    cfg_legacy = ThresholdConfig(r_max=1.0, gamma=0.9, bound_form="simulation_lemma")
    assert tv_to_value_bound(1.0, cfg_legacy) == pytest.approx(1.0)  # clamped from ~8.18


def test_tv_to_value_bound_applies_tail_correction():
    cfg = ThresholdConfig(r_max=1.0, eps_tail=0.5)
    assert tv_to_value_bound(0.3, cfg) == pytest.approx(0.3 + 0.7 * 0.5)


def test_eps_tail_for_depth_prefers_depth_table():
    cfg = ThresholdConfig(eps_tail=0.1, eps_tail_by_depth={0: 0.4, 2: 0.2})
    assert eps_tail_for_depth(cfg, None) == pytest.approx(0.1)
    assert eps_tail_for_depth(cfg, 0) == pytest.approx(0.4)
    # Depth 1 falls back to the deepest configured level below it.
    assert eps_tail_for_depth(cfg, 1) == pytest.approx(0.4)
    assert eps_tail_for_depth(cfg, 2) == pytest.approx(0.2)
    assert eps_tail_for_depth(cfg, 5) == pytest.approx(0.2)
    # No table: global value everywhere.
    assert eps_tail_for_depth(ThresholdConfig(eps_tail=0.3), 4) == pytest.approx(0.3)


def test_tv_to_value_bound_uses_depth_dependent_eps_tail():
    cfg = ThresholdConfig(r_max=1.0, eps_tail=0.0, eps_tail_by_depth={1: 0.5})
    assert tv_to_value_bound(0.2, cfg, depth=0) == pytest.approx(0.2)
    assert tv_to_value_bound(0.2, cfg, depth=1) == pytest.approx(0.2 + 0.8 * 0.5)


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
