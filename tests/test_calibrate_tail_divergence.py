"""CPU tests for the RQ2/RQ3/RQ4 calibration aggregation (no vLLM server)."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_tail_divergence",
    Path(__file__).resolve().parent.parent / "scripts" / "calibrate_tail_divergence.py",
)
cal = importlib.util.module_from_spec(_SPEC)
sys.modules["calibrate_tail_divergence"] = cal
_SPEC.loader.exec_module(cal)


def _args(**overrides):
    base = dict(
        horizons=[8, 16],
        quantiles=[0.9, 0.99],
        delta=1e-6,
        grade=True,
        assumed_eps_tail=0.0,
        r_max=1.0,
        default_bf=4,
        n_min=1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_spearman_and_pearson_basics():
    assert cal.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert cal.spearman([1, 2, 3], [10, 20, 15]) == pytest.approx(0.5)
    assert cal.pearson([1, 1, 1], [1, 2, 3]) is None  # zero variance
    assert cal.spearman([1], [2]) is None


def test_quantile_interpolates():
    xs = [0.0, 1.0, 2.0, 3.0]
    assert cal.quantile(xs, 0.0) == 0.0
    assert cal.quantile(xs, 1.0) == 3.0
    assert cal.quantile(xs, 0.5) == pytest.approx(1.5)
    assert cal.quantile([], 0.5) is None


def test_tanh_tv_matches_ratio_identity():
    import math

    tv = cal.tanh_tv([math.log(0.2)], [math.log(0.1)])
    assert tv == pytest.approx((0.2 - 0.1) / (0.2 + 0.1))


def _record(depth, d_m_8, d_m_16, d_l, sigma2=None, k0=2):
    rec = {
        "depth": depth,
        "k0": k0,
        "pairs": [{"pair": (0, 1), "d_m": {8: d_m_8, 16: d_m_16}, "d_l": d_l}],
    }
    if sigma2 is not None:
        rec["sigma2_oracle"] = sigma2
    return rec


def test_summarize_reports_tail_quantiles_and_coverage():
    records = [
        _record(0, 0.1, 0.15, 0.2, sigma2=0.01),
        _record(1, 0.3, 0.35, 0.4, sigma2=0.04),
        _record(1, 0.5, 0.6, 0.6, sigma2=0.09),
    ]
    summary = cal.summarize(records, _args())
    h8 = summary["per_horizon"]["8"]
    # D_m and D_L are perfectly rank-aligned in this synthetic set.
    assert h8["spearman_dm_dl"] == pytest.approx(1.0)
    # Tail ratios: (0.2-0.1)/0.9, (0.4-0.3)/0.7, (0.6-0.5)/0.5.
    assert h8["eps_tail_quantiles"]["0.99"] == pytest.approx(0.2, abs=0.01)
    assert set(h8["eps_tail_by_depth"]) == {"0", "1"}
    # Coverage at the calibrated quantile must be near 1 by construction.
    assert h8["coverage_at_main_quantile"] >= 2 / 3


def test_summarize_rq4_and_allocation_regret():
    records = [
        _record(0, 0.05, 0.05, 0.05, sigma2=0.0),
        _record(0, 0.4, 0.5, 0.5, sigma2=0.05),
        _record(0, 0.8, 0.9, 0.9, sigma2=0.2),
    ]
    summary = cal.summarize(records, _args())
    rq4 = summary["rq4"]
    assert rq4["num_nodes"] == 3
    assert rq4["spearman_cs_sigma2"] == pytest.approx(1.0)
    dd = summary["direction_d"]
    assert dd["budget"] == 12
    # Oracle is optimal for J; VDRA must be no worse than uniform here.
    assert dd["J_oracle"] <= dd["J_vdra"] + 1e-9
    assert dd["J_vdra"] <= dd["J_uniform"] + 1e-9
    assert dd["regret_vdra"] >= 0.0
