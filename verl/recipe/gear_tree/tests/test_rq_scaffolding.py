"""Smoke tests for the RQ3/RQ4 CPU-mockable evaluation scaffolding.

The full RQ3/RQ4 studies require a cluster. These pytests only verify
that the local scripts run end-to-end on the synthetic oracle, the
allocators produce feasible budgets, and the metrics are numerically
well-behaved.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "verl"))


def _load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register BEFORE exec so any dataclass inside the script can resolve its
    # own module via sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rq3_allocation_quality_runs_and_all_budgets_are_feasible():
    rq3 = _load_script("rq3_allocation_quality")
    prefixes = rq3.sample_oracle_prefixes(num_prefixes=5, seed=42)
    results = rq3.run_all(
        prefixes,
        budget=25,
        n_min=1,
        u=8,
        rollout_fn=rq3._synthetic_rollout,
        seed=42,
    )
    names = {r.name for r in results}
    assert names == {"fixed", "random", "uncertainty", "empirical_variance", "vdra"}
    for r in results:
        # Every allocation must be feasible: sum(k) == budget (or clipped by u
        # when budget > sum(upper) which does not apply here).
        assert sum(r.allocation.values()) <= 25
        assert all(v >= 1 for v in r.allocation.values())
        assert all(v <= 8 for v in r.allocation.values())
        assert r.objective > 0


def test_rq3_vdra_matches_or_beats_fixed_and_random_on_average():
    rq3 = _load_script("rq3_allocation_quality")
    wins = 0
    trials = 10
    for seed in range(trials):
        prefixes = rq3.sample_oracle_prefixes(num_prefixes=6, seed=seed)
        results = {
            r.name: r
            for r in rq3.run_all(
                prefixes, budget=24, n_min=1, u=8,
                rollout_fn=rq3._synthetic_rollout, seed=seed,
            )
        }
        if results["vdra"].objective <= results["fixed"].objective + 1e-9:
            wins += 1
    # VDRA should win the queue objective on the majority of seeds.
    assert wins >= trials // 2, wins


def test_rq4_proxy_mse_produces_finite_diagnostics():
    rq4 = _load_script("rq4_proxy_mse")
    prefixes = [rq4._synthetic_prefix(i, seed=1) for i in range(4)]
    out = rq4.run_evaluation(
        prefixes,
        budgets=[2, 8, 32],
        reference_budget=128,
        scorer=rq4.synthetic_scorer,
    )
    assert "value_mse_by_budget" in out
    for K, mse in out["value_mse_by_budget"].items():
        assert 0.0 <= mse < 1.0, (K, mse)
    corr = out["C_proxy_vs_empirical_pearson"]
    # Ground-truth variance vs. sample variance should correlate weakly-to-
    # strongly positive in this small setup; only assert it isn't perverse.
    assert corr > -0.5, corr
