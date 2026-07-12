"""Parity tests: vendored ``gear_core`` must be behaviorally identical to the
original ``treetune`` modules it was copied from.

The vendored formula files are byte-identical to their sources (only import
lines were rewritten); these tests exercise the public entry points through
BOTH import roots and assert exactly-equal outputs, so any accidental drift in
the copy is caught. They require the original ``treetune`` package to still be
importable (repo root on ``sys.path``); they are skipped once treetune is
removed at the end of the migration.

Run:
    PYTHONPATH=.:verl python -m pytest verl/recipe/gear_tree/tests/test_vendor_parity.py -q
"""

import dataclasses

import numpy as np
import pytest

# --- original treetune modules (baseline) ---
treetune = pytest.importorskip("treetune.gear.budget_allocation")

from treetune.gear import budget_allocation as ba_orig
from treetune.gear import thresholds as th_orig
from treetune.gear import tv_distance as tv_orig
from treetune.gear import log_prob_matrix as lp_orig
from treetune.episode_generators import tree_update_modes as tum_orig
from treetune.tasks import math_grader as mg_orig

# --- vendored copies ---
from recipe.gear_tree.gear_core.gear import budget_allocation as ba_new
from recipe.gear_tree.gear_core.gear import thresholds as th_new
from recipe.gear_tree.gear_core.gear import tv_distance as tv_new
from recipe.gear_tree.gear_core.gear import log_prob_matrix as lp_new
from recipe.gear_tree.gear_core import tree_update_modes as tum_new
from recipe.gear_tree.gear_core.grading import math_grader as mg_new


def _seg(mod, seg_id, fast, full):
    return mod.SegmentLP(
        segment_id=seg_id, K=len(fast), m=len(full),
        fast=np.array(fast), full=np.array(full),
    )


@pytest.mark.parametrize("mode,kw", [
    ("spo", {}),
    ("treepo_original", {"treepo_global_weight": 0.25}),
    ("treerl_original", {"treerl_gamma": 0.5}),
])
def test_tree_update_values_parity(mode, kw):
    args = dict(child_reward=0.8, parent_reward=0.3, root_reward=0.1, mode=mode, **kw)
    o = tum_orig.compute_tree_update_values(**args)
    n = tum_new.compute_tree_update_values(**args)
    assert o == n


def test_thresholds_parity():
    cfg_o = th_orig.ThresholdConfig(epsilon=0.02, r_max=1.0, gamma=0.9, alpha=0.05, K=10)
    cfg_n = th_new.ThresholdConfig(epsilon=0.02, r_max=1.0, gamma=0.9, alpha=0.05, K=10)
    for delta in (0.0, 0.005, 0.01):
        eo = th_orig.compute_eta(cfg_o, delta)
        en = th_new.compute_eta(cfg_n, delta)
        assert eo == en
        assert th_orig.compute_tau(cfg_o, eo) == th_new.compute_tau(cfg_n, en)
    for tv in (0.0, 0.1, 0.5, 0.9):
        assert th_orig.tv_to_value_bound(tv, cfg_o) == th_new.tv_to_value_bound(tv, cfg_n)


def test_budget_allocation_parity():
    for tv in (0.0, 0.2, 0.7):
        assert ba_orig.simulation_lemma_gap(tv, 0.9) == ba_new.simulation_lemma_gap(tv, 0.9)
    pair_tvs = {(0, 1): 0.1, (0, 2): 0.3, (0, 3): 0.25, (1, 2): 0.4, (1, 3): 0.05, (2, 3): 0.6}
    assert ba_orig.reward_variance_from_pair_tvs(pair_tvs, n=4, gamma=0.9) == \
        ba_new.reward_variance_from_pair_tvs(pair_tvs, n=4, gamma=0.9)

    nodes = [
        {"id": "a", "gear_reward_variance": 0.30},
        {"id": "b", "gear_reward_variance": 0.05},
        {"id": "c", "gear_reward_variance": 0.50},
    ]
    o = ba_orig.allocate_branch_factors(nodes, total_budget=9, lambda_=0.02, n_min=0)
    n = ba_new.allocate_branch_factors(nodes, total_budget=9, lambda_=0.02, n_min=0)
    # AllocationSummary is a distinct class per package, so compare field values.
    assert dataclasses.asdict(o) == dataclasses.asdict(n)


def test_tv_distance_parity():
    fast_a, full_a = [-1.0, -2.0, -0.5], [-1.0, -2.0, -0.5, -3.0, -2.5]
    fast_b, full_b = [-1.2, -1.8, -0.7], [-1.2, -1.8, -0.7, -2.9, -2.2]
    a_o, b_o = _seg(lp_orig, "a", fast_a, full_a), _seg(lp_orig, "b", fast_b, full_b)
    a_n, b_n = _seg(lp_new, "a", fast_a, full_a), _seg(lp_new, "b", fast_b, full_b)
    assert tv_orig.tv_m(a_o, b_o) == tv_new.tv_m(a_n, b_n)
    assert tv_orig.avg_lp_K(a_o) == tv_new.avg_lp_K(a_n)
    assert tv_orig.avg_lp_m(a_o) == tv_new.avg_lp_m(a_n)


@pytest.mark.parametrize("given,gt", [
    ("\\frac{1}{2}", "0.5"),
    ("2", "2"),
    ("x^2+1", "1+x^2"),
    ("7", "8"),
])
def test_grade_answer_parity(given, gt):
    assert mg_orig.grade_answer(given_answer=given, ground_truth=gt) == \
        mg_new.grade_answer(given_answer=given, ground_truth=gt)
