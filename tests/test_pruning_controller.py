"""Semantics of the simulation-lemma pruning trace (Summary.md §8/§10).

Pruning must be certified by the *bound*: a pair is a prune candidate only
when value_upper_bound <= epsilon (the bound proves the two child values are
within the acceptable error).  The old check ``value_gap <= upper_bound`` is
kept as the ``bound_holds`` diagnostic for the oracle audit.
"""

import pytest

from treetune.gear.pruning_controller import (
    summarize_records,
    trace_records_from_matrices,
)
from treetune.gear.thresholds import ThresholdConfig


def _records(tv_by_pair, value_gaps=None, cfg=None):
    cfg = cfg or ThresholdConfig(epsilon=0.05, r_max=1.0)
    n = max(max(pair) for pair in tv_by_pair) + 1
    prob_matrix = [[0.0] * n for _ in range(n)]
    return trace_records_from_matrices(
        node_id="node",
        depth=1,
        default_branch_factor=n,
        predicted_k=n,
        prob_matrix=prob_matrix,
        pair_tvs=tv_by_pair,
        threshold_cfg=cfg,
        value_gaps=value_gaps,
    )


def test_prune_candidate_requires_certified_small_bound():
    records = _records({(0, 1): 0.01, (0, 2): 0.6, (1, 2): 0.04})
    by_pair = {rec.pair: rec for rec in records}
    # Linear bound = TV; epsilon = 0.05.
    assert by_pair["0,1"].prune_candidate is True
    assert by_pair["1,2"].prune_candidate is True
    assert by_pair["0,2"].prune_candidate is False
    assert by_pair["0,2"].keep is True


def test_large_bound_never_becomes_prune_candidate_even_if_gap_within_bound():
    # Old inverted logic: gap (0.3) <= bound (0.6) => prune.  That merely says
    # the bound holds; the pair must be kept.
    records = _records({(0, 1): 0.6}, value_gaps={(0, 1): 0.3})
    rec = records[0]
    assert rec.value_upper_bound == pytest.approx(0.6)
    assert rec.prune_candidate is False
    assert rec.bound_holds is True


def test_bound_holds_diagnostic_detects_violations():
    records = _records({(0, 1): 0.1}, value_gaps={(0, 1): 0.5})
    rec = records[0]
    assert rec.bound_holds is False  # gap 0.5 > bound 0.1: proxy failed here
    assert rec.prune_candidate is False


def test_summarize_counts_prune_candidates():
    records = _records({(0, 1): 0.01, (0, 2): 0.9})
    summary = summarize_records(records)
    assert summary["num_prune_candidates"] == 1
    assert summary["num_available_records"] == 2
