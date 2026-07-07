import pytest

from treetune.gear.local_value_share import (
    confidence_radius,
    pair_budget,
    sampled_tv_from_logps,
    select_candidate_pairs,
    stable_softmax,
)


def test_stable_softmax_handles_large_logprobs():
    probs = stable_softmax([1000.0, 999.0, -1000.0])
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]


def test_sampled_tv_normalizes_local_support():
    # Raw exp would overflow here; the sampled-set TV should remain finite.
    tv = sampled_tv_from_logps([1000.0, 999.0], [999.0, 1000.0])
    assert 0.0 < tv < 1.0


def test_sampled_tv_identical_distributions_zero():
    assert sampled_tv_from_logps([-100.0, -101.0], [-100.0, -101.0]) == pytest.approx(0.0)


def test_confidence_radius_decreases_with_support_size():
    assert confidence_radius(100, 0.05) < confidence_radius(10, 0.05)


def test_pair_budget_matches_quarter_width_squared():
    assert pair_budget(8, 0.25) == 16
    assert pair_budget(2, 0.25) == 1


def test_select_candidate_pairs_prefers_close_cheap_scores():
    pairs = select_candidate_pairs(["a", "b", "c"], [0.0, 10.0, 0.1], budget=1)
    assert pairs == [(0, 2)]
