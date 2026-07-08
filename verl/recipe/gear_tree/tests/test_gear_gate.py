"""CPU tests for the GEAR online gate (perplexity predict-k prune)."""

import math

from recipe.gear_tree.gear_gate import GearGate


def test_predict_k_matches_treetune_simple_formula():
    gate = GearGate(k_algorithm="simple", n_min=0, skip_near_leaf_expand=False)
    # ppl = exp(-sum_logprobs/num_tokens); k = ceil(ppl).
    node = {"sum_logprobs": -3.0, "num_tokens": 3}  # ppl = exp(1.0) ~ 2.718 -> k=3
    bf = gate.branch_factor(node, depth=1, default_bf=6)
    assert bf == min(math.ceil(math.exp(1.0)), 6)
    assert node["gear_predicted_k"] == math.ceil(math.exp(1.0))


def test_branch_factor_never_exceeds_default_and_respects_n_min():
    gate = GearGate(k_algorithm="simple", n_min=2, skip_near_leaf_expand=False)
    # Very confident node (ppl ~ 1) -> k=1, but n_min floors to 2.
    node = {"sum_logprobs": -0.01, "num_tokens": 10}
    assert gate.branch_factor(node, depth=1, default_bf=6) == 2
    # High-perplexity node -> k large, but capped at default_bf.
    node2 = {"sum_logprobs": -50.0, "num_tokens": 2}
    assert gate.branch_factor(node2, depth=1, default_bf=6) == 6


def test_root_and_near_leaf_keep_default_width():
    gate = GearGate(k_algorithm="simple", skip_near_leaf_expand=True, max_depth=3)
    node = {"sum_logprobs": -0.01, "num_tokens": 10}
    assert gate.branch_factor(node, depth=0, default_bf=6) == 6  # root
    assert gate.branch_factor(node, depth=2, default_bf=6) == 6  # near-leaf (max_depth-1)


def test_no_scorer_share_is_noop():
    gate = GearGate(enable_share=True, scorer=None)
    children = [
        {"text": "a", "full_text": "P a"},
        {"text": "b", "full_text": "P b"},
    ]
    out = gate.filter_children({"gear_segment_id": "root"}, depth=1, default_bf=2, children=children)
    assert all(c["gear_action"] == "expand" for c in out)
