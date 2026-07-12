"""CPU tests for the GEAR online gate (prune / share / VDRA budget allocation)."""

import asyncio
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


class _AsyncTableScorer:
    """Async scorer (LPScorer-like): identical siblings share, distinct don't."""

    def __init__(self, table):
        self.table = dict(table)
        self.calls = 0

    async def score_one(self, prefix, y):
        self.calls += 1
        return self.table[(prefix, y)]


def _identical_sibling_table():
    # Both prefixes assign identical likelihoods to both continuations.
    return {
        ("P a", "a"): -1.0,
        ("P a", "b"): -2.0,
        ("P b", "a"): -1.0,
        ("P b", "b"): -2.0,
    }


def test_async_scorer_share_triggers_via_sync_entry_point():
    # Regression for the silent-failure bug: filter_children used to call the
    # async score_one synchronously, crash on the coroutine, and swallow the
    # exception - share never fired and share_error_count stayed 0.
    scorer = _AsyncTableScorer(_identical_sibling_table())
    gate = GearGate(enable_share=True, scorer=scorer, epsilon=0.9, alpha=0.5)
    children = [
        {"text": "a", "full_text": "P a"},
        {"text": "b", "full_text": "P b"},
    ]
    out = gate.filter_children({"gear_segment_id": "root"}, depth=1, default_bf=2, children=children)
    assert gate.share_error_count == 0
    assert scorer.calls > 0
    assert out[1]["gear_action"] == "share"
    assert out[1]["gear_share_target"] == out[0]["gear_segment_id"]


def test_async_scorer_share_triggers_via_async_entry_point():
    scorer = _AsyncTableScorer(_identical_sibling_table())
    gate = GearGate(enable_share=True, scorer=scorer, epsilon=0.9, alpha=0.5)
    children = [
        {"text": "a", "full_text": "P a"},
        {"text": "b", "full_text": "P b"},
    ]
    out = asyncio.run(
        gate.filter_children_async({"gear_segment_id": "root"}, depth=1, default_bf=2, children=children)
    )
    assert gate.share_error_count == 0
    assert out[1]["gear_action"] == "share"


def test_scorer_failure_is_counted_not_swallowed():
    class BoomScorer:
        async def score_one(self, prefix, y):
            raise RuntimeError("boom")

    gate = GearGate(enable_share=True, scorer=BoomScorer(), epsilon=0.9)
    children = [
        {"text": "a", "full_text": "P a"},
        {"text": "b", "full_text": "P b"},
    ]
    out = gate.filter_children({"gear_segment_id": "root"}, depth=1, default_bf=2, children=children)
    assert gate.share_error_count == 1
    assert all(c["gear_action"] == "expand" for c in out)


class _PilotExpander:
    """Deterministic pilot expander: per-prefix continuations 'X0', 'X1', ..."""

    async def expand(self, *, current_node, prefix, depth, max_tokens, branch_factor):
        tag = current_node.get("tag", "n")
        return [
            {
                "text": f" {tag}{i}",
                "full_text": f"{prefix} {tag}{i}",
                "sum_logprobs": -0.5 * (i + 1),
                "num_tokens": 1,
                "finish_reason": "length",
            }
            for i in range(int(branch_factor))
        ]


class _DispersionScorer:
    """Async scorer making node 'hot' children disagree and 'cold' agree."""

    async def score_one(self, prefix, y):
        if "hot" in prefix:
            # Likelihood depends strongly on which pilot child scores which
            # continuation -> large pairwise TV -> large C_s.
            return -1.0 if prefix.endswith(y.strip()[-1]) else -30.0
        return -5.0  # cold node: identical likelihoods -> TV = 0


def test_allocate_batch_async_moves_budget_to_high_dispersion_node():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        n_min=1,
        n_tv_estimates=2,
        skip_near_leaf_expand=False,
    )
    assert gate.use_batch_allocation
    hot = {"full_text": "hot", "tag": "h", "gear_segment_id": "hot"}
    cold = {"full_text": "cold", "tag": "c", "gear_segment_id": "cold"}
    asyncio.run(gate.allocate_batch_async([hot, cold], depth=1, default_bf=4, node_expander=_PilotExpander()))

    assert hot["gear_reward_variance"] > cold["gear_reward_variance"]
    total = hot["gear_branch_allocation"] + cold["gear_branch_allocation"]
    assert total == 8  # budget conserved: 2 nodes x default_bf 4
    assert cold["gear_branch_allocation"] >= 1  # n_min floor kept
    assert hot["gear_branch_allocation"] > cold["gear_branch_allocation"]
    # branch_factor consumes the written allocation.
    assert gate.branch_factor(hot, depth=1, default_bf=4) == hot["gear_branch_allocation"]


def test_allocate_batch_async_scoring_failure_falls_back_to_uniform():
    class BoomScorer:
        async def score_one(self, prefix, y):
            raise RuntimeError("boom")

    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=BoomScorer(),
        n_tv_estimates=2,
        skip_near_leaf_expand=False,
    )
    nodes = [
        {"full_text": "a", "gear_segment_id": "a"},
        {"full_text": "b", "gear_segment_id": "b"},
    ]
    asyncio.run(gate.allocate_batch_async(nodes, depth=1, default_bf=3, node_expander=_PilotExpander()))
    assert gate.allocation_error_count == 1
    assert [n["gear_branch_allocation"] for n in nodes] == [3, 3]
