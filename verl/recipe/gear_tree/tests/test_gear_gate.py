"""CPU tests for the GEAR online gate (prune / share / VDRA budget allocation)."""

import asyncio
import math

import pytest

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
    gate = GearGate(k_algorithm="simple", enable_share=True, scorer=None)
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
    """Deterministic pilot expander with exact token ids."""

    async def expand(self, *, current_node, prefix, depth, max_tokens, branch_factor):
        tag = current_node.get("tag", "n")
        base_ids = list(current_node.get("full_token_ids") or [ord(ch) for ch in prefix])
        out = []
        for i in range(int(branch_factor)):
            origin = int(current_node.get("pilot_index", i))
            token = 100 + origin
            out.append(
                {
                    "text": f" {tag}{i}",
                    "full_text": f"{prefix} {tag}{i}",
                    "full_token_ids": base_ids + [token],
                    "response_token_ids": [token],
                    "sum_logprobs": -0.5 * (i + 1),
                    "num_tokens": 1,
                    "finish_reason": "length",
                    "tag": tag,
                    "pilot_index": origin,
                }
            )
        return out


class _DispersionScorer:
    """Async scorer making node 'hot' children disagree and 'cold' agree."""

    async def score_one_tokens(self, prefix_token_ids, continuation_token_ids):
        if prefix_token_ids and prefix_token_ids[0] == 1:
            # Likelihood depends strongly on which pilot child scores which
            # continuation -> large pairwise TV -> large C_s.
            return -1.0 if prefix_token_ids[-1] == continuation_token_ids[0] else -30.0
        return -5.0  # cold node: identical likelihoods -> TV = 0

    async def score_one(self, prefix, y):
        if "hot" in prefix:
            return -1.0 if prefix.endswith(y.strip()[-1]) else -30.0
        return -5.0


def test_allocate_batch_async_moves_budget_to_high_dispersion_node():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        n_min=1,
        pilot_branch_factor=8,
        allocation_runtime="depth_batch",
        skip_near_leaf_expand=False,
    )
    assert gate.use_batch_allocation
    hot = {"full_text": "hot", "full_token_ids": [1], "tag": "h", "gear_segment_id": "hot"}
    cold = {"full_text": "cold", "full_token_ids": [2], "tag": "c", "gear_segment_id": "cold"}
    asyncio.run(gate.allocate_batch_async([hot, cold], depth=1, default_bf=4, node_expander=_PilotExpander()))

    assert hot["vdra_dispersion_C"] > cold["vdra_dispersion_C"]
    total = hot["gear_branch_allocation"] + cold["gear_branch_allocation"]
    assert total == min(8, hot["gear_predicted_k"] + cold["gear_predicted_k"])
    assert cold["gear_branch_allocation"] >= 1  # n_min floor kept
    assert hot["gear_branch_allocation"] > cold["gear_branch_allocation"]
    # branch_factor consumes the written allocation.
    assert gate.branch_factor(hot, depth=1, default_bf=4) == hot["gear_branch_allocation"]


def test_allocate_batch_async_scoring_failure_is_explicit():
    class BoomScorer:
        async def score_one(self, prefix, y):
            raise RuntimeError("boom")

    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=BoomScorer(),
        pilot_branch_factor=2,
        allocation_runtime="depth_batch",
        skip_near_leaf_expand=False,
    )
    nodes = [
        {"full_text": "a", "full_token_ids": [1], "gear_segment_id": "a"},
        {"full_text": "b", "full_token_ids": [2], "gear_segment_id": "b"},
    ]
    with pytest.raises(RuntimeError, match="no fallback"):
        asyncio.run(
            gate.allocate_batch_async(
                nodes, depth=1, default_bf=3, node_expander=_PilotExpander()
            )
        )
    assert gate.allocation_error_count == 1



class _MixedFinishPilotExpander:
    """Phase 1: one terminal + N-1 continuable pilots; phase 2: stop blocks."""

    def __init__(self):
        self.calls = []

    async def expand(self, *, current_node, prefix, depth, max_tokens, branch_factor):
        self.calls.append((prefix, branch_factor))
        if len(self.calls) == 1:
            return [
                {
                    "text": f" p{i}",
                    "full_text": f"{prefix} p{i}",
                    "sum_logprobs": -0.1 * (i + 1),
                    "num_tokens": 2,
                    "response_token_ids": [10 + i, 20 + i],
                    "full_token_ids": list(current_node.get("full_token_ids") or []) + [10 + i, 20 + i],
                    "finish_reason": "stop" if i == 0 else "length",
                }
                for i in range(int(branch_factor))
            ]
        return [
            {
                "text": f" z{len(self.calls)}_{i}",
                "full_text": f"{prefix} z{len(self.calls)}_{i}",
                "sum_logprobs": -0.2,
                "num_tokens": 3,
                "response_token_ids": [30, 31, 32],
                "full_token_ids": list(current_node.get("full_token_ids") or []) + [30, 31, 32],
                "finish_reason": "stop",
            }
            for i in range(int(branch_factor))
        ]


def test_estimate_node_async_records_shortcut_and_support_accounting():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        n_min=1,
        pilot_branch_factor=3,
        likelihood_samples_per_distribution=1,
        pilot_execution_mode="weighted_reuse",
    )
    node = {"full_text": "hot", "full_token_ids": [1], "gear_segment_id": "n0"}
    asyncio.run(
        gate.estimate_node_async(
            node, depth=1, default_bf=2, node_expander=_MixedFinishPilotExpander()
        )
    )

    assert node["vdra_pilot_children_generated"] == 3
    assert node["vdra_pilot_children_shortcut"] == 1
    assert len(node["vdra_shortcut_children"]) == 1
    assert node["vdra_shortcut_children"][0]["finish_reason"] == "stop"
    # Reused = post-pruning continuable survivors + shortcut leaves.
    assert node["vdra_pilot_children_reused"] == (
        len(node["vdra_reusable_pilot_children"]) + 1
    )
    assert node["vdra_pilot_children_discarded"] == (
        3 - node["vdra_pilot_children_reused"]
    )
    # predicted_k counts the shortcut leaf as satisfied demand.
    assert node["vdra_predicted_k"] == 1 + len(node["vdra_reusable_pilot_children"])
    # Second-phase support generation is charged to pilot overhead (2
    # continuable pilots x r=1 blocks x 3 tokens each).
    assert node["vdra_pilot_support_children_generated"] == 2
    assert node["vdra_pilot_support_generated_tokens"] == 6
    assert node["vdra_generation_request_count"] == 3 + 2
    assert node["vdra_C_total"] == node["vdra_dispersion_C"]
    assert node["vdra_C_cross"] > 0.0
    assert node["vdra_C_total"] > node["vdra_C_continuation"]
    assert len(node["vdra_cluster_id_per_pilot"]) == node["vdra_pilot_children_generated"]
    assert sum(node["vdra_cluster_size"].values()) == node["vdra_pilot_children_generated"]
    all_pilots = node["vdra_all_pilot_children"]
    assert all(pilot.get("vdra_cluster_id") is not None for pilot in all_pilots)
    representatives = node["vdra_reusable_pilot_children"] + node["vdra_shortcut_children"]
    assert sum(pilot["vdra_cluster_multiplicity"] for pilot in representatives) == 3
    assert sum(pilot["vdra_representative_weight"] for pilot in representatives) == pytest.approx(1.0)


def test_prepare_proxy_fields_computes_empirical_variance_from_rollouts():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        allocation_proxy="empirical_variance",
        pilot_branch_factor=4,
    )
    node = {"gear_segment_id": "n"}
    seen = []

    async def rollout_fn(n, count):
        seen.append(count)
        return [0.0, 1.0, 0.0, 1.0][:count]

    asyncio.run(gate._prepare_proxy_fields(node, default_bf=2, proxy_rollout_fn=rollout_fn))
    assert seen == [4]  # k0 rollouts for the empirical baseline
    assert node["vdra_empirical_reward_variance"] == pytest.approx(0.25)


def test_prepare_proxy_fields_oracle_uses_configured_rollout_count():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        allocation_proxy="oracle",
        oracle_rollouts_per_node=6,
    )
    node = {"gear_segment_id": "n"}

    async def rollout_fn(n, count):
        return [1.0] * count

    asyncio.run(gate._prepare_proxy_fields(node, default_bf=2, proxy_rollout_fn=rollout_fn))
    assert node["vdra_oracle_value_dispersion"] == 0.0


def test_rollout_proxies_error_without_online_runtime_support():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        allocation_proxy="empirical_variance",
    )
    with pytest.raises(ValueError, match="online allocation runtime"):
        asyncio.run(gate._prepare_proxy_fields({}, default_bf=2, proxy_rollout_fn=None))


def test_external_score_proxy_requires_configured_callable():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        allocation_proxy="external_score",
    )
    with pytest.raises(ValueError, match="external_score_module"):
        asyncio.run(gate._prepare_proxy_fields({}, default_bf=2, proxy_rollout_fn=None))
    gate.external_score_fn = lambda node: 0.5
    node = {}
    asyncio.run(gate._prepare_proxy_fields(node, default_bf=2, proxy_rollout_fn=None))
    assert node["vdra_external_dispersion_C"] == 0.5


def test_strict_main_config_rejects_insufficient_pilot_branch_factor():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        pilot_branch_factor=6,
        likelihood_samples_per_distribution=2,
        tv_first_phase_tokens=100,
        strict_vdra=True,
        use_residual_budget=True,
    )
    with pytest.raises(ValueError, match="pilot_branch_factor > max default"):
        gate.validate_main_config(max_default_branch_factor=6, segment_length=100)


def test_strict_main_config_rejects_reused_pilot_longer_than_segment():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        pilot_branch_factor=8,
        tv_first_phase_tokens=120,
        strict_vdra=True,
        pilot_execution_mode="weighted_reuse",
    )
    with pytest.raises(ValueError, match="pilot length"):
        gate.validate_main_config(max_default_branch_factor=6, segment_length=100)


def test_fresh_iid_allows_pilot_longer_than_segment():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        pilot_branch_factor=8,
        tv_first_phase_tokens=120,
        strict_vdra=True,
    )
    gate.validate_main_config(max_default_branch_factor=6, segment_length=100)


def test_strict_vdra_rejects_sampling_distribution_mismatch():
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=_DispersionScorer(),
        pilot_branch_factor=8,
        strict_vdra=True,
        rollout_temperature=0.7,
        rollout_top_p=1.0,
    )
    with pytest.raises(ValueError, match="temperature=1.0"):
        gate.validate_main_config(max_default_branch_factor=6, segment_length=100)
