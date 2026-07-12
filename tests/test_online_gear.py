import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import _jsonnet
import pytest

from treetune.gear.online_budget import OnlineQueueItem, RootQueueManager, SharedReservePool
from treetune.gear.thresholds import ThresholdConfig
from treetune.inference_strategies.gear_inference_strategy import GEARInferenceStrategy


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class DepthBranchFactor:
    def __init__(self, by_depth):
        self.by_depth = dict(by_depth)

    def __call__(self, node):
        return self.by_depth.get(int(node.get("depth", 0)), self.by_depth.get("default", 1))


class FakeExpander:
    program_kwargs = {"logprobs": 1}

    def __init__(self, branch_factors=None, *, omit_logprobs=False, ppl=2.0):
        self.branch_factor_strategy = DepthBranchFactor(branch_factors or {0: 1, 1: 3, "default": 1})
        self.calls = []
        self.omit_logprobs = omit_logprobs
        self.ppl = float(ppl)

    async def expand(self, *, current_node, prefix, depth, max_tokens=None, branch_factor=None):
        self.calls.append({"depth": depth, "branch_factor": branch_factor, "prefix": prefix})
        nodes = []
        for idx in range(int(branch_factor or self.branch_factor_strategy(current_node))):
            finish_reason = "length" if depth == 0 else "stop"
            node = {
                "text": f" n{depth}-{idx}",
                "full_text": f"{prefix} n{depth}-{idx}",
                "finish_reason": finish_reason,
                "depth": depth + 1,
            }
            if not self.omit_logprobs:
                node["sum_logprobs"] = -math.log(self.ppl)
                node["num_tokens"] = 1
            nodes.append(node)
        return nodes


class NoOverrideExpander:
    program_kwargs = {"logprobs": 1}

    async def expand(self, current_node, prefix, depth):
        return []


class NoLogprobExpander(FakeExpander):
    program_kwargs = {"logprobs": 0}


class FakeClient:
    async def prompt_logprobs(self, prompt):
        return [-0.1]

    async def completion_with_token_entropies(self, prompt, *, max_tokens, temperature=0.7, top_logprobs=5):
        return " token", [0.1], [" token"]


def make_strategy(
    expander,
    *,
    generation_mode="single_request",
    allocation_mode="budget_allocation",
    use_residual_budget=True,
):
    strategy = object.__new__(GEARInferenceStrategy)
    strategy.node_expander = expander
    strategy.gear_k_algorithm = "simple"
    strategy.gear_generation_mode = generation_mode
    strategy.gear_budget_queue_count = 2
    strategy.gear_budget_queue_capacity = 8
    strategy.gear_budget_queue_timeout_seconds = 999.0
    strategy.gear_budget_lambda = 0.02
    strategy.gear_n_min = 0
    strategy.gear_pilot_branch_factor = 2
    strategy.gear_likelihood_samples_per_distribution = 1
    strategy.gear_tv_subnode_max_tokens = 5
    strategy.gear_tv_second_phase_tokens = 3
    strategy.gear_tv_includes_half_factor = True
    strategy.gear_strict_vdra = True
    strategy.gear_invalid_support_policy = "error"
    strategy.gear_budget_mode = "fixed_main"
    strategy.gear_allocation_proxy = "vdra"
    strategy.gear_tv_estimator = "tanh"
    strategy.gear_budget_overhead_mode = "flexible"
    strategy.gear_allocation_mode = allocation_mode
    strategy.gear_use_residual_budget = use_residual_budget
    strategy.gear_skip_near_leaf_expand = False
    strategy.gear_root_allocation = False
    strategy.cfg_thresholds = ThresholdConfig(epsilon=0.02, gamma=0.9)
    strategy.M = 8
    strategy.tokenizer = lambda text: type("Tok", (), {"input_ids": list(range(len(text.split())))})()
    strategy._ensure_lp_client = lambda: FakeClient()

    def reward_function(*, query, response, dataset_instance):
        return 1.0, {}

    strategy.reward_function = reward_function
    return strategy


def test_generation_mode_single_request_vs_replicated_requests():
    single = make_strategy(FakeExpander(), generation_mode="single_request")
    single_tree = asyncio.run(single._construct_budget_allocated_tree("Q", max_depth=2))

    replicated = make_strategy(FakeExpander(), generation_mode="replicated_requests")
    replicated_tree = asyncio.run(replicated._construct_budget_allocated_tree("Q", max_depth=2))

    assert single_tree["gear_generation_mode"] == "single_request"
    assert replicated_tree["gear_generation_mode"] == "replicated_requests"
    assert single_tree["gear_generation_request_count"] == 2
    assert replicated_tree["gear_generation_request_count"] == 3
    assert single_tree["gear_direct_expand_count"] == 1
    assert replicated_tree["gear_direct_expand_count"] == 1


def test_skip_near_leaf_expand_uses_default_branch_factor_without_k_prediction():
    strategy = make_strategy(FakeExpander(branch_factors={0: 1, 1: 3, "default": 1}))
    strategy.gear_skip_near_leaf_expand = True

    tree = asyncio.run(strategy._construct_budget_allocated_tree("Q", max_depth=2))

    near_leaf = tree["children"][0]
    assert near_leaf["gear_near_leaf_skip"] is True
    assert near_leaf["gear_predicted_k"] == 3
    assert near_leaf["gear_allocated_branch_factor"] == 3
    assert len(near_leaf["children"]) == 3
    assert tree["gear_k_prediction_count"] == 0
    assert tree["gear_allocated_branch_factor_by_depth"][1] == 3
    assert tree["gear_reserve_contributed"] == 0


def test_gear_requires_branch_factor_override_and_logprobs():
    with pytest.raises(RuntimeError, match="branch_factor"):
        make_strategy(NoOverrideExpander())._validate_gear_generation_contract()
    with pytest.raises(RuntimeError, match="logprobs: 1"):
        make_strategy(NoLogprobExpander())._validate_gear_generation_contract()


def test_generated_nodes_must_have_logprob_metadata():
    strategy = make_strategy(FakeExpander(omit_logprobs=True))
    with pytest.raises(RuntimeError, match="sum_logprobs and num_tokens"):
        asyncio.run(strategy._construct_budget_allocated_tree("Q", max_depth=2))


def test_queue_flush_draws_reserve_share_and_uses_largest_remainder():
    async def go():
        reserve = SharedReservePool(queue_count=2)
        await reserve.add(3)
        manager = RootQueueManager(
            queue_count=2,
            queue_capacity=1,
            timeout_seconds=0.0,
            reserve_pool=reserve,
            lambda_=0.0,
        )
        nodes = [
            {"gear_segment_id": "a", "gear_reward_variance": 1.0, "vdra_predicted_k": 3},
            {"gear_segment_id": "b", "gear_reward_variance": 1.0, "vdra_predicted_k": 3},
        ]
        for node in nodes:
            manager.enqueue(OnlineQueueItem(node=node, default_branch_factor=1, depth=1))
        return await manager.flush_ready(), reserve

    results, reserve = asyncio.run(go())
    assert sum(result.reserve_draw for result in results) == 3
    assert reserve.value == 0
    allocated = {}
    for result in results:
        allocated.update(result.summary.allocations)
    assert sum(allocated.values()) == 5
    assert set(allocated) == {"a", "b"}


def test_queue_flush_can_ignore_residual_reserve():
    async def go():
        reserve = SharedReservePool(queue_count=2)
        await reserve.add(3)
        manager = RootQueueManager(
            queue_count=2,
            queue_capacity=1,
            timeout_seconds=0.0,
            reserve_pool=reserve,
            lambda_=0.0,
            use_residual_budget=False,
        )
        nodes = [
            {"gear_segment_id": "a", "gear_reward_variance": 1.0, "vdra_predicted_k": 3},
            {"gear_segment_id": "b", "gear_reward_variance": 1.0, "vdra_predicted_k": 3},
        ]
        for node in nodes:
            manager.enqueue(OnlineQueueItem(node=node, default_branch_factor=1, depth=1))
        return await manager.flush_ready(), reserve

    results, reserve = asyncio.run(go())
    assert sum(result.reserve_draw for result in results) == 0
    assert reserve.value == 3
    assert reserve.consumed == 0
    allocated = {}
    for result in results:
        allocated.update(result.summary.allocations)
    assert sum(allocated.values()) == 2
    assert set(allocated) == {"a", "b"}


def test_prune_only_never_queues_or_consumes_reserve_and_prunes_branch_factor():
    strategy = make_strategy(
        FakeExpander(branch_factors={0: 1, 1: 3, "default": 1}, ppl=2.0),
        allocation_mode="prune_only",
        use_residual_budget=False,
    )

    tree = asyncio.run(strategy._construct_budget_allocated_tree("Q", max_depth=2))

    child = tree["children"][0]
    assert tree["gear_allocation_mode"] == "prune_only"
    assert tree["gear_use_residual_budget"] is False
    assert child["gear_predicted_k"] == 2
    assert child["gear_allocated_branch_factor"] == 2
    assert len(child["children"]) == 2
    assert tree["gear_queued_node_count"] == 0
    assert tree["gear_queue_flush_count"] == 0
    assert tree["gear_reserve_consumed"] == 0
    assert tree["gear_reserve_contributed"] == 0
    assert tree["gear_allocated_branch_factor_by_depth"][1] == 2


def test_prune_only_caps_predicted_k_at_default_branch_factor():
    strategy = make_strategy(
        FakeExpander(branch_factors={0: 1, 1: 3, "default": 1}, ppl=5.0),
        allocation_mode="prune_only",
        use_residual_budget=False,
    )

    tree = asyncio.run(strategy._construct_budget_allocated_tree("Q", max_depth=2))

    child = tree["children"][0]
    assert child["gear_predicted_k"] == 5
    assert child["gear_allocated_branch_factor"] == 3
    assert len(child["children"]) == 3
    assert tree["gear_queued_node_count"] == 0
    assert tree["gear_reserve_consumed"] == 0
    assert tree["gear_allocated_branch_factor_by_depth"][1] == 3


def _render_gear_inference_strategy(*addons):
    imports = [
        f'(import "{CONFIGS / "gear_defaults.libsonnet"}")',
        *(f'(import "{addon}")' for addon in addons),
        f'(import "{CONFIGS / "gear_overlay.libsonnet"}")',
    ]
    snippet = "(" + " + ".join(imports) + ").episode_generator.inference_strategy"
    return json.loads(_jsonnet.evaluate_snippet("gear_options", snippet))


def test_gear_allocation_options_render_to_inference_strategy():
    default_cfg = _render_gear_inference_strategy()
    no_residual_cfg = _render_gear_inference_strategy(
        CONFIGS / "ablations" / "abl_no_residual_budget.jsonnet"
    )
    no_allocation_cfg = _render_gear_inference_strategy(
        CONFIGS / "ablations" / "abl_no_allocation.jsonnet"
    )

    assert default_cfg["gear_allocation_mode"] == "budget_allocation"
    assert default_cfg["gear_use_residual_budget"] is True
    assert no_residual_cfg["gear_allocation_mode"] == "budget_allocation"
    assert no_residual_cfg["gear_use_residual_budget"] is False
    assert no_allocation_cfg["gear_allocation_mode"] == "prune_only"
    assert no_allocation_cfg["gear_use_residual_budget"] is False

def test_one_depth_root_predicts_k_and_queues_when_k_covers_default_branch_factor(monkeypatch):
    from treetune.inference_strategies import gear_inference_strategy as gear_module

    class FakeRootKEstimator:
        mode = "hierachical"

        def __init__(self, **kwargs):
            pass

        async def estimate_k_for_parent(self, parent, *, depth, duplicate_tv_threshold):
            return SimpleNamespace(
                predicted_k=2,
                reward_variance=1.0,
                pair_tvs={},
                samples=[],
                duplicate_pairs=[],
                unique_candidates=[],
                logp_matrix=[],
            )

    monkeypatch.setattr(gear_module, "ConditionalTVEstimator", FakeRootKEstimator)
    strategy = make_strategy(
        FakeExpander(branch_factors={0: 2, "default": 2}, ppl=2.0),
        allocation_mode="budget_allocation",
        use_residual_budget=True,
    )
    strategy.gear_k_algorithm = "hierarchical"

    tree = asyncio.run(strategy._construct_budget_allocated_tree("Q", max_depth=1))

    assert tree["gear_predicted_k"] == 2
    assert tree["gear_queued_node_count"] == 1
    assert tree["gear_queue_flush_count"] == 1
    assert tree["gear_allocated_branch_factor"] == 2
    assert tree["gear_queue_base_budget"] == 2
    assert tree["gear_default_branch_factor"] == 2
    assert len(tree["children"]) == 2


def test_one_depth_root_predicts_k_and_prunes_without_allocation_when_k_is_small():
    strategy = make_strategy(
        FakeExpander(branch_factors={0: 3, "default": 3}, ppl=2.0),
        allocation_mode="budget_allocation",
        use_residual_budget=True,
    )
    strategy.gear_k_algorithm = "hierarchical"
    strategy.gear_pilot_branch_factor = 2

    tree = asyncio.run(strategy._construct_budget_allocated_tree("Q", max_depth=1))

    assert tree["gear_predicted_k"] == 1
    assert tree["gear_queued_node_count"] == 0
    assert tree["gear_direct_expand_count"] == 1
    assert tree["gear_allocated_branch_factor"] == 1
    assert tree["gear_default_branch_factor"] == 3
    assert len(tree["children"]) == 1
