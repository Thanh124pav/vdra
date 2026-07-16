"""CPU test for the async (agent-loop) tree rollout path, with a mock server.

mock AsyncLLMServerManager -> AsyncServerSegmentGenerator -> async_build_tree ->
edges -> edges_to_dataproto. Proves the async re-target's data path works without
a GPU / real vLLM server.
"""

import asyncio
import random
import types

import torch

from recipe.gear_tree.async_tree_rollout import (
    AsyncServerSegmentGenerator,
    SegmentNodeExpander,
    build_tree_edges_async,
    collect_tree_edges,
)
from recipe.gear_tree.gear_gate import GearGate
from recipe.gear_tree.gear_core.reward_function import MathRewardFunction
from recipe.gear_tree.tree_data import edges_to_dataproto


class MockServerManager:
    """Mimics AsyncLLMServerManager.generate (one sample per call)."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.calls = 0

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kw):
        self.calls += 1
        cap = sampling_params["max_tokens"]
        ntok = self.rng.randint(1, 3)
        tids = [self.rng.randint(10, 99) for _ in range(ntok)]
        logps = [-self.rng.random() for _ in range(ntok)]
        # Sometimes hit the cap (=> "length" => expand); else stop.
        stop_reason = "length" if (ntok >= 3 and cap and cap > 3) else "stop"
        await asyncio.sleep(0)  # yield, exercise concurrency
        return types.SimpleNamespace(token_ids=tids, log_probs=logps, stop_reason=stop_reason)


class MockTok:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " " + "_".join(map(str, ids))


def test_async_segment_generator_fires_n_concurrent():
    mgr = MockServerManager(seed=1)
    gen = AsyncServerSegmentGenerator(mgr, MockTok(), free_max_tokens=32)
    samples = asyncio.run(gen.segment_fn([1, 2, 3], branch_factor=4, max_tokens=8))
    assert len(samples) == 4
    assert mgr.calls == 4
    for s in samples:
        assert s.finish_reason in ("length", "stop")
        assert len(s.token_ids) == len(s.logprobs)


def test_async_full_path_to_dataproto():
    mgr = MockServerManager(seed=5)
    gen = AsyncServerSegmentGenerator(mgr, MockTok(), free_max_tokens=32)
    gate = GearGate(k_algorithm="simple", n_min=1, skip_near_leaf_expand=True, max_depth=2)
    reward_fn = MathRewardFunction()
    inst = {"problem": "p", "answer": "4", "reward_model": {"ground_truth": "4"}, "_treetune__idx": 0}

    edges = asyncio.run(
        build_tree_edges_async(
            "PROMPT", [7, 8, 9], inst,
            segment_generator=gen, reward_fn=reward_fn,
            tree_shape=[2, 2], M=8, gear_gate=gate, only_adv_greater_than_zero=False,
        )
    )
    assert edges

    # Simulate the agent-loop manager stashing per-prompt edges in non_tensor_batch.
    class _DP:
        non_tensor_batch = {"gear_tree_edges": [edges]}

    flat = collect_tree_edges(_DP())
    assert flat == edges

    data = edges_to_dataproto(flat, MockTok(), max_prompt_length=16, max_response_length=8)
    assert data.batch["advantages"].shape[1] == 8
    assert data.batch["input_ids"].shape[0] == len(flat)
    assert torch.all(data.batch["response_mask"].sum(-1) > 0)


def test_batch_allocation_path_builds_valid_tree():
    """VDRA budget-allocation path end-to-end with mock server + async scorer."""

    class MockAsyncScorer:
        def __init__(self):
            self.calls = 0

        async def score_one(self, prefix, y):
            self.calls += 1
            # Deterministic pseudo-likelihood in a realistic range.
            return -1.0 - (hash((prefix[-8:], y[:8])) % 100) / 25.0

    mgr = MockServerManager(seed=11)
    gen = AsyncServerSegmentGenerator(mgr, MockTok(), free_max_tokens=32)
    tok = MockTok()
    tok.encode = lambda text, add_special_tokens=False: [ord(c) % 90 for c in text][:16]
    scorer = MockAsyncScorer()
    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=scorer,
        n_min=1,
        pilot_branch_factor=2,
        allocation_runtime="depth_batch",
        skip_near_leaf_expand=True,
        max_depth=3,
    )
    assert gate.use_batch_allocation
    reward_fn = MathRewardFunction()
    inst = {"problem": "p", "answer": "4", "reward_model": {"ground_truth": "4"}, "_treetune__idx": 0}

    edges = asyncio.run(
        build_tree_edges_async(
            "PROMPT", [7, 8, 9], inst,
            segment_generator=gen, reward_fn=reward_fn,
            tree_shape=[2, 2, 2], M=8, gear_gate=gate,
            only_adv_greater_than_zero=False,
            gear_node_expander=SegmentNodeExpander(gen, tok),
        )
    )
    assert edges
    assert gate.allocation_error_count == 0


def test_vineppo_async_annotation_runs():
    mgr = MockServerManager(seed=9)
    gen = AsyncServerSegmentGenerator(mgr, MockTok(), free_max_tokens=32)
    reward_fn = MathRewardFunction()
    inst = {"problem": "p", "answer": "4", "reward_model": {"ground_truth": "4"}, "_treetune__idx": 0}
    edges = asyncio.run(
        build_tree_edges_async(
            "Q", [3, 4], inst, segment_generator=gen, reward_fn=reward_fn,
            tree_shape=[2, 2], M=8, vineppo_K=3, only_adv_greater_than_zero=False,
        )
    )
    assert edges  # MC-value annotation path executed without error



def test_retained_pilot_is_completed_to_main_segment_length():
    from recipe.gear_tree.tree_rollout import _expand_reusing_pilots

    calls = []

    async def segment_fn(prompt_ids, branch_factor, max_tokens):
        calls.append((list(prompt_ids), branch_factor, max_tokens))
        return [types.SimpleNamespace(
            token_ids=[9, 10][:max_tokens],
            text="XY"[:max_tokens],
            finish_reason="length",
            logprobs=[-0.1] * int(max_tokens),
            sum_logprobs=-0.1 * int(max_tokens),
            num_tokens=int(max_tokens),
        )]

    node = {
        "full_token_ids": [1, 2],
        "vdra_pilot_children": [{
            "text": "A",
            "full_text": "QA",
            "finish_reason": "length",
            "response_token_ids": [3],
            "actor_shifted_log_probs": [-0.2],
            "sum_logprobs": -0.2,
            "num_tokens": 1,
        }],
    }
    samples = asyncio.run(_expand_reusing_pilots(node, 1, 3, segment_fn))
    assert samples[0].token_ids == [3, 9, 10]
    assert samples[0].num_tokens == 3
    assert calls == [([1, 2, 3], 1, 2)]
    assert node["vdra_pilot_completion_generated_tokens"] == 2



def test_online_timeout_queue_flushes_before_final_drain():
    from recipe.gear_tree.tree_rollout import async_build_tree

    class DistinctPilotExpander:
        async def expand(self, *, current_node, prefix, depth, max_tokens, branch_factor):
            if prefix == "PROMPT":
                return [
                    {
                        "text": f" p{i}",
                        "full_text": f"{prefix} p{i}",
                        "finish_reason": "length",
                        "sum_logprobs": -0.1 * (i + 1),
                        "num_tokens": 1,
                        "response_token_ids": [10 + i],
                        "actor_shifted_log_probs": [-0.1],
                        "full_token_ids": [7, 10 + i],
                    }
                    for i in range(int(branch_factor))
                ]
            suffix = "s0" if prefix.endswith("p0") else "s1"
            return [
                {
                    "text": f" {suffix}_{i}",
                    "full_text": f"{prefix} {suffix}_{i}",
                    "finish_reason": "stop",
                    "sum_logprobs": -0.2,
                    "num_tokens": 1,
                    "response_token_ids": [20 + i],
                    "actor_shifted_log_probs": [-0.2],
                    "full_token_ids": list(current_node.get("full_token_ids", [])) + [20 + i],
                }
                for i in range(int(branch_factor))
            ]

    class DistinctScorer:
        async def score_one(self, prefix, y):
            own = (prefix.endswith("p0") and "s0" in y) or (prefix.endswith("p1") and "s1" in y)
            return -0.1 if own else -20.0

    async def segment_fn(prompt_ids, branch_factor, max_tokens):
        return [
            types.SimpleNamespace(
                token_ids=[30 + i],
                text=f" c{i}",
                finish_reason="stop",
                logprobs=[-0.3],
                sum_logprobs=-0.3,
                num_tokens=1,
            )
            for i in range(int(branch_factor))
        ]

    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=DistinctScorer(),
        pilot_branch_factor=2,
        likelihood_samples_per_distribution=1,
        queue_capacity=8,
        queue_timeout_seconds=0.001,
        tv_first_phase_tokens=1,
        tv_second_phase_tokens=1,
        skip_near_leaf_expand=False,
        root_allocation=True,
        max_depth=1,
    )
    tree = asyncio.run(
        async_build_tree(
            "PROMPT",
            [7],
            {"_treetune__idx": "timeout"},
            tree_shape=[1],
            M=3,
            segment_fn=segment_fn,
            grade_fn=lambda query, response, inst: 1.0,
            gear_gate=gate,
            gear_node_expander=DistinctPilotExpander(),
        )
    )

    assert tree["vdra_predicted_k"] == 2
    assert tree["vdra_unmet_demand"] == 0
    assert tree["vdra_flush_reason"] == "timeout"
    assert tree["gear_queue_timeout_flush_count"] == 1
    assert tree["vdra_queue_final_drain_count"] == 0
    assert "reward" in tree
    assert all("reward" in child for child in tree.get("children", []))



class _HotColdPilotExpander:
    """First phase: 3 continuable pilots ' p0..p2'; second phase: ' s<i>_k'."""

    async def expand(self, *, current_node, prefix, depth, max_tokens, branch_factor):
        if " p" not in prefix:
            return [
                {
                    "text": f" p{i}",
                    "full_text": f"{prefix} p{i}",
                    "finish_reason": "length",
                    "sum_logprobs": -0.1,
                    "num_tokens": 1,
                    "response_token_ids": [50 + i],
                    "actor_shifted_log_probs": [-0.1],
                    "full_token_ids": list(current_node.get("full_token_ids", [])) + [50 + i],
                }
                for i in range(int(branch_factor))
            ]
        marker = prefix.rstrip()[-1]
        return [
            {
                "text": f" s{marker}_{i}",
                "full_text": f"{prefix} s{marker}_{i}",
                "finish_reason": "stop",
                "sum_logprobs": -0.2,
                "num_tokens": 1,
                "response_token_ids": [60 + i],
                "actor_shifted_log_probs": [-0.2],
                "full_token_ids": list(current_node.get("full_token_ids", [])) + [60 + i],
            }
            for i in range(int(branch_factor))
        ]


class _HotColdScorer:
    """Cold prefixes: identical likelihoods (TV=0). Hot: pilots disagree."""

    async def score_one(self, prefix, y):
        if "cold" in prefix:
            return -5.0
        marker = prefix.rstrip()[-1]
        return -0.1 if f"s{marker}" in y else -25.0


def _hotcold_segment_fn(root_texts):
    async def segment_fn(prompt_ids, branch_factor, max_tokens):
        if list(prompt_ids) == [7]:
            return [
                types.SimpleNamespace(
                    token_ids=[10 + i],
                    text=f" {root_texts[i % len(root_texts)]}",
                    finish_reason="length",
                    logprobs=[-0.1],
                    sum_logprobs=-0.1,
                    num_tokens=1,
                )
                for i in range(int(branch_factor))
            ]
        return [
            types.SimpleNamespace(
                token_ids=[80 + i],
                text=f" leaf{i}",
                finish_reason="stop",
                logprobs=[-0.3],
                sum_logprobs=-0.3,
                num_tokens=1,
            )
            for i in range(int(branch_factor))
        ]

    return segment_fn


def _hotcold_gate(**overrides):
    kwargs = dict(
        k_algorithm="budget_allocation",
        scorer=_HotColdScorer(),
        n_min=1,
        pilot_branch_factor=3,
        likelihood_samples_per_distribution=1,
        tv_first_phase_tokens=1,
        tv_second_phase_tokens=1,
        skip_near_leaf_expand=True,
        max_depth=3,
        queue_count=1,
    )
    kwargs.update(overrides)
    return GearGate(**kwargs)


def test_parallel_siblings_share_one_capacity_flush():
    """Sibling frontier nodes must co-occupy an allocation queue: with a long
    timeout, two concurrently-estimated hot siblings fill the queue to
    capacity and are solved in ONE batch — and the build must not idle for
    the timeout (the old serial builder slept queue_timeout per node)."""

    from recipe.gear_tree.tree_rollout import async_build_tree

    gate = _hotcold_gate(queue_capacity=2, queue_timeout_seconds=5.0)
    tree = asyncio.run(
        async_build_tree(
            "PROMPT",
            [7],
            {"_treetune__idx": "cap"},
            tree_shape=[2, 2, 2],
            M=3,
            segment_fn=_hotcold_segment_fn(["hot", "hotter"]),
            grade_fn=lambda query, response, inst: 1.0,
            gear_gate=gate,
            gear_node_expander=_HotColdPilotExpander(),
        )
    )

    assert tree["vdra_queue_capacity_flush_count"] == 1
    assert tree["gear_queue_timeout_flush_count"] == 0
    [flush] = tree["vdra_queue_flush_records"]
    assert flush["queue_size_at_flush"] == 2
    assert flush["flush_reason"] == "capacity"
    # No serial per-node timeout sleeps (queue_timeout_seconds is 5s).
    assert tree["tree_construction_seconds"] < 2.0
    assert "reward" in tree


def test_parallel_cold_siblings_fund_hot_sibling_via_reserve():
    """Cold siblings (duplicate pilots -> predicted_k=1) contribute saved
    branches to the reserve; the hot sibling's timeout flush draws them and
    expands beyond its default width."""

    from recipe.gear_tree.tree_rollout import async_build_tree

    gate = _hotcold_gate(
        queue_capacity=8, queue_timeout_seconds=0.2, pilot_branch_factor=4
    )
    tree = asyncio.run(
        async_build_tree(
            "PROMPT",
            [7],
            {"_treetune__idx": "reserve"},
            tree_shape=[3, 2, 2],
            M=3,
            segment_fn=_hotcold_segment_fn(["hot", "cold", "colder"]),
            grade_fn=lambda query, response, inst: 1.0,
            gear_gate=gate,
            gear_node_expander=_HotColdPilotExpander(),
        )
    )

    children = {child["text"].strip(): child for child in tree["children"]}
    assert children["cold"]["vdra_predicted_k"] == 1
    assert children["cold"]["vdra_saved_k"] == 1
    assert children["colder"]["vdra_saved_k"] == 1
    hot = children["hot"]
    assert hot["vdra_predicted_k"] == 4
    assert hot["vdra_allocated_k"] == 4  # default 2 + 2 drawn from the reserve
    assert hot["vdra_additional_k"] == 2
    assert tree["gear_reserve_contributed"] == 2
    assert tree["gear_reserve_consumed"] == 2
    assert tree["gear_queue_timeout_flush_count"] == 1


def test_all_terminal_pilots_shortcut_into_graded_leaves():
    """Both pilots hit EOS in phase 1: no TV pair exists, the build must not
    crash (old D1 behavior) and complete pilot answers become graded leaf
    children, capped by the final allocated branch budget."""

    from recipe.gear_tree.tree_rollout import async_build_tree

    class TerminalPilotExpander:
        async def expand(self, *, current_node, prefix, depth, max_tokens, branch_factor):
            return [
                {
                    "text": f" done{i}",
                    "full_text": f"{prefix} done{i}",
                    "finish_reason": "stop",
                    "sum_logprobs": -0.1,
                    "num_tokens": 1,
                    "response_token_ids": [40 + i],
                    "actor_shifted_log_probs": [-0.1],
                    "full_token_ids": [7, 40 + i],
                }
                for i in range(int(branch_factor))
            ]

    class NeverScorer:
        async def score_one(self, prefix, y):
            raise AssertionError("terminal pilots must not be TV-scored")

    async def segment_fn(prompt_ids, branch_factor, max_tokens):
        raise AssertionError("shortcut leaves cover the budget; no fresh generation")

    gate = GearGate(
        k_algorithm="budget_allocation",
        scorer=NeverScorer(),
        pilot_branch_factor=2,
        likelihood_samples_per_distribution=1,
        queue_timeout_seconds=0.001,
        tv_first_phase_tokens=1,
        tv_second_phase_tokens=1,
        skip_near_leaf_expand=False,
        root_allocation=True,
        max_depth=1,
    )
    tree = asyncio.run(
        async_build_tree(
            "PROMPT",
            [7],
            {"_treetune__idx": "shortcut"},
            tree_shape=[1],
            M=3,
            segment_fn=segment_fn,
            grade_fn=lambda query, response, inst: 1.0,
            gear_gate=gate,
            gear_node_expander=TerminalPilotExpander(),
        )
    )

    children = tree.get("children") or []
    assert len(children) == 1
    assert all(child["leaf"] and "reward" in child for child in children)
    assert tree["vdra_pilot_children_shortcut"] == 1
    assert tree["vdra_shortcut_overage"] == 1  # one terminal pilot exceeded the 1-branch budget
    assert tree["vdra_predicted_k"] == 2
    assert tree["vdra_dispersion_C"] == 0.0
    assert "reward" in tree


def test_expand_reusing_pilots_clamps_fresh_branches_to_token_cap():
    from recipe.gear_tree.tree_rollout import _expand_reusing_pilots

    async def segment_fn(prompt_ids, branch_factor, max_tokens):
        return [
            types.SimpleNamespace(
                token_ids=[90 + i],
                text=f" x{i}",
                finish_reason="stop",
                logprobs=[-0.1],
                sum_logprobs=-0.1,
                num_tokens=1,
            )
            for i in range(int(branch_factor))
        ]

    budget = {"cap": 2, "used": 0, "cap_hit_count": 0, "free_max_tokens": 4}
    node = {"full_token_ids": [1]}
    samples = asyncio.run(
        _expand_reusing_pilots(node, 3, 2, segment_fn, token_budget=budget)
    )
    # per_branch=2 tokens, remaining=2 -> only 1 of 3 fresh branches allowed.
    assert len(samples) == 1
    assert node["vdra_token_cap_hit"] is True
    assert budget["cap_hit_count"] == 1
    assert budget["used"] == 1  # the generated branch was 1 token long


def test_fixed_total_generated_records_cap_accounting():
    from recipe.gear_tree.tree_rollout import (
        _uniform_generated_token_cap,
        async_build_tree,
    )

    gate = _hotcold_gate(
        queue_capacity=2, queue_timeout_seconds=5.0, budget_mode="fixed_total_generated"
    )
    tree = asyncio.run(
        async_build_tree(
            "PROMPT",
            [7],
            {"_treetune__idx": "cap-mode"},
            tree_shape=[2, 2, 2],
            M=3,
            segment_fn=_hotcold_segment_fn(["hot", "hotter"]),
            grade_fn=lambda query, response, inst: 1.0,
            gear_gate=gate,
            gear_node_expander=_HotColdPilotExpander(),
            free_max_tokens=4,
        )
    )

    expected_cap = _uniform_generated_token_cap([2, 2, 2], 3, 3, 4)
    assert tree["vdra_budget_mode"] == "fixed_total_generated"
    assert tree["vdra_token_cap"] == expected_cap
    assert 0 < tree["vdra_generated_tokens_under_cap"] <= expected_cap
    assert "reward" in tree


def test_oracle_proxy_runs_graded_rollouts_and_is_flagged():
    from recipe.gear_tree.tree_rollout import async_build_tree

    gate = _hotcold_gate(
        queue_capacity=2,
        queue_timeout_seconds=5.0,
        allocation_proxy="oracle",
        oracle_rollouts_per_node=4,
    )
    tree = asyncio.run(
        async_build_tree(
            "PROMPT",
            [7],
            {"_treetune__idx": "oracle"},
            tree_shape=[2, 2, 2],
            M=3,
            segment_fn=_hotcold_segment_fn(["hot", "hotter"]),
            grade_fn=lambda query, response, inst: 1.0,
            gear_gate=gate,
            gear_node_expander=_HotColdPilotExpander(),
        )
    )

    assert tree["vdra_allocation_proxy"] == "oracle"
    scored_nodes = [
        n for n in _iter(tree) if n.get("vdra_oracle_value_dispersion") is not None
    ]
    assert scored_nodes
    for node in scored_nodes:
        # grade_fn is constant 1.0 -> oracle dispersion 0; rollout cost logged.
        assert node["vdra_oracle_value_dispersion"] == 0.0
        assert node["vdra_dispersion_C"] == 0.0
        assert node["vdra_proxy_rollout_tokens"] == 4  # 4 one-token rollouts


def _iter(tree):
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.get("children") or [])


def test_duplicate_pilots_count_all_generated_for_compute_accounting():
    from recipe.gear_tree.tree_rollout import _expand_reusing_pilots

    async def segment_fn(prompt_ids, branch_factor, max_tokens):
        return []

    all_pilots = [
        {
            "text": f" p{i}",
            "finish_reason": "length",
            "response_token_ids": [10 + i],
            "actor_shifted_log_probs": [-0.1],
            "sum_logprobs": -0.1,
            "num_tokens": 1,
        }
        for i in range(8)
    ]
    node = {
        "full_token_ids": [1, 2],
        "vdra_all_pilot_children": all_pilots,
        "vdra_reusable_pilot_children": all_pilots[:3],
    }
    samples = asyncio.run(_expand_reusing_pilots(node, 2, None, segment_fn))
    assert len(samples) == 2
    assert node["vdra_pilot_children_generated"] == 8
    assert node["vdra_pilot_children_reused"] == 2
    assert node["vdra_pilot_children_discarded"] == 6
    assert node["vdra_pilot_reuse_rate"] == 0.25
    assert node["vdra_pilot_generated_tokens"] == 8
