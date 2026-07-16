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
    assert tree["vdra_unmet_demand"] == 1
    assert tree["vdra_flush_reason"] == "timeout"
    assert tree["gear_queue_timeout_flush_count"] == 1
    assert tree["vdra_queue_final_drain_count"] == 0
    assert "reward" in tree
    assert all("reward" in child for child in tree.get("children", []))



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
