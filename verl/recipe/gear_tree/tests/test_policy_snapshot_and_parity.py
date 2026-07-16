import asyncio
import json
import types

import pytest

from recipe.gear_tree.async_tree_rollout import AsyncServerSegmentGenerator, _build_gate, build_tree_edges_async
from recipe.gear_tree.gear_core.reward_function import MathRewardFunction
from recipe.gear_tree.logprob_parity import compare_logprob_arrays, validate_jsonl, validate_parity_record


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " " + "_".join(map(str, ids))

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 97 for ch in text]


class _Server:
    async def generate(self, request_id, *, prompt_ids, sampling_params, **_):
        return types.SimpleNamespace(token_ids=[11], log_probs=[-0.3], stop_reason="stop")


def test_generated_edges_carry_policy_snapshot_id():
    gen = AsyncServerSegmentGenerator(_Server(), _Tok(), free_max_tokens=8)
    edges = asyncio.run(
        build_tree_edges_async(
            "PROMPT",
            [7],
            {"problem": "p", "answer": "4", "reward_model": {"ground_truth": "4"}},
            segment_generator=gen,
            reward_fn=MathRewardFunction(),
            tree_shape=[1],
            M=4,
            only_adv_greater_than_zero=False,
            policy_snapshot_id="global_step:9",
        )
    )
    assert edges
    assert {edge["policy_snapshot_id"] for edge in edges} == {"global_step:9"}


def test_build_gate_rejects_scorer_snapshot_mismatch():
    gt = {
        "policy_snapshot_id": "global_step:1",
        "tree_shape": [1],
        "gear": {
            "enabled": True,
            "scorer_api_base": "http://127.0.0.1:8000/v1",
            "scorer_model": "served-model",
            "scorer_snapshot_id": "global_step:0",
        },
    }
    with pytest.raises(RuntimeError, match="scorer snapshot"):
        _build_gate(gt, tokenizer=_Tok())


def test_build_gate_resolves_nonempty_scorer_model_without_network():
    gt = {
        "policy_snapshot_id": "global_step:1",
        "tree_shape": [1],
        "gear": {
            "enabled": True,
            "scorer_api_base": "http://127.0.0.1:8000/v1",
            "scorer_model": "served-model",
            "scorer_snapshot_id": "global_step:1",
        },
    }
    gate = _build_gate(gt, tokenizer=_Tok())
    assert gate.scorer.scorer_model == "served-model"
    assert gate.scorer.scorer_snapshot_id == "global_step:1"


def test_logprob_parity_helpers_report_max_and_mean(tmp_path):
    out = compare_logprob_arrays([-0.1, -0.2], [-0.1001, -0.1999], atol=1e-2)
    assert out["num_tokens"] == 2.0
    with pytest.raises(AssertionError, match="max_delta"):
        compare_logprob_arrays([-0.1], [-0.5], atol=1e-3)

    record = {
        "model": "m",
        "tokenizer": "tok",
        "prompt_token_ids": [1, 2],
        "response_token_ids": [3],
        "temperature": 0.0,
        "rollout_logprobs": [-0.2],
        "actor_logprobs": [-0.2002],
    }
    assert validate_parity_record(record, atol=1e-2)["max_delta"] < 1e-2
    path = tmp_path / "parity.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert validate_jsonl(path, atol=1e-2)["records"] == 1.0
