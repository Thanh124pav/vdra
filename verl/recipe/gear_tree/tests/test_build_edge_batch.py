"""CPU test for build_edge_batch with a fake vLLM engine (no GPU)."""

import random
import types

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from recipe.gear_tree.vllm_rollout_tree import build_edge_batch
from recipe.gear_tree.gear_gate import GearGate


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return "T" + "".join(f"_{i}" for i in ids)


class _Logprob:
    def __init__(self, lp):
        self.logprob = lp


class FakeVLLMEngine:
    """Mimics vllm.LLM.generate returning RequestOutput-like objects."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def generate(self, prompts, sampling_params, use_tqdm=False, **kw):
        n = sampling_params.n
        max_tokens = sampling_params.max_tokens
        completions = []
        for b in range(n):
            ntok = self.rng.randint(1, 3)
            toks = [self.rng.randint(10, 99) for _ in range(ntok)]
            # logprobs[i] is a dict {token_id: Logprob}
            lps = [{tid: _Logprob(-self.rng.random())} for tid in toks]
            finish = "stop" if self.rng.random() > 0.5 else "length"
            completions.append(
                types.SimpleNamespace(
                    token_ids=toks, text=f" seg{b}", finish_reason=finish, logprobs=lps
                )
            )
        return [types.SimpleNamespace(outputs=completions)]


class _SP:  # fake SamplingParams captured by VLLMTreeRollout._sampling_params
    def __init__(self, n, temperature, top_p, top_k, max_tokens, logprobs, seed):
        self.n = n
        self.max_tokens = max_tokens


def _patch_sampling_params(monkeypatch):
    import recipe.gear_tree.tree_rollout as tr

    def fake(self, n, max_tokens):
        return _SP(n, self.temperature, self.top_p, self.top_k,
                   max_tokens if max_tokens is not None else self.free_max_tokens, 1, self.seed)

    monkeypatch.setattr(tr.VLLMTreeRollout, "_sampling_params", fake, raising=True)


def _make_prompts(bsz=2, plen=6):
    input_ids = torch.zeros((bsz, plen), dtype=torch.long)
    for i in range(bsz):
        input_ids[i, -3:] = torch.tensor([100 + i, 101 + i, 102 + i])
    batch = TensorDict({"input_ids": input_ids}, batch_size=bsz)
    ntb = {
        "reward_model": np.array([{"ground_truth": "42"}] * bsz, dtype=object),
        "extra_info": np.array([{"problem": "p"}] * bsz, dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=ntb)


def test_build_edge_batch_produces_valid_dataproto(monkeypatch):
    _patch_sampling_params(monkeypatch)
    prompts = _make_prompts()
    gate = GearGate(k_algorithm="simple", n_min=1, skip_near_leaf_expand=True, max_depth=2)

    out = build_edge_batch(
        prompts,
        inference_engine=FakeVLLMEngine(seed=3),
        tokenizer=FakeTokenizer(),
        tree_shape=[2, 2],
        M=4,
        gear_gate=gate,
        max_prompt_length=16,
        max_response_length=8,
        only_adv_greater_than_zero=False,
    )
    assert out.batch["input_ids"].shape[1] == 24
    assert out.batch["advantages"].shape[1] == 8
    assert out.batch["input_ids"].shape[0] == out.batch["advantages"].shape[0] >= 1
    # every row has at least one valid response token
    assert torch.all(out.batch["response_mask"].sum(dim=-1) > 0)
