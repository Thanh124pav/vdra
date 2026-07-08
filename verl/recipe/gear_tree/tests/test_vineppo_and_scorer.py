"""CPU tests for VinePPO MC-value/step-advantages and the offline LP scorer."""

import math
import types

from recipe.gear_tree.tree_rollout import SegmentSample
from recipe.gear_tree.vineppo_advantage import mc_value, step_advantages
from recipe.gear_tree.engine_scorer import EngineLPScorer


def test_step_advantages_td_residual():
    # values has len(step_rewards)+1; terminal must be None.
    step_rewards = [0.0, 0.0, 1.0]
    values = [0.4, 0.6, 0.9, None]
    adv = step_advantages(step_rewards, values)
    # A[i] = step_rewards[i] + values[i+1] - values[i]; terminal value -> 0.
    assert adv[0] == 0.0 + 0.6 - 0.4
    assert adv[1] == 0.0 + 0.9 - 0.6
    assert adv[2] == 1.0 + 0.0 - 0.9


def test_step_advantages_backfill_missing():
    step_rewards = [0.0, 0.0]
    values = [None, None, None]  # all missing -> back-filled from terminal 0
    adv = step_advantages(step_rewards, list(values))
    # values[2]=0; values[1]=sr[1]+0=0; values[0]=sr[0]+0=0 -> all adv 0.
    assert adv == [0.0, 0.0]


def test_mc_value_mean_reward_with_unfinished_penalty():
    def rollout_fn(prefix_ids, K):
        # 2 finished, 1 truncated.
        return [
            SegmentSample(token_ids=[1], text="a", finish_reason="stop"),
            SegmentSample(token_ids=[2], text="b", finish_reason="stop"),
            SegmentSample(token_ids=[3], text="c", finish_reason="length"),
        ]

    def grade(q, r, inst):
        return 1.0  # finished rollouts score 1

    v = mc_value("pre", [1, 2], {}, rollout_fn=rollout_fn, grade_fn=grade, K=3, unfinished_penalty=-0.5)
    # rewards = [1, 1, -0.5] -> mean = 0.5
    assert v == (1.0 + 1.0 - 0.5) / 3.0


class _Logprob:
    def __init__(self, lp):
        self.logprob = lp


class FakeEngine:
    """prompt_logprobs[i] = {token_id: Logprob}; [0] is None."""

    def __init__(self, lp_by_token):
        self.lp_by_token = lp_by_token

    def generate(self, prompts, sampling_params, use_tqdm=False, **kw):
        ids = prompts[0]["prompt_token_ids"]
        plp = [None]
        for tid in ids[1:]:
            plp.append({tid: _Logprob(self.lp_by_token[tid])})
        return [types.SimpleNamespace(prompt_logprobs=plp)]


class FakeTok:
    def encode(self, text, add_special_tokens=False):
        # each char -> its ordinal (deterministic token ids)
        return [ord(c) for c in text]


def test_engine_scorer_sums_suffix_logprobs(monkeypatch):
    # Patch SamplingParams import inside engine_scorer.
    import sys

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    lp = {ord(c): -0.1 * (i + 1) for i, c in enumerate("preY")}
    scorer = EngineLPScorer(FakeEngine(lp), FakeTok())
    # prefix="pre" (3 toks), full="preY" (4 toks) -> suffix_len=1 -> logprob of 'Y'.
    val = scorer.score_one("pre", "Y")
    assert val == lp[ord("Y")]
    # caching returns same value.
    assert scorer.score_one("pre", "Y") == val
