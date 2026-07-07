import asyncio
import math

import pytest

from treetune.gear.tv_estimators import ConditionalTVEstimator, TVSample


class FakeScorer:
    def __init__(self):
        self.calls = []

    async def score_one(self, prefix, continuation):
        self.calls.append((prefix, continuation))
        return float(len(prefix) - len(continuation))


class TableScorer:
    def __init__(self, scores):
        self.scores = dict(scores)

    async def score_one(self, prefix, continuation):
        return self.scores[(prefix, continuation)]


class FakeExpander:
    async def expand(self, *args, **kwargs):
        raise AssertionError("not used")


class RecordingExpander:
    def __init__(self):
        self.calls = []

    async def expand(self, *args, **kwargs):
        self.calls.append(kwargs)
        branch_factor = kwargs["branch_factor"]
        prefix = kwargs["prefix"]
        depth = kwargs["depth"]
        return [
            {
                "text": f" c{i}",
                "full_text": f"{prefix} c{i}",
                "sum_logprobs": float(i),
                "finish_reason": "length",
                "depth": depth + 1,
            }
            for i in range(branch_factor)
        ]


class PerplexityExpander:
    def __init__(self, first_phase_ppls):
        self.calls = []
        self.first_phase_ppls = list(first_phase_ppls)

    async def expand(self, *args, **kwargs):
        self.calls.append(kwargs)
        branch_factor = kwargs["branch_factor"]
        prefix = kwargs["prefix"]
        depth = kwargs["depth"]
        if len(self.calls) == 1:
            nodes = []
            for idx in range(branch_factor):
                ppl = self.first_phase_ppls[idx % len(self.first_phase_ppls)]
                nodes.append(
                    {
                        "text": f" first{idx}",
                        "full_text": f"{prefix} first{idx}",
                        "sum_logprobs": -math.log(float(ppl)),
                        "num_tokens": 1,
                        "finish_reason": "length",
                        "depth": depth + 1,
                    }
                )
            return nodes
        return [
            {
                "text": f" second{i}",
                "full_text": f"{prefix} second{i}",
                "sum_logprobs": -0.1,
                "num_tokens": 1,
                "finish_reason": "stop",
                "depth": depth + 1,
            }
            for i in range(branch_factor)
        ]


def test_conditional_tv_estimator_caches_logp_matrix_scores():
    scorer = FakeScorer()
    estimator = ConditionalTVEstimator(
        scorer=scorer,
        node_expander=FakeExpander(),
        gamma=0.5,
        n_tv_estimates=2,
    )
    samples = [
        TVSample(first={"full_text": "p1"}, second={"text": "a"}),
        TVSample(first={"full_text": "p2"}, second={"text": "bb"}),
    ]

    async def go():
        first = await estimator.estimate_from_samples(samples)
        second = await estimator.estimate_from_samples(samples)
        return first, second

    first, second = asyncio.run(go())
    assert len(scorer.calls) == 4  # 2 prefixes x 2 support continuations, only once.
    assert first.logp_matrix == second.logp_matrix
    assert set(first.pair_tvs) == {(0, 1)}


def test_pair_tvs_can_use_half_factor():
    estimator = ConditionalTVEstimator(
        scorer=FakeScorer(),
        node_expander=FakeExpander(),
        gamma=0.5,
        n_tv_estimates=2,
        tv_includes_half_factor=True,
    )

    pair_tvs = estimator._pair_tvs([[1.0, 0.0], [0.0, 1.0]])

    assert pair_tvs[(0, 1)] == pytest.approx(1.0)


def test_estimator_uses_restricted_support_probabilities_without_softmax():
    scorer = TableScorer(
        {
            ("p1", "a"): math.log(0.2),
            ("p1", "bb"): math.log(0.05),
            ("p2", "a"): math.log(0.1),
            ("p2", "bb"): math.log(0.01),
        }
    )
    estimator = ConditionalTVEstimator(
        scorer=scorer,
        node_expander=FakeExpander(),
        gamma=0.5,
        n_tv_estimates=2,
    )
    samples = [
        TVSample(first={"full_text": "p1"}, second={"text": "a"}),
        TVSample(first={"full_text": "p2"}, second={"text": "bb"}),
    ]

    result = asyncio.run(estimator.estimate_from_samples(samples))

    assert result.prob_matrix[0] == pytest.approx([0.2, 0.05])
    assert result.prob_matrix[1] == pytest.approx([0.1, 0.01])
    assert sum(result.prob_matrix[0]) == pytest.approx(0.25)
    assert sum(result.prob_matrix[1]) == pytest.approx(0.11)
    assert result.pair_tvs[(0, 1)] == pytest.approx(0.07)


def test_estimate_for_parent_generates_subnode_samples_with_budgeted_expansion():
    scorer = FakeScorer()
    expander = RecordingExpander()
    estimator = ConditionalTVEstimator(
        scorer=scorer,
        node_expander=expander,
        gamma=0.5,
        n_tv_estimates=3,
        first_phase_tokens=11,
        second_phase_tokens=7,
    )

    result = asyncio.run(estimator.estimate_for_parent({"full_text": "root"}, depth=0))

    assert len(result.samples) == 3
    assert expander.calls[0]["branch_factor"] == 3
    assert expander.calls[0]["max_tokens"] == 11
    assert all(call["branch_factor"] == 1 for call in expander.calls[1:])
    assert all(call["max_tokens"] == 7 for call in expander.calls[1:])
    assert result.pair_tvs


def test_perplexity_parent_without_ppl_uses_hierarchical_first_phase_count():
    expander = PerplexityExpander(first_phase_ppls=[3.5, 1.1, 4.2])
    estimator = ConditionalTVEstimator(
        scorer=FakeScorer(),
        node_expander=expander,
        gamma=0.5,
        mode="perplexity",
        n_tv_estimates=9,
    )

    result = asyncio.run(estimator.estimate_for_parent({"full_text": "root"}, depth=0))

    assert expander.calls[0]["branch_factor"] == 3
    assert [call["branch_factor"] for call in expander.calls[1:]] == [4, 2, 5]
    assert len(result.samples) == 11


def test_perplexity_parent_with_ppl_uses_adaptive_first_phase_count():
    expander = PerplexityExpander(first_phase_ppls=[1.1, 1.1, 1.1, 1.1])
    estimator = ConditionalTVEstimator(
        scorer=FakeScorer(),
        node_expander=expander,
        gamma=0.5,
        mode="perplexity",
        n_tv_estimates=2,
    )

    result = asyncio.run(
        estimator.estimate_for_parent(
            {"full_text": "root", "sum_logprobs": -math.log(3.5), "num_tokens": 1},
            depth=0,
        )
    )

    assert expander.calls[0]["branch_factor"] == 4
    assert all(call["branch_factor"] == 2 for call in expander.calls[1:])
    assert len(result.samples) == 8


class MixedFinishExpander:
    def __init__(self):
        self.calls = []

    async def expand(self, *args, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            prefix = kwargs["prefix"]
            return [
                {
                    "text": " done",
                    "full_text": f"{prefix} done",
                    "finish_reason": "stop",
                },
                {
                    "text": " partial",
                    "full_text": f"{prefix} partial",
                    "finish_reason": "length",
                },
            ]
        prefix = kwargs["prefix"]
        return [
            {
                "text": " continuation",
                "full_text": f"{prefix} continuation",
                "finish_reason": "stop",
            }
        ]


def test_estimate_does_not_continue_terminal_first_phase_nodes():
    expander = MixedFinishExpander()
    estimator = ConditionalTVEstimator(
        scorer=FakeScorer(),
        node_expander=expander,
        gamma=0.5,
        n_tv_estimates=2,
    )

    result = asyncio.run(estimator.estimate_for_parent({"full_text": "root"}, depth=0))

    assert len(expander.calls) == 2
    assert expander.calls[1]["prefix"] == "root partial"
    assert [candidate["finish_reason"] for candidate in result.candidates] == [
        "stop",
        "length",
    ]
    assert len(result.samples) == 1


def test_estimate_k_for_parent_deduplicates_first_prefixes_by_tv():
    class KExpander:
        def __init__(self):
            self.calls = []

        async def expand(self, *args, **kwargs):
            self.calls.append(kwargs)
            prefix = kwargs["prefix"]
            depth = kwargs["depth"]
            branch_factor = kwargs["branch_factor"]
            if len(self.calls) == 1:
                return [
                    {"text": " a", "full_text": "root a", "sum_logprobs": -0.1, "num_tokens": 1, "finish_reason": "length", "depth": depth + 1},
                    {"text": " b", "full_text": "root b", "sum_logprobs": -0.2, "num_tokens": 1, "finish_reason": "length", "depth": depth + 1},
                ][:branch_factor]
            return [
                {"text": " z", "full_text": f"{prefix} z", "sum_logprobs": -0.1, "num_tokens": 1, "finish_reason": "stop", "depth": depth + 1}
            ]

    scorer = TableScorer({
        ("root a", " z"): math.log(0.4),
        ("root b", " z"): math.log(0.39),
    })
    estimator = ConditionalTVEstimator(
        scorer=scorer,
        node_expander=KExpander(),
        gamma=0.5,
        mode="hierarchical",
        n_tv_estimates=4,
        tv_includes_half_factor=True,
    )

    result = asyncio.run(
        estimator.estimate_k_for_parent(
            {"full_text": "root", "sum_logprobs": -0.1, "num_tokens": 1},
            depth=0,
            duplicate_tv_threshold=0.02,
        )
    )

    assert result.predicted_k == 1
    assert len(result.unique_candidates) == 1
    assert result.duplicate_pairs == [(0, 1)]
