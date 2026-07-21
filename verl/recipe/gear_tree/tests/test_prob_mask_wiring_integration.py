"""Integration tests for the probability-mask CALL CHAIN (PLAN.md §1/§2).

Unit tests over the helpers cannot catch a broken signature between the
rollout layers, which is exactly how a runtime blocker slipped in. These
tests exercise the real call chain:

    build_tree_edges_async(probability_mask_threshold=...)
        -> extract_edges_from_tree(probability_mask_threshold=...)
        -> stamped response/prob-mask token counts

with a stubbed segment generator and reward function (no server, no GPU), so
an unexpected-keyword or a dropped forward fails here instead of at runtime.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()

pytest.importorskip("torch")

from recipe.gear_tree import async_tree_rollout  # noqa: E402
from recipe.gear_tree.prob_mask import count_prob_mask_active_tokens  # noqa: E402
from recipe.gear_tree.tree_advantage import extract_edges_from_tree  # noqa: E402

# exp(-0.2) ~= 0.819; exp(-0.02) ~= 0.980.
LP_LOW = -0.2
LP_HIGH = -0.02


class TestSignatureContract:
    """Required test 2: the async rollout path must ACCEPT the threshold —
    a missing keyword is a runtime TypeError in production."""

    def test_build_tree_edges_async_accepts_the_threshold(self):
        params = inspect.signature(
            async_tree_rollout.build_tree_edges_async
        ).parameters
        assert "probability_mask_threshold" in params

    def test_extract_edges_from_tree_accepts_the_threshold(self):
        params = inspect.signature(extract_edges_from_tree).parameters
        assert "probability_mask_threshold" in params

    def test_tree_agent_loop_reads_the_per_request_value(self):
        """The rollout worker must read the per-request value, not a stale
        constructor-time copy (PLAN.md §2)."""
        src = inspect.getsource(async_tree_rollout)
        assert "policy_probability_mask_threshold" in src
        assert "probability_mask_threshold=probability_mask_threshold" in src


def _stub_tree(old_log_probs):
    """One parent with four children; child i carries ``old_log_probs[i]``."""
    rewards = (0.8, 0.2, 0.5, 0.5)  # mean 0.5 -> advantages [+.3, -.3, 0, 0]
    return {
        "reward": 0.5,
        "reward_std": 0.25,
        "full_text": "Q",
        "_request_object": {"_treetune__idx": 7, "problem": "1+1"},
        "children": [
            {
                "text": f" s{i}",
                "full_text": f"Q s{i}",
                "reward": r,
                "reward_std": 0.0,
                "leaf": True,
                "response_token_ids": [11, 12, 13],
                "actor_shifted_log_probs": list(old_log_probs[i]),
            }
            for i, r in enumerate(rewards)
        ],
    }


class TestThresholdReachesExtraction:
    """Required test 3: the requested threshold decides the stamped counts."""

    # Two low (active) + one high (masked) per row.
    LPS = [[LP_LOW, LP_HIGH, LP_LOW]] * 4

    @pytest.mark.parametrize(
        "threshold,expected_active",
        [
            (0.9, 2),    # only the two exp~0.819 tokens are below 0.9
            (0.99, 3),   # all three are below 0.99
            (0.5, 0),    # none is below 0.5
        ],
    )
    def test_extraction_stamps_counts_for_the_requested_threshold(
        self, threshold, expected_active
    ):
        records = extract_edges_from_tree(
            _stub_tree(self.LPS),
            tree_update_mode="spo",
            only_adv_greater_than_zero=True,
            emit_zero_slots=True,
            probability_mask_threshold=threshold,
        )
        assert len(records) == 4
        for rec in records:
            assert rec["response_token_count"] == 3
            assert rec["prob_mask_token_count"] == expected_active
            assert rec["probability_mask_threshold"] == threshold
            # The stamped count agrees with the shared predicate.
            assert rec["prob_mask_token_count"] == count_prob_mask_active_tokens(
                self.LPS[0], threshold
            )

    def test_zero_slots_carry_the_counts_without_payload(self):
        records = extract_edges_from_tree(
            _stub_tree(self.LPS),
            tree_update_mode="spo",
            only_adv_greater_than_zero=True,
            emit_zero_slots=True,
            probability_mask_threshold=0.9,
        )
        slots = [r for r in records if r.get("trainable_edge_id", "x") is None]
        assert len(slots) == 2
        for slot in slots:
            assert slot["prob_mask_token_count"] == 2
            assert slot["probability_mask_threshold"] == 0.9
            assert "actor_shifted_log_probs" not in slot


class _StubSegmentGenerator:
    """Minimal async segment generator: never expands (depth-0 tree)."""

    free_max_tokens = 16

    async def segment_fn(self, prefix_ids, k, _params=None):
        return []


class TestAsyncCallChain:
    """Required test 1 + 2: build_tree_edges_async forwards the threshold to
    extraction and raises no unexpected-keyword error."""

    def _run(self, threshold: float, monkeypatch):
        captured = {}
        real_extract = async_tree_rollout.extract_edges_from_tree

        def _spy(tree, **kwargs):
            captured.update(kwargs)
            return real_extract(tree, **kwargs)

        monkeypatch.setattr(async_tree_rollout, "extract_edges_from_tree", _spy)

        async def _fake_build_tree(prompt_text, prompt_ids, inst, **kwargs):
            return _stub_tree([[LP_LOW, LP_HIGH, LP_LOW]] * 4)

        monkeypatch.setattr(async_tree_rollout, "async_build_tree", _fake_build_tree)

        edges = asyncio.run(
            async_tree_rollout.build_tree_edges_async(
                "Q",
                [1, 2],
                {"_treetune__idx": 7},
                segment_generator=_StubSegmentGenerator(),
                reward_fn=lambda q, r, i: (1.0, {}),
                tree_shape=[4],
                M=8,
                only_adv_greater_than_zero=True,
                emit_zero_slots=True,
                probability_mask_threshold=threshold,
            )
        )
        return edges, captured

    def test_threshold_is_forwarded_to_extraction(self, monkeypatch):
        edges, captured = self._run(0.99, monkeypatch)
        assert captured["probability_mask_threshold"] == 0.99
        assert edges, "the chain must still produce records"

    @pytest.mark.parametrize(
        "threshold,expected_active", [(0.9, 2), (0.99, 3), (0.5, 0)]
    )
    def test_counts_stamped_through_the_full_chain(
        self, threshold, expected_active, monkeypatch
    ):
        edges, _ = self._run(threshold, monkeypatch)
        for rec in edges:
            assert rec["probability_mask_threshold"] == threshold
            assert rec["prob_mask_token_count"] == expected_active

    def test_default_threshold_is_the_historical_value(self, monkeypatch):
        """Omitting the argument must not crash and must use 0.9."""
        captured = {}
        real_extract = async_tree_rollout.extract_edges_from_tree

        def _spy(tree, **kwargs):
            captured.update(kwargs)
            return real_extract(tree, **kwargs)

        monkeypatch.setattr(async_tree_rollout, "extract_edges_from_tree", _spy)

        async def _fake_build_tree(prompt_text, prompt_ids, inst, **kwargs):
            return _stub_tree([[LP_LOW, LP_HIGH, LP_LOW]] * 4)

        monkeypatch.setattr(async_tree_rollout, "async_build_tree", _fake_build_tree)

        asyncio.run(
            async_tree_rollout.build_tree_edges_async(
                "Q",
                [1, 2],
                {"_treetune__idx": 7},
                segment_generator=_StubSegmentGenerator(),
                reward_fn=lambda q, r, i: (1.0, {}),
                tree_shape=[4],
                M=8,
            )
        )
        assert captured["probability_mask_threshold"] == 0.9
