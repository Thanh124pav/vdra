"""PLAN.md §1-§14 correction spec: probability mask across all four
canonical combinations, plus the compatibility and accounting rules.

Covers required tests 1-9, 19-25 (the distributed ones 12-13 live in
test_fsdp_canonical_parity.py / test_logical_dispatch_distributed.py, and
10-11/14-18/21-22 in their own files as noted).

Combinations exercised:

    segment_mean + use_prob_mask=false
    segment_mean + use_prob_mask=true
    token_mean   + use_prob_mask=false
    token_mean   + use_prob_mask=true
"""

from __future__ import annotations

import math

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()

torch = pytest.importorskip("torch")

from recipe.gear_tree.policy_loss import (  # noqa: E402
    compute_policy_loss_vdra_segment_mean,
)
from recipe.gear_tree.prob_mask import (  # noqa: E402
    count_prob_mask_active_tokens,
    effective_action_mask,
    probability_mask_active,
)

# exp(-0.2) ~= 0.819 < 0.9 -> ACTIVE; exp(-0.02) ~= 0.980 > 0.9 -> masked out.
LP_ACTIVE = -0.2
LP_INACTIVE = -0.02


def _config(
    *,
    aggregation: str,
    use_prob_mask: bool,
    threshold: float = 0.9,
    reduction: str = "mean",
):
    from verl.workers.config.actor import ActorConfig, PolicyLossConfig

    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=8,
        policy_loss=PolicyLossConfig(
            loss_mode="vdra_segment_mean_ppo",
            policy_aggregation=aggregation,
            segment_token_reduction=reduction,
            use_prob_mask=use_prob_mask,
            probability_mask_threshold=threshold,
        ),
    )


def _rows(advs, lengths, old_lps, n_tokens: int = 6):
    """``old_lps[i]`` is the per-token old log-prob list for row ``i``.

    ``log_prob == old_log_prob`` so the ratio is exactly 1 and the per-token
    surrogate is exactly ``-advantage``.
    """
    n = len(advs)
    old = torch.zeros(n, n_tokens)
    for i, lps in enumerate(old_lps):
        for t, lp in enumerate(lps):
            old[i, t] = lp
    log_prob = old.clone()
    advantages = torch.tensor(advs, dtype=torch.float32).unsqueeze(-1).expand(n, n_tokens).clone()
    mask = torch.zeros(n, n_tokens)
    for i, k in enumerate(lengths):
        mask[i, :k] = 1.0
    return old, log_prob, advantages, mask


def _loss(config, rows, **kwargs):
    old, log_prob, advantages, mask = rows
    loss, *_ = compute_policy_loss_vdra_segment_mean(
        old_log_prob=old,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=mask,
        config=config,
        **kwargs,
    )
    return loss


class TestSharedPredicate:
    def test_threshold_boundary_is_strict_less_than(self):
        """Required test 7: a token whose old prob EQUALS the threshold is
        NOT active (strict ``<``, not ``<=``)."""
        exactly_at = math.log(0.9)
        assert probability_mask_active(exactly_at, 0.9) is False
        assert probability_mask_active(exactly_at - 1e-6, 0.9) is True
        t = torch.tensor([exactly_at, exactly_at - 1e-6])
        assert probability_mask_active(t, 0.9).tolist() == [False, True]

    def test_count_matches_tensor_mask_semantics(self):
        """Required test 6 (half): extraction counting and the actor mask
        use the SAME predicate, so they agree token for token."""
        lps = [LP_ACTIVE, LP_INACTIVE, LP_ACTIVE]
        for threshold in (0.5, 0.83, 0.9, 1.0):
            counted = count_prob_mask_active_tokens(lps, threshold)
            tensor_mask = probability_mask_active(torch.tensor(lps), threshold)
            assert counted == int(tensor_mask.sum())

    def test_changing_threshold_changes_counts_consistently(self):
        """Required test 6: the count moves with the threshold, identically
        on both sides."""
        lps = [LP_ACTIVE, LP_INACTIVE]
        # 0.9 keeps only the -0.2 token; 0.99 keeps both.
        assert count_prob_mask_active_tokens(lps, 0.9) == 1
        assert count_prob_mask_active_tokens(lps, 0.99) == 2
        assert int(probability_mask_active(torch.tensor(lps), 0.9).sum()) == 1
        assert int(probability_mask_active(torch.tensor(lps), 0.99).sum()) == 2

    @pytest.mark.parametrize("use_prob_mask", [False, True])
    def test_dummy_rows_are_masked_explicitly(self, use_prob_mask):
        """Required test 9: dummy rows contribute nothing under BOTH
        use_prob_mask values — never relying on the probability mask."""
        response_mask = torch.ones(2, 3)
        # Row 1 is a dummy whose old log-probs would otherwise be ACTIVE.
        old = torch.full((2, 3), LP_ACTIVE)
        is_dummy = torch.tensor([0, 1])
        mask = effective_action_mask(
            response_mask,
            old,
            use_prob_mask=use_prob_mask,
            probability_mask_threshold=0.9,
            is_dummy=is_dummy,
        )
        assert mask[0].all()
        assert not mask[1].any()


class TestSegmentMeanUnmasked:
    def test_l1_l2_zero_zero_gives_quarter(self):
        """Required test 1 + 2: zero-advantage slots count in M_B."""
        rows = _rows([0.7, -0.3], [3, 2], [[LP_ACTIVE] * 3, [LP_ACTIVE] * 2])
        loss = _loss(
            _config(aggregation="segment_mean", use_prob_mask=False),
            rows,
            original_logical_segment_count=4,
        )
        assert torch.allclose(loss, torch.tensor((-0.7 + 0.3) / 4.0), atol=1e-6)


class TestSegmentMeanMasked:
    def test_masked_row_uses_only_active_tokens_for_its_mean(self):
        """segment_mean + use_prob_mask=true: the per-segment mean is over
        the ACTIVE tokens only, and M_B is still the pre-filter slot count."""
        # Row 0: 3 valid tokens, only 2 active. Row 1: 2 valid, both active.
        rows = _rows(
            [0.6, -0.4],
            [3, 2],
            [[LP_ACTIVE, LP_INACTIVE, LP_ACTIVE], [LP_ACTIVE] * 2],
        )
        loss = _loss(
            _config(aggregation="segment_mean", use_prob_mask=True),
            rows,
            original_logical_segment_count=4,
        )
        # Per-row token means of a constant -A over the ACTIVE tokens are
        # still -A; M_B = 4 slots.
        assert torch.allclose(loss, torch.tensor((-0.6 + 0.4) / 4.0), atol=1e-6)

    def test_empty_active_mask_gives_zero_segment_loss_but_keeps_slot(self):
        """Required test 8: a segment with no active token contributes a
        differentiable ZERO while still counting in M_B."""
        rows = _rows(
            [0.6, -0.4],
            [2, 2],
            [[LP_INACTIVE] * 2, [LP_ACTIVE] * 2],  # row 0 fully masked out
        )
        cfg = _config(aggregation="segment_mean", use_prob_mask=True)
        loss = _loss(cfg, rows, original_logical_segment_count=4)
        # Only row 1 contributes: (+0.4)/4.
        assert torch.allclose(loss, torch.tensor(0.4 / 4.0), atol=1e-6)
        assert torch.isfinite(loss)


class TestTokenMeanDenominatorSelection:
    def test_unmasked_uses_response_token_count(self):
        """Required test 3: zero-slot RESPONSE tokens count in T_B_response."""
        rows = _rows([0.7, -0.3], [3, 2], [[LP_ACTIVE] * 3, [LP_ACTIVE] * 2])
        t_b = 3 + 2 + 5  # + 5 tokens across the zero slots
        loss = _loss(
            _config(aggregation="token_mean", use_prob_mask=False),
            rows,
            original_logical_response_token_count=t_b,
        )
        assert torch.allclose(
            loss, torch.tensor(-(0.7 * 3 - 0.3 * 2) / t_b), atol=1e-6
        )

    def test_masked_uses_prob_mask_token_count(self):
        """Required test 4: zero-slot ACTIVE tokens count in T_B_prob_mask."""
        # Row 0 has 3 valid tokens but only 2 active.
        rows = _rows(
            [0.7, -0.3],
            [3, 2],
            [[LP_ACTIVE, LP_INACTIVE, LP_ACTIVE], [LP_ACTIVE] * 2],
        )
        t_b_mask = 2 + 2 + 3  # + 3 ACTIVE tokens across the zero slots
        loss = _loss(
            _config(aggregation="token_mean", use_prob_mask=True),
            rows,
            original_logical_prob_mask_token_count=t_b_mask,
        )
        # Numerator counts only the 2 active tokens of row 0.
        assert torch.allclose(
            loss, torch.tensor(-(0.7 * 2 - 0.3 * 2) / t_b_mask), atol=1e-6
        )

    def test_masked_numerator_is_never_divided_by_unmasked_count(self):
        """Required test 5: supplying only the RESPONSE count under
        use_prob_mask=true is a hard error, not a silent normalization."""
        rows = _rows([0.7], [3], [[LP_ACTIVE, LP_INACTIVE, LP_ACTIVE]])
        with pytest.raises(ValueError, match="prob_mask_token_count"):
            _loss(
                _config(aggregation="token_mean", use_prob_mask=True),
                rows,
                original_logical_response_token_count=8,
            )

    def test_unmasked_requires_the_response_count(self):
        rows = _rows([0.7], [3], [[LP_ACTIVE] * 3])
        with pytest.raises(ValueError, match="response_token_count"):
            _loss(
                _config(aggregation="token_mean", use_prob_mask=False),
                rows,
                original_logical_prob_mask_token_count=8,
            )

    def test_the_two_denominators_give_different_losses(self):
        """The distinction is real: same rows, different denominators."""
        rows = _rows(
            [0.7, -0.3],
            [3, 2],
            [[LP_ACTIVE, LP_INACTIVE, LP_ACTIVE], [LP_ACTIVE] * 2],
        )
        unmasked = _loss(
            _config(aggregation="token_mean", use_prob_mask=False),
            rows,
            original_logical_response_token_count=10,
        )
        masked = _loss(
            _config(aggregation="token_mean", use_prob_mask=True),
            rows,
            original_logical_prob_mask_token_count=10,
        )
        assert not torch.allclose(unmasked, masked)


class TestSplitInvarianceAllCombinations:
    """Required test 10: micro-batch splitting preserves the full
    logical-batch objective in every canonical combination."""

    @pytest.mark.parametrize("aggregation", ["segment_mean", "token_mean"])
    @pytest.mark.parametrize("use_prob_mask", [False, True])
    def test_split_sums_to_whole(self, aggregation, use_prob_mask):
        advs = [0.7, -0.3, 0.5, -0.9, 0.2, -0.4]
        lengths = [3, 2, 4, 1, 5, 2]
        old_lps = [
            [LP_ACTIVE, LP_INACTIVE, LP_ACTIVE],
            [LP_ACTIVE] * 2,
            [LP_ACTIVE, LP_ACTIVE, LP_INACTIVE, LP_ACTIVE],
            [LP_INACTIVE],
            [LP_ACTIVE] * 5,
            [LP_INACTIVE, LP_ACTIVE],
        ]
        rows = _rows(advs, lengths, old_lps)
        cfg = _config(aggregation=aggregation, use_prob_mask=use_prob_mask)
        stamps = {
            "original_logical_segment_count": 8,
            "original_logical_response_token_count": 25,
            "original_logical_prob_mask_token_count": 18,
        }
        full = _loss(cfg, rows, **stamps)
        split_sum = torch.tensor(0.0)
        for sl in (slice(0, 2), slice(2, 5), slice(5, 6)):
            split_sum = split_sum + _loss(cfg, tuple(t[sl] for t in rows), **stamps)
        assert torch.allclose(full, split_sum, atol=1e-6)


class TestSegmentMeanIndependence:
    def test_segment_mean_ignores_tree_counts(self):
        """Required test 24: segment_mean is independent of the tree count
        and the per-tree segment count."""
        rows = _rows([0.7, -0.3], [3, 2], [[LP_ACTIVE] * 3, [LP_ACTIVE] * 2])
        cfg = _config(aggregation="segment_mean", use_prob_mask=False)
        base = _loss(cfg, rows, original_logical_segment_count=4)
        with_tree_inputs = _loss(
            cfg,
            rows,
            original_logical_segment_count=4,
            tree_total_segment_count=torch.tensor([9.0, 2.0]),
            original_optimizer_batch_tree_count=7,
        )
        assert torch.allclose(base, with_tree_inputs, atol=1e-7)

    def test_tree_balanced_differs_on_unequal_tree_sizes(self):
        """Required test 25: the labeled ablation weighs trees equally and
        therefore differs when trees hold unequal segment counts."""
        rows = _rows([0.7, -0.3], [3, 2], [[LP_ACTIVE] * 3, [LP_ACTIVE] * 2])
        seg = _loss(
            _config(aggregation="segment_mean", use_prob_mask=False),
            rows,
            original_logical_segment_count=2,
        )
        tree = _loss(
            _config(
                aggregation="tree_balanced_segment_mean", use_prob_mask=False
            ),
            rows,
            tree_total_segment_count=torch.tensor([1.0, 3.0]),
            original_optimizer_batch_tree_count=2,
        )
        assert not torch.allclose(seg, tree)


class TestSchemaGuards:
    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_threshold_rejected(self, bad):
        from verl.workers.config.actor import PolicyLossConfig

        with pytest.raises(ValueError, match="probability_mask_threshold"):
            PolicyLossConfig(probability_mask_threshold=bad)

    def test_threshold_one_is_accepted(self):
        from verl.workers.config.actor import PolicyLossConfig

        assert PolicyLossConfig(probability_mask_threshold=1.0).probability_mask_threshold == 1.0

    def test_both_use_prob_mask_values_supported(self):
        from verl.workers.config.actor import PolicyLossConfig

        assert PolicyLossConfig(use_prob_mask=True).use_prob_mask is True
        assert PolicyLossConfig(use_prob_mask=False).use_prob_mask is False
