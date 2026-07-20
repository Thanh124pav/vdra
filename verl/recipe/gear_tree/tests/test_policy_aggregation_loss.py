"""PLAN.md §1.3 (user decision 2026-07-21): canonical paper objectives.

Unit tests for the ``policy_aggregation`` dispatch of
``compute_policy_loss_vdra_segment_mean``:

* ``segment_mean``: uniform ``1/M_B`` over PRE-FILTER logical segment slots
  — ``[L1, L2, 0, 0] -> (L1 + L2) / 4`` (required test 3);
* ``token_mean``: uniform ``1/T_B`` over PRE-FILTER logical tokens — zero
  slots' token lengths stay in the denominator (required test 4);
* micro-batch split invariance under the fixed stamped denominators
  (required test 7);
* fail-fast on every forbidden fallback (strict contract);
* the three explicit aggregation modes produce different results on
  heterogeneous trees (required alias-decision test 4).

The denominators are hand-stamped here exactly as the trainer stamps them
in production (``original_logical_segment_count`` /
``original_logical_token_count``); the dp-size reducer compensation is a
``dp_actor`` concern and is exercised by the FSDP2 parity harness.
"""

from __future__ import annotations

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


def _config(reduction: str = "mean", aggregation: str | None = None, **pl_kwargs):
    from verl.workers.config.actor import ActorConfig, PolicyLossConfig

    kwargs = dict(
        loss_mode="vdra_segment_mean_ppo",
        segment_token_reduction=reduction,
        use_prob_mask=False,
        **pl_kwargs,
    )
    if aggregation is not None:
        kwargs["policy_aggregation"] = aggregation
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=8,
        policy_loss=PolicyLossConfig(**kwargs),
    )


def _rows(advs: list[float], lengths: list[int], n_tokens: int = 6):
    """Rows with ratio == 1 so the per-token surrogate is exactly ``-A``."""
    n = len(advs)
    log_prob = torch.zeros(n, n_tokens)
    old_log_prob = torch.zeros(n, n_tokens)
    advantages = torch.tensor(advs).unsqueeze(-1).expand(n, n_tokens).clone()
    mask = torch.zeros(n, n_tokens)
    for i, k in enumerate(lengths):
        mask[i, :k] = 1.0
    return old_log_prob, log_prob, advantages, mask


def _loss(config, rows, **kwargs) -> torch.Tensor:
    old_log_prob, log_prob, advantages, mask = rows
    loss, *_ = compute_policy_loss_vdra_segment_mean(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=mask,
        config=config,
        **kwargs,
    )
    return loss


class TestSegmentMean:
    def test_l1_l2_zero_zero_gives_quarter(self):
        """Required test 3: [L1, L2, 0, 0] -> (L1 + L2) / 4. The two zero
        slots are metadata-only (no tensor rows) but stay in M_B."""
        rows = _rows([0.7, -0.3], lengths=[3, 2])
        loss = _loss(
            _config(aggregation="segment_mean"),
            rows,
            original_logical_segment_count=4,
        )
        # ratio==1 -> L1 = -0.7, L2 = +0.3 (token-mean of constant rows).
        assert torch.allclose(loss, torch.tensor((-0.7 + 0.3) / 4.0), atol=1e-7)

    def test_denominator_is_not_row_count(self):
        rows = _rows([0.7, -0.3], lengths=[3, 2])
        loss_m4 = _loss(
            _config(aggregation="segment_mean"),
            rows,
            original_logical_segment_count=4,
        )
        loss_m2 = _loss(
            _config(aggregation="segment_mean"),
            rows,
            original_logical_segment_count=2,
        )
        assert torch.allclose(loss_m4 * 2.0, loss_m2, atol=1e-7)

    def test_missing_stamp_fails_fast(self):
        rows = _rows([0.7], lengths=[2])
        with pytest.raises(ValueError, match="original_logical_segment_count"):
            _loss(_config(aggregation="segment_mean"), rows)

    def test_nonpositive_stamp_fails(self):
        rows = _rows([0.7], lengths=[2])
        with pytest.raises(ValueError, match="must be > 0"):
            _loss(
                _config(aggregation="segment_mean"),
                rows,
                original_logical_segment_count=0,
            )

    def test_legacy_flag_conflict_fails(self):
        rows = _rows([0.7], lengths=[2])
        with pytest.raises(ValueError, match="batch_slot_mean_ablation"):
            _loss(
                _config(aggregation="segment_mean", batch_slot_mean_ablation=True),
                rows,
                original_logical_segment_count=4,
            )


class TestTokenMean:
    def test_zero_slot_token_lengths_stay_in_denominator(self):
        """Required test 4: T_B includes the response_token_count of zero
        slots (here 5 tokens across the two zero slots)."""
        rows = _rows([0.7, -0.3], lengths=[3, 2])
        t_b = 3 + 2 + 5
        loss = _loss(
            _config(aggregation="token_mean"),
            rows,
            original_logical_token_count=t_b,
        )
        expected = -(0.7 * 3 + (-0.3) * 2) / t_b
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-7)

    def test_longer_segments_weigh_more(self):
        rows_short_first = _rows([0.7, -0.3], lengths=[1, 4])
        rows_long_first = _rows([0.7, -0.3], lengths=[4, 1])
        cfg = _config(aggregation="token_mean")
        l1 = _loss(rows=rows_short_first, config=cfg, original_logical_token_count=10)
        l2 = _loss(rows=rows_long_first, config=cfg, original_logical_token_count=10)
        assert not torch.allclose(l1, l2)

    def test_missing_stamp_fails_fast(self):
        rows = _rows([0.7], lengths=[2])
        with pytest.raises(ValueError, match="original_logical_token_count"):
            _loss(_config(aggregation="token_mean"), rows)

    def test_sum_reduction_rejected(self):
        from verl.workers.config.actor import PolicyLossConfig

        with pytest.raises(ValueError, match="token_mean"):
            PolicyLossConfig(
                loss_mode="vdra_segment_mean_ppo",
                policy_aggregation="token_mean",
                segment_token_reduction="sum",
            )


class TestMicroSplitInvariance:
    """Required test 7: with the stamped denominators fixed per logical
    batch, summing per-micro-batch losses reproduces the full-batch loss."""

    @pytest.mark.parametrize("aggregation", ["segment_mean", "token_mean"])
    def test_split_sums_to_whole(self, aggregation):
        advs = [0.7, -0.3, 0.5, -0.9, 0.2, -0.4]
        lengths = [3, 2, 4, 1, 5, 2]
        rows = _rows(advs, lengths)
        cfg = _config(aggregation=aggregation)
        stamps = {
            "original_logical_segment_count": 8,
            "original_logical_token_count": 25,
        }
        full = _loss(cfg, rows, **stamps)
        split_sum = torch.tensor(0.0)
        for sl in (slice(0, 2), slice(2, 5), slice(5, 6)):
            part = tuple(t[sl] for t in rows)
            split_sum = split_sum + _loss(cfg, part, **stamps)
        assert torch.allclose(full, split_sum, atol=1e-7)


class TestAggregationModesDiffer:
    def test_three_modes_differ_on_heterogeneous_trees(self):
        """Required alias-decision test 4: token_mean, segment_mean and
        tree_balanced_segment_mean disagree when tree sizes and segment
        lengths differ."""
        advs = [0.7, -0.3, 0.5, -0.9]
        lengths = [1, 4, 2, 3]
        rows = _rows(advs, lengths)
        seg = _loss(
            _config(aggregation="segment_mean"),
            rows,
            original_logical_segment_count=4,
        )
        tok = _loss(
            _config(aggregation="token_mean"),
            rows,
            original_logical_token_count=10,
        )
        tree = _loss(
            _config(aggregation="tree_balanced_segment_mean"),
            rows,
            original_optimizer_batch_tree_count=2,
            tree_total_segment_count=torch.tensor([1.0, 3.0, 3.0, 3.0]),
        )
        assert not torch.allclose(seg, tok)
        assert not torch.allclose(seg, tree)
        assert not torch.allclose(tok, tree)


class TestSchemaGuards:
    def test_default_aggregation_is_still_tree_balanced_until_flip(self):
        from verl.workers.config.actor import PolicyLossConfig

        assert PolicyLossConfig().policy_aggregation == "tree_balanced_segment_mean"

    def test_retired_global_segment_mean_rejected_with_rename_message(self):
        from verl.workers.config.actor import PolicyLossConfig

        with pytest.raises(ValueError, match="tree_balanced_segment_mean"):
            PolicyLossConfig(policy_aggregation="global_segment_mean")

    def test_unknown_aggregation_rejected(self):
        from verl.workers.config.actor import PolicyLossConfig

        with pytest.raises(ValueError, match="policy_aggregation"):
            PolicyLossConfig(policy_aggregation="segment_average")

    def test_loss_level_rejects_retired_name(self):
        rows = _rows([0.7], lengths=[2])
        cfg = _config()
        # Simulate a dict-shaped config that bypassed dataclass validation.
        object.__setattr__(cfg.policy_loss, "policy_aggregation", "global_segment_mean")
        with pytest.raises(ValueError, match="renamed"):
            _loss(cfg, rows)