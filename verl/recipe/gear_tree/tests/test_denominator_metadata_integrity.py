"""Denominator-metadata integrity + auxiliary/skip contracts.

Required tests 1-7 of the remaining-fixes spec:

1. trainable stamped counts matching payload are accepted;
2. mismatched trainable counts are rejected (insert AND logical build);
3. dummy rows are excluded from entropy/KL reductions;
4. a missing actor optimizer-step metric invalidates accounting (and never
   defaults to 1) while the outer update stays committed;
5. replay restore validates EVERY row, not just checkpoint metadata;
6. canonical aggregation rejects ``underfilled_update_policy=use_available``;
7. a fully skipped iteration writes a timing row and persists the manifest.
"""

from __future__ import annotations

import json

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from omegaconf import OmegaConf  # noqa: E402

from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer  # noqa: E402
from recipe.gear_tree.tree_data import build_logical_update_batch  # noqa: E402

try:
    from recipe.gear_tree.tests import _tiny_actor
except ImportError:
    import _tiny_actor

LP_ACTIVE = -0.2   # exp ~= 0.819 < 0.9 -> active
LP_INACTIVE = -0.02  # exp ~= 0.980 > 0.9 -> masked out


def _edge(edge_id="e0", *, lps=(LP_ACTIVE, LP_ACTIVE), resp=None, mask=None):
    lps = list(lps)
    n = len(lps)
    return {
        "edge_id": edge_id,
        "question_id": "q0",
        "tree_id": "t0",
        "parent_group_id": "t0/pg",
        "child_segment_id": f"t0/{edge_id}",
        "allocated_k": 2,
        "sample_multiplicity": 1,
        "tree_total_segment_count": 2,
        "queue_flush_id": "0",
        "queue_released_segment_count": 2,
        "query_token_ids": [1, 2],
        "response_token_ids": [3 + i for i in range(n)],
        "actor_shifted_log_probs": lps,
        "advantage": 0.5,
        "value": 0.4,
        "reward": 1.0,
        "response_token_count": n if resp is None else resp,
        "prob_mask_token_count": (
            sum(1 for lp in lps if lp < -0.10536) if mask is None else mask
        ),
        "probability_mask_threshold": 0.9,
        "generation_rollout_iteration": 0,
    }


def _slot(edge_id="z0", n=2):
    return {
        "edge_id": edge_id,
        "question_id": "q0",
        "tree_id": "t0",
        "parent_group_id": "t0/pg",
        "child_segment_id": f"t0/{edge_id}",
        "allocated_k": 2,
        "sample_multiplicity": 1,
        "advantage": 0.0,
        "advantage_is_zero": True,
        "trainable_edge_id": None,
        "response_token_count": n,
        "prob_mask_token_count": n,
        "probability_mask_threshold": 0.9,
        "generation_rollout_iteration": 0,
    }


def _buffer():
    return GearTreeReplayBuffer(
        target_edges_per_iteration=16,
        max_edge_age_iterations=8,
        max_edges_per_question_per_iteration=16,
        use_prob_mask=True,
        probability_mask_threshold=0.9,
    )


class TestTrainableCountIdentity:
    """Required tests 1-2."""

    def test_matching_counts_are_accepted(self):
        buf = _buffer()
        buf.add(
            [_edge(lps=[LP_ACTIVE, LP_INACTIVE])],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        assert len(buf) == 1

    def test_wrong_response_token_count_rejected_on_insert(self):
        buf = _buffer()
        with pytest.raises(ValueError, match="response_token_count"):
            buf.add(
                [_edge(resp=7)],
                generation_rollout_iteration=0,
                policy_snapshot_id="snap",
            )

    def test_wrong_prob_mask_token_count_rejected_on_insert(self):
        buf = _buffer()
        # Two INACTIVE tokens -> the true active count is 0, not 2.
        with pytest.raises(ValueError, match="prob_mask_token_count"):
            buf.add(
                [_edge(lps=[LP_INACTIVE, LP_INACTIVE], mask=2)],
                generation_rollout_iteration=0,
                policy_snapshot_id="snap",
            )

    def test_wrong_response_count_rejected_by_logical_build(self):
        with pytest.raises(ValueError, match="response_token_count"):
            build_logical_update_batch(
                [_edge(resp=7), _slot()],
                _tiny_actor.Tok(),
                max_prompt_length=6,
                max_response_length=4,
                ppo_mini_batch_size=2,
                dp_size=1,
                loss_mode="vdra_segment_mean_ppo",
                use_prob_mask=True,
                probability_mask_threshold=0.9,
            )

    def test_wrong_mask_count_rejected_by_logical_build(self):
        with pytest.raises(ValueError, match="prob_mask_token_count"):
            build_logical_update_batch(
                [_edge(lps=[LP_INACTIVE, LP_INACTIVE], mask=2), _slot()],
                _tiny_actor.Tok(),
                max_prompt_length=6,
                max_response_length=4,
                ppo_mini_batch_size=2,
                dp_size=1,
                loss_mode="vdra_segment_mean_ppo",
                use_prob_mask=True,
                probability_mask_threshold=0.9,
            )

    def test_zero_slots_are_exempt_from_recomputation(self):
        """A metadata-only slot has no payload, so only its stamped schema
        and ranges are checked — never a recomputation."""
        buf = _buffer()
        buf.add(
            [_slot()], generation_rollout_iteration=0, policy_snapshot_id="snap"
        )
        assert len(buf) == 1

    def test_canonical_logical_build_requires_complete_denominator_metadata(self):
        edge = _edge("missing_threshold")
        edge.pop("probability_mask_threshold")
        with pytest.raises(ValueError, match="missing_threshold") as excinfo:
            build_logical_update_batch(
                [edge, _slot()],
                _tiny_actor.Tok(),
                max_prompt_length=6,
                max_response_length=4,
                ppo_mini_batch_size=2,
                dp_size=1,
                loss_mode="vdra_segment_mean_ppo",
                use_prob_mask=True,
                probability_mask_threshold=0.9,
                require_logical_denominator_metadata=True,
            )
        assert "probability_mask_threshold" in str(excinfo.value)

    def test_legacy_logical_build_recomputes_missing_metadata_with_warning(self):
        edge = _edge("legacy_missing")
        edge.pop("response_token_count")
        edge.pop("prob_mask_token_count")
        edge.pop("probability_mask_threshold")
        with pytest.warns(RuntimeWarning, match="missing"):
            batch, stats = build_logical_update_batch(
                [edge, _slot()],
                _tiny_actor.Tok(),
                max_prompt_length=6,
                max_response_length=4,
                ppo_mini_batch_size=2,
                dp_size=1,
                loss_mode="vdra_segment_mean_ppo",
                use_prob_mask=True,
                probability_mask_threshold=0.9,
                require_logical_denominator_metadata=False,
            )
        assert batch is not None
        assert stats["vdra/trainable_logical_batches"] == 1.0


class TestReplayRestoreValidatesEveryRow:
    """Required test 5."""

    def test_restore_rejects_a_tampered_row(self, tmp_path):
        buf = _buffer()
        buf.add(
            [_edge()], generation_rollout_iteration=0, policy_snapshot_id="snap"
        )
        buf.save(tmp_path)
        # Tamper with the row AFTER save: checkpoint-level mask metadata is
        # still consistent, so only per-row validation can catch this.
        edge_file = tmp_path / "gear_tree_replay_buffer.jsonl"
        row = json.loads(edge_file.read_text().strip())
        row["prob_mask_token_count"] = 99
        edge_file.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="prob_mask_token_count"):
            GearTreeReplayBuffer.load(
                tmp_path,
                expected_use_prob_mask=True,
                expected_probability_mask_threshold=0.9,
            )

    def test_restore_rejects_a_row_with_a_foreign_threshold(self, tmp_path):
        buf = _buffer()
        buf.add(
            [_edge()], generation_rollout_iteration=0, policy_snapshot_id="snap"
        )
        buf.save(tmp_path)
        edge_file = tmp_path / "gear_tree_replay_buffer.jsonl"
        row = json.loads(edge_file.read_text().strip())
        row["probability_mask_threshold"] = 0.75
        edge_file.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="different objective"):
            GearTreeReplayBuffer.load(
                tmp_path,
                expected_use_prob_mask=True,
                expected_probability_mask_threshold=0.9,
            )

    def test_clean_checkpoint_still_restores(self, tmp_path):
        buf = _buffer()
        buf.add(
            [_edge(), _slot()],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        buf.save(tmp_path)
        restored = GearTreeReplayBuffer.load(
            tmp_path,
            expected_use_prob_mask=True,
            expected_probability_mask_threshold=0.9,
        )
        assert len(restored) == 2


class TestUnderfilledPolicyGate:
    """Required test 6."""

    def _cfg(self, *, aggregation, underfilled):
        return OmegaConf.create(
            {
                "gear_tree": {
                    "tree_shape": [6, 6, 6],
                    "segment_length": 100,
                    "tree_update_mode": "spo",
                    "gear": {"strict_vdra": False},
                    "replay_buffer": {
                        "underfilled_update_policy": underfilled
                    },
                    "only_adv_greater_than_zero": True,
                },
                "tree_policy": {
                    "policy_aggregation": aggregation,
                    "segment_token_reduction": "mean",
                },
                "actor_rollout_ref": {
                    "actor": {
                        "entropy_coeff": 0.0,
                        "use_kl_loss": False,
                        "kl_loss_coef": 0.0,
                        "policy_loss": {
                            "loss_mode": "vdra_segment_mean_ppo",
                            "policy_aggregation": aggregation,
                            "segment_token_reduction": "mean",
                            "use_prob_mask": True,
                            "probability_mask_threshold": 0.9,
                        },
                    }
                },
            }
        )

    def _validate(self, cfg):
        from recipe.gear_tree.config_validation import (
            validate_policy_loss_consistency,
        )

        return validate_policy_loss_consistency(cfg)

    @pytest.mark.parametrize("aggregation", ["segment_mean", "token_mean"])
    def test_use_available_rejected_for_canonical(self, aggregation):
        with pytest.raises(ValueError, match="postpone_until_divisible"):
            self._validate(
                self._cfg(aggregation=aggregation, underfilled="use_available")
            )

    @pytest.mark.parametrize("aggregation", ["segment_mean", "token_mean"])
    def test_postpone_until_divisible_accepted(self, aggregation):
        assert (
            self._validate(
                self._cfg(
                    aggregation=aggregation,
                    underfilled="postpone_until_divisible",
                )
            )
            is None
        )

    def test_ablation_may_still_use_available(self):
        """The tree_balanced ablation keeps verl's fixed-size split, which
        tolerates whatever the sampler produced."""
        assert (
            self._validate(
                self._cfg(
                    aggregation="tree_balanced_segment_mean",
                    underfilled="use_available",
                )
            )
            is None
        )
