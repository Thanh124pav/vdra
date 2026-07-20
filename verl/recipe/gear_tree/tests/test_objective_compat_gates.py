"""PLAN.md §10/§11/§13 correction spec: strict gates + replay compatibility.

Required tests 18-22:

18. strict mode rejects entropy/KL with sparse zero-slot execution;
19. strict mode rejects ``global_segment_mean``;
20. non-strict mode maps it to ``tree_balanced_segment_mean`` with a warning
    (and NEVER to ``segment_mean``);
21. replay restore rejects a probability-mask threshold mismatch;
22. replay restore rejects legacy zero slots lacking active-token counts
    when masked token mean is selected.
"""

from __future__ import annotations

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()

pytest.importorskip("torch")

import json  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

from recipe.gear_tree.config_validation import (  # noqa: E402
    validate_policy_loss_consistency,
)
from recipe.gear_tree.replay_buffer import (  # noqa: E402
    LOGICAL_SLOT_SCHEMA_VERSION,
    GearTreeReplayBuffer,
)


def _config(
    *,
    strict=True,
    policy_aggregation="segment_mean",
    actor_aggregation=None,
    entropy_coeff=0.0,
    use_kl_loss=False,
    kl_loss_coef=0.0,
):
    return OmegaConf.create(
        {
            "gear_tree": {
                "tree_shape": [6, 6, 6],
                "segment_length": 100,
                "tree_update_mode": "spo",
                "gear": {"strict_vdra": strict},
                "replay_buffer": {},
                "only_adv_greater_than_zero": True,
            },
            "tree_policy": {
                "policy_aggregation": policy_aggregation,
                "segment_token_reduction": "mean",
            },
            "actor_rollout_ref": {
                "actor": {
                    "entropy_coeff": entropy_coeff,
                    "use_kl_loss": use_kl_loss,
                    "kl_loss_coef": kl_loss_coef,
                    "policy_loss": {
                        "loss_mode": "vdra_segment_mean_ppo",
                        "policy_aggregation": (
                            actor_aggregation
                            if actor_aggregation is not None
                            else policy_aggregation
                        ),
                        "segment_token_reduction": "mean",
                        "use_prob_mask": True,
                        "probability_mask_threshold": 0.9,
                    },
                }
            },
        }
    )


class TestAuxiliaryObjectiveGate:
    """Required test 18: sparse omission preserves the policy-gradient term
    only, so strict canonical sparse mode must refuse entropy/KL."""

    def test_strict_rejects_entropy_coeff(self):
        with pytest.raises(ValueError, match="entropy_coeff"):
            validate_policy_loss_consistency(_config(entropy_coeff=0.01))

    def test_strict_rejects_use_kl_loss(self):
        with pytest.raises(ValueError, match="use_kl_loss"):
            validate_policy_loss_consistency(_config(use_kl_loss=True))

    def test_strict_rejects_kl_loss_coef(self):
        with pytest.raises(ValueError, match="kl_loss_coef"):
            validate_policy_loss_consistency(_config(kl_loss_coef=0.05))

    def test_strict_accepts_all_zero_auxiliaries(self):
        assert validate_policy_loss_consistency(_config()) is None

    def test_non_strict_allows_them_as_labeled_ablation(self):
        # Non-strict is an explicitly objective-changing ablation.
        assert (
            validate_policy_loss_consistency(
                _config(
                    strict=False,
                    policy_aggregation="tree_balanced_segment_mean",
                    entropy_coeff=0.01,
                )
            )
            is None
        )


class TestGlobalSegmentMeanCompat:
    def test_strict_rejects_legacy_name(self):
        """Required test 19."""
        with pytest.raises(ValueError, match="renamed"):
            validate_policy_loss_consistency(
                _config(policy_aggregation="global_segment_mean")
            )

    def test_non_strict_maps_to_tree_balanced_with_warning(self):
        """Required test 20: mapped to the ABLATION, never to segment_mean,
        and both duplicated fields are canonicalized end to end."""
        cfg = _config(strict=False, policy_aggregation="global_segment_mean")
        with pytest.warns(DeprecationWarning, match="global_segment_mean"):
            assert validate_policy_loss_consistency(cfg) is None
        assert cfg.tree_policy.policy_aggregation == "tree_balanced_segment_mean"
        assert (
            cfg.actor_rollout_ref.actor.policy_loss.policy_aggregation
            == "tree_balanced_segment_mean"
        )

    def test_legacy_name_never_becomes_segment_mean(self):
        cfg = _config(strict=False, policy_aggregation="global_segment_mean")
        with pytest.warns(DeprecationWarning):
            validate_policy_loss_consistency(cfg)
        assert cfg.tree_policy.policy_aggregation != "segment_mean"

    def test_actor_side_legacy_name_is_canonicalized_too(self):
        cfg = _config(
            strict=False,
            policy_aggregation="tree_balanced_segment_mean",
            actor_aggregation="global_segment_mean",
        )
        with pytest.warns(DeprecationWarning):
            validate_policy_loss_consistency(cfg)
        assert (
            cfg.actor_rollout_ref.actor.policy_loss.policy_aggregation
            == "tree_balanced_segment_mean"
        )


def _slot(edge_id: str, threshold: float = 0.9, active: int = 2):
    return {
        "edge_id": edge_id,
        "question_id": "q0",
        "tree_id": "t0",
        "parent_group_id": "t0/pg",
        "policy_snapshot_id": "snap",
        "generation_rollout_iteration": 0,
        "advantage": 0.0,
        "advantage_is_zero": True,
        "trainable_edge_id": None,
        "response_token_count": 3,
        "prob_mask_token_count": active,
        "probability_mask_threshold": threshold,
    }


def _saved_buffer(tmp_path, *, use_prob_mask=True, threshold=0.9):
    buf = GearTreeReplayBuffer(
        target_edges_per_iteration=8,
        max_edge_age_iterations=4,
        max_edges_per_question_per_iteration=8,
        use_prob_mask=use_prob_mask,
        probability_mask_threshold=threshold,
    )
    buf.add(
        [_slot("z0", threshold=threshold)],
        generation_rollout_iteration=0,
        policy_snapshot_id="snap",
    )
    buf.save(tmp_path)
    return buf


class TestReplayObjectiveCompat:
    def test_matching_configuration_restores(self, tmp_path):
        _saved_buffer(tmp_path, use_prob_mask=True, threshold=0.9)
        restored = GearTreeReplayBuffer.load(
            tmp_path,
            expected_use_prob_mask=True,
            expected_probability_mask_threshold=0.9,
        )
        assert len(restored) == 1

    def test_threshold_mismatch_fails_fast(self, tmp_path):
        """Required test 21."""
        _saved_buffer(tmp_path, use_prob_mask=True, threshold=0.9)
        with pytest.raises(ValueError, match="probability_mask_threshold"):
            GearTreeReplayBuffer.load(
                tmp_path,
                expected_use_prob_mask=True,
                expected_probability_mask_threshold=0.75,
            )

    def test_mask_mode_mismatch_fails_fast(self, tmp_path):
        _saved_buffer(tmp_path, use_prob_mask=False, threshold=0.9)
        with pytest.raises(ValueError, match="use_prob_mask"):
            GearTreeReplayBuffer.load(
                tmp_path,
                expected_use_prob_mask=True,
                expected_probability_mask_threshold=0.9,
            )

    def test_legacy_checkpoint_without_mask_metadata_fails_fast(self, tmp_path):
        """Required test 22: a pre-schema-2 replay has no verifiable
        active-token counts, and they can never be recomputed."""
        _saved_buffer(tmp_path)
        meta_path = tmp_path / "gear_tree_replay_buffer_meta.json"
        meta = json.loads(meta_path.read_text())
        meta.pop("use_prob_mask")
        meta.pop("probability_mask_threshold")
        meta["logical_slot_schema_version"] = 1
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="predates|schema"):
            GearTreeReplayBuffer.load(
                tmp_path,
                expected_use_prob_mask=True,
                expected_probability_mask_threshold=0.9,
            )

    def test_explicit_reset_discards_instead_of_failing(self, tmp_path):
        _saved_buffer(tmp_path, use_prob_mask=True, threshold=0.9)
        with pytest.warns(RuntimeWarning, match="discarding replay rows"):
            restored = GearTreeReplayBuffer.load(
                tmp_path,
                expected_use_prob_mask=True,
                expected_probability_mask_threshold=0.5,
                reset_replay_on_objective_mismatch=True,
            )
        assert len(restored) == 0
        assert restored.metrics["replay_reset_on_objective_mismatch"] == 1

    def test_saved_metadata_records_the_objective_identity(self, tmp_path):
        _saved_buffer(tmp_path, use_prob_mask=False, threshold=0.75)
        meta = json.loads(
            (tmp_path / "gear_tree_replay_buffer_meta.json").read_text()
        )
        assert meta["use_prob_mask"] is False
        assert meta["probability_mask_threshold"] == 0.75
        assert meta["logical_slot_schema_version"] == LOGICAL_SLOT_SCHEMA_VERSION

    def test_slot_without_active_count_is_rejected_on_insert(self):
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=8,
            max_edge_age_iterations=4,
            max_edges_per_question_per_iteration=8,
        )
        bad = _slot("z0")
        bad.pop("prob_mask_token_count")
        with pytest.raises(ValueError, match="missing required fields"):
            buf.add(
                [bad], generation_rollout_iteration=0, policy_snapshot_id="snap"
            )

    def test_slot_active_count_above_response_count_rejected(self):
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=8,
            max_edge_age_iterations=4,
            max_edges_per_question_per_iteration=8,
        )
        bad = _slot("z0", active=99)
        with pytest.raises(ValueError, match="prob_mask_token_count"):
            buf.add(
                [bad], generation_rollout_iteration=0, policy_snapshot_id="snap"
            )
