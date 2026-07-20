"""PLAN.md P0.1 — segment_token_reduction config wiring tests.

These tests exercise the real ``PolicyLossConfig`` / ``ActorConfig`` dataclass
schema (not raw YAML/DictConfig), so that a production loss reads exactly the
reduction the config advertises. Any regression that reverts
``segment_token_reduction`` to a silent ``"mean"`` fallback, or that fails to
propagate ``"sum"`` overrides through Hydra composition, will fail here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("hydra")
pytest.importorskip("torch")

import torch  # noqa: E402  after importorskip

from omegaconf import OmegaConf  # noqa: E402
from verl.workers.config.actor import ActorConfig, PolicyLossConfig  # noqa: E402
from recipe.gear_tree.policy_loss import (  # noqa: E402
    _resolve_policy_loss_field,
    _resolve_segment_token_reduction,
)


def _make_policy_loss_cfg(reduction: str = "mean") -> PolicyLossConfig:
    return PolicyLossConfig(
        loss_mode="vdra_segment_mean_ppo",
        segment_token_reduction=reduction,
    )


def _make_actor_cfg(reduction: str = "mean") -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=32,
        policy_loss=_make_policy_loss_cfg(reduction=reduction),
    )


class TestSchemaField:
    def test_default_is_mean(self):
        cfg = _make_policy_loss_cfg()
        assert cfg.segment_token_reduction == "mean"

    def test_sum_is_accepted(self):
        cfg = _make_policy_loss_cfg("sum")
        assert cfg.segment_token_reduction == "sum"

    def test_case_insensitive_normalisation(self):
        cfg = _make_policy_loss_cfg("Sum")
        assert cfg.segment_token_reduction == "sum"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="segment_token_reduction"):
            _make_policy_loss_cfg("average")


class TestResolverLookupLevel:
    """PLAN.md P0.1: `_resolve_segment_token_reduction` must read from
    ``actor.policy_loss.segment_token_reduction``, not from the ActorConfig
    top-level. The reverse (reading top-level) was the silent-mean regression.
    """

    def test_actor_config_delegates_to_policy_loss(self):
        actor_cfg = _make_actor_cfg("sum")
        assert _resolve_segment_token_reduction(actor_cfg) == "sum"

    def test_policy_loss_config_direct(self):
        pl_cfg = _make_policy_loss_cfg("sum")
        # Direct pass-through (older test-only callers) also works.
        assert _resolve_segment_token_reduction(pl_cfg) == "sum"

    def test_missing_field_defaults_to_mean(self):
        # DictConfig with no policy_loss / no reduction key — legacy fallback.
        cfg = OmegaConf.create({})
        assert _resolve_segment_token_reduction(cfg) == "mean"

    def test_dictconfig_actor_with_policy_loss_sum(self):
        cfg = OmegaConf.create(
            {
                "policy_loss": {
                    "loss_mode": "vdra_segment_mean_ppo",
                    "segment_token_reduction": "sum",
                }
            }
        )
        assert _resolve_segment_token_reduction(cfg) == "sum"

    def test_invalid_value_from_dictconfig_raises(self):
        cfg = OmegaConf.create(
            {"policy_loss": {"segment_token_reduction": "average"}}
        )
        with pytest.raises(ValueError, match="segment_token_reduction"):
            _resolve_segment_token_reduction(cfg)


class TestProductionLossReadsSum:
    """PLAN.md P0.1: a production-path `sum` config must reach
    ``compute_policy_loss_vdra_segment_mean`` and produce a numerically
    different result than the same rows under `mean` on non-uniform lengths.
    """

    def _fake_rows(self, active_lens: list[int], max_len: int = 6):
        n = len(active_lens)
        response_mask = torch.zeros((n, max_len))
        for i, k in enumerate(active_lens):
            response_mask[i, :k] = 1.0
        # exp(-0.2) ≈ 0.82 < 0.9, so treetune's use_prob_mask keeps every
        # active token instead of clearing the action mask.
        old_log_prob = torch.full((n, max_len), -0.2)
        log_prob = torch.full((n, max_len), -0.2)
        # Simple per-row advantage; make row losses non-degenerate.
        advantages = torch.full((n, max_len), 0.5)
        return old_log_prob, log_prob, advantages, response_mask

    def test_mean_vs_sum_differ_on_uneven_lengths(self):
        from recipe.gear_tree.policy_loss import compute_policy_loss_vdra_segment_mean

        old_log_prob, log_prob, advantages, response_mask = self._fake_rows(
            active_lens=[2, 4, 6]
        )
        cfg_mean = _make_actor_cfg("mean")
        cfg_sum = _make_actor_cfg("sum")

        # Batch-slot mean loss (PLAN P0.4): pass tree_group_ids + count as a
        # single tree with N_seg = 3, so both losses share the same outer
        # denominator and any difference must come from the within-segment
        # reduction.
        tree_ids = torch.zeros((3,), dtype=torch.long)
        tree_totals = torch.full((3,), 3.0)

        loss_mean, *_ = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg_mean,
            tree_group_ids=tree_ids,
            tree_total_segment_count=tree_totals,
        )
        loss_sum, *_ = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=cfg_sum,
            tree_group_ids=tree_ids,
            tree_total_segment_count=tree_totals,
        )
        # On uneven lengths mean and sum can never coincide as long as row
        # losses are non-zero — cross-check they diverge.
        assert not torch.isclose(loss_mean, loss_sum), (
            f"mean {loss_mean.item()} and sum {loss_sum.item()} must differ on "
            "uneven active lengths"
        )


class TestHydraComposition:
    """PLAN.md P0.1 acceptance: Hydra-compose the real main config and
    instantiate the real ``ActorConfig`` / ``PolicyLossConfig``.

    Notes
    -----
    The gear_tree_trainer.yaml composes verl's ``ppo_trainer`` base via a
    Hydra searchpath. If the runtime does not have Hydra installed we skip;
    the raw YAML-shape check is covered elsewhere.
    """

    def test_canonical_config_mean(self):
        from pathlib import Path
        import yaml

        cfg_path = Path(
            "verl/recipe/gear_tree/config/gear_tree_trainer.yaml"
        ).resolve()
        raw = yaml.safe_load(cfg_path.read_text())
        assert raw["tree_policy"]["segment_token_reduction"] == "mean"
        assert (
            raw["actor_rollout_ref"]["actor"]["policy_loss"][
                "segment_token_reduction"
            ]
            == "mean"
        )
        assert (
            raw["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"]
            == "vdra_segment_mean_ppo"
        )
        # Instantiate the real dataclass with the canonical value.
        pl = PolicyLossConfig(
            loss_mode="vdra_segment_mean_ppo",
            segment_token_reduction=raw["actor_rollout_ref"]["actor"][
                "policy_loss"
            ]["segment_token_reduction"],
        )
        assert pl.segment_token_reduction == "mean"

    def test_sum_override_composes(self):
        pl = PolicyLossConfig(
            loss_mode="vdra_segment_mean_ppo",
            segment_token_reduction="sum",
        )
        assert pl.segment_token_reduction == "sum"

    @staticmethod
    def _compose_real(overrides=None):
        # PLAN.md M5: REAL hydra.compose of the shipped main config,
        # including the pkg://verl.trainer.config searchpath.
        pytest.importorskip("hydra")
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        config_dir = Path(__file__).resolve().parents[1] / "config"
        with initialize_config_dir(
            config_dir=str(config_dir), version_base=None
        ):
            return compose(
                config_name="gear_tree_trainer", overrides=overrides or []
            )

    def test_real_compose_canonical_mean(self):
        cfg = self._compose_real()
        actor = cfg.actor_rollout_ref.actor
        assert actor.policy_loss.loss_mode == "vdra_segment_mean_ppo"
        assert actor.policy_loss.segment_token_reduction == "mean"
        assert cfg.tree_policy.segment_token_reduction == "mean"
        assert cfg.tree_policy.policy_aggregation == "global_segment_mean"

    def test_real_compose_sum_override_reaches_actor_policy_loss(self):
        cfg = self._compose_real(
            overrides=[
                "tree_policy.segment_token_reduction=sum",
                "actor_rollout_ref.actor.policy_loss.segment_token_reduction=sum",
            ]
        )
        assert (
            cfg.actor_rollout_ref.actor.policy_loss.segment_token_reduction
            == "sum"
        )
        assert cfg.tree_policy.segment_token_reduction == "sum"

    def test_real_compose_passes_extracted_trainer_validation(self):
        from recipe.gear_tree.config_validation import (
            validate_policy_loss_consistency,
        )

        assert validate_policy_loss_consistency(self._compose_real()) is None

    def test_real_compose_round_trips_typed_policy_loss(self):
        cfg = self._compose_real()
        from omegaconf import OmegaConf

        from verl.utils.config import omega_conf_to_dataclass

        pl_node = OmegaConf.create(
            OmegaConf.to_container(
                cfg.actor_rollout_ref.actor.policy_loss, resolve=True
            )
        )
        pl = omega_conf_to_dataclass(pl_node, dataclass_type=PolicyLossConfig)
        assert pl.loss_mode == "vdra_segment_mean_ppo"
        assert pl.segment_token_reduction == "mean"


class TestPolicyLossFieldReads:
    """PLAN.md P0.F: ``use_prob_mask`` / ``ratio_threshold`` are declared on
    ``PolicyLossConfig`` and must be read from ``config.policy_loss.*``, not
    from the ActorConfig top level."""

    def test_actor_config_policy_loss_level_is_read(self):
        cfg = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=32,
            policy_loss=PolicyLossConfig(
                loss_mode="vdra_segment_mean_ppo",
                use_prob_mask=False,
                ratio_threshold=5.0,
            ),
        )
        assert _resolve_policy_loss_field(cfg, "use_prob_mask", True) is False
        assert _resolve_policy_loss_field(cfg, "ratio_threshold", 10.0) == 5.0

    def test_wrong_level_duplicate_is_ignored(self):
        cfg = OmegaConf.create(
            {
                "use_prob_mask": True,  # wrong level — must never be read
                "ratio_threshold": 99.0,  # wrong level — must never be read
                "policy_loss": {
                    "use_prob_mask": False,
                    "ratio_threshold": 5.0,
                },
            }
        )
        assert _resolve_policy_loss_field(cfg, "use_prob_mask", True) is False
        assert _resolve_policy_loss_field(cfg, "ratio_threshold", 10.0) == 5.0

    def test_bare_policy_loss_config_direct(self):
        pl = PolicyLossConfig(
            loss_mode="vdra_segment_mean_ppo",
            use_prob_mask=False,
            ratio_threshold=7.5,
        )
        assert _resolve_policy_loss_field(pl, "use_prob_mask", True) is False
        assert _resolve_policy_loss_field(pl, "ratio_threshold", 10.0) == 7.5

    def test_missing_everywhere_falls_back_to_default(self):
        cfg = OmegaConf.create({})
        assert _resolve_policy_loss_field(cfg, "use_prob_mask", True) is True
        assert (
            _resolve_policy_loss_field(cfg, "ratio_threshold", 10.0) == 10.0
        )

    def test_use_prob_mask_override_changes_production_loss(self):
        """An override under actor.policy_loss must change actual loss
        output: with the prob mask on, tokens whose old probability is
        >= 0.9 are excluded from the surrogate."""
        from recipe.gear_tree.policy_loss import (
            compute_policy_loss_vdra_segment_mean,
        )

        n, t = 2, 4
        # High-probability old tokens (exp(-0.01) ~ 0.99): masked out when
        # use_prob_mask=True, included when False.
        old_log_prob = torch.full((n, t), -0.01)
        log_prob = torch.full((n, t), -0.5)
        advantages = torch.full((n, t), 1.0)
        response_mask = torch.ones((n, t))

        def _cfg(use_prob_mask: bool) -> ActorConfig:
            return ActorConfig(
                strategy="fsdp",
                rollout_n=1,
                ppo_micro_batch_size_per_gpu=32,
                policy_loss=PolicyLossConfig(
                    loss_mode="vdra_segment_mean_ppo",
                    use_prob_mask=use_prob_mask,
                ),
            )

        seg_counts = torch.full((n,), float(t))
        loss_masked, *_ = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=_cfg(True),
            tree_total_segment_count=seg_counts,
            original_optimizer_batch_tree_count=1,
        )
        loss_unmasked, *_ = compute_policy_loss_vdra_segment_mean(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=_cfg(False),
            tree_total_segment_count=seg_counts,
            original_optimizer_batch_tree_count=1,
        )
        assert torch.isclose(loss_masked, torch.tensor(0.0))
        assert not torch.isclose(loss_unmasked, loss_masked)


class TestStartupConsistencyCheck:
    """PLAN.md P0.1: the trainer must refuse a config where
    ``tree_policy.segment_token_reduction`` and
    ``actor.policy_loss.segment_token_reduction`` disagree.

    We exercise the pure validation function without spinning up Ray.
    """

    def _minimal_config(self, tree_reduction: str, actor_reduction: str):
        return OmegaConf.create(
            {
                "gear_tree": {
                    "tree_shape": [6, 6, 6],
                    "segment_length": 100,
                    "gear": {"strict_vdra": True},
                    "replay_buffer": {},
                },
                "tree_policy": {
                    "policy_aggregation": "global_segment_mean",
                    "segment_token_reduction": tree_reduction,
                },
                "actor_rollout_ref": {
                    "actor": {
                        "policy_loss": {
                            "loss_mode": "vdra_segment_mean_ppo",
                            "segment_token_reduction": actor_reduction,
                        }
                    }
                },
            }
        )

    def test_matching_values_pass(self):
        # PLAN.md M5: run the REAL extracted trainer validation, not a
        # mirrored re-implementation.
        from recipe.gear_tree.config_validation import (
            validate_policy_loss_consistency,
        )

        assert (
            validate_policy_loss_consistency(
                self._minimal_config("mean", "mean")
            )
            is None
        )
        assert (
            validate_policy_loss_consistency(
                self._minimal_config("sum", "sum")
            )
            is None
        )

    def test_mismatched_values_are_rejected(self):
        from recipe.gear_tree.config_validation import (
            validate_policy_loss_consistency,
        )

        with pytest.raises(ValueError, match="P0.1"):
            validate_policy_loss_consistency(
                self._minimal_config("mean", "sum")
            )

    def test_invalid_reduction_is_rejected(self):
        from recipe.gear_tree.config_validation import (
            validate_policy_loss_consistency,
        )

        with pytest.raises(ValueError, match="exactly 'mean' or"):
            validate_policy_loss_consistency(
                self._minimal_config("median", "median")
            )

    def test_wrong_loss_mode_for_canonical_aggregation_is_rejected(self):
        from recipe.gear_tree.config_validation import (
            validate_policy_loss_consistency,
        )

        cfg = self._minimal_config("mean", "mean")
        cfg.actor_rollout_ref.actor.policy_loss.loss_mode = (
            "vdra_node_balanced_ppo"
        )
        with pytest.raises(ValueError, match="vdra_segment_mean_ppo"):
            validate_policy_loss_consistency(cfg)

    def test_trainer_startup_routes_through_extracted_validator(self):
        # Source guard: the trainer must call the extracted function so the
        # gate and the trainer can never diverge.
        import inspect

        from recipe.gear_tree import gear_ray_trainer

        source = inspect.getsource(
            gear_ray_trainer.RayGearTreeTrainer._validate_replay_startup
        )
        assert "validate_policy_loss_consistency(" in source
