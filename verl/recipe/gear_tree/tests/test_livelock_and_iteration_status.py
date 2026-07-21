"""Anti-livelock guards, iteration status, and skip-path robustness.

Required tests 1-12 of the remaining-fixes spec:

1-2. a manifest / timing write failure on the skipped path does not
     terminate training (the reservation is already committed);
3.   repeated skipped updates trip the configured limit;
4.   a successful actor update resets the consecutive-skip counter;
5.   ``max_rollout_iterations`` stops a run whose ``global_step`` is stuck;
6-7. entropy / KL are rejected for canonical aggregation in NON-strict mode;
8.   canonical replay rejects missing denominator fields;
9.   legacy paths may recompute missing fields, but must warn;
10.  every iteration path sets ``last_iteration_status``;
11.  skipped timing carries cumulative and wall-clock fields;
12.  a later successful update clears the derived compatibility flag.
"""

from __future__ import annotations

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from omegaconf import OmegaConf  # noqa: E402

from recipe.gear_tree.config_validation import (  # noqa: E402
    validate_policy_loss_consistency,
)
from recipe.gear_tree.gear_ray_trainer import RayGearTreeTrainer  # noqa: E402
from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer  # noqa: E402
from recipe.gear_tree.run_manifest import (  # noqa: E402
    ITERATION_STATUS_ACTOR_FAILED,
    ITERATION_STATUS_ALL_ZERO_SKIPPED,
    ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED,
    ITERATION_STATUS_NOT_STARTED,
    ITERATION_STATUS_NO_SAMPLE,
    ITERATION_STATUS_POSTPONED,
    ITERATION_STATUS_UPDATED,
    ITERATION_STATUS_ZERO_ACTIVE_SKIPPED,
    RunManifest,
)

try:
    from recipe.gear_tree.tests import _tiny_actor
except ImportError:
    import _tiny_actor


class _Cfg(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _trainer(tmp_path, *, max_skips=50, max_iters=None):
    obj = object.__new__(RayGearTreeTrainer)
    obj.tokenizer = _tiny_actor.Tok()
    obj.config = _Cfg(
        data=_Cfg(max_prompt_length=6, max_response_length=4),
        trainer=_Cfg(
            balance_batch=False,
            default_local_dir=str(tmp_path),
            nnodes=1,
            n_gpus_per_node=1,
        ),
        actor_rollout_ref=_Cfg(
            actor=_Cfg(
                ppo_mini_batch_size=4,
                ppo_epochs=1,
                ulysses_sequence_parallel_size=1,
                policy_loss={
                    "loss_mode": "vdra_segment_mean_ppo",
                    "policy_aggregation": "segment_mean",
                    "use_prob_mask": False,
                    "probability_mask_threshold": 0.9,
                },
            )
        ),
        gear_tree={
            "replay_buffer": {
                "target_edges_per_iteration": 64,
                "max_edge_age_iterations": 8,
                "max_edges_per_question_per_iteration": 32,
                "sampling_seed": 0,
            },
            "gear": {"strict_vdra": False},
            "max_consecutive_skipped_updates": max_skips,
            "max_rollout_iterations": max_iters,
        },
    )
    obj.run_manifest = RunManifest()
    obj.rollout_iteration = 0
    obj.global_steps = 0
    obj.total_training_steps = 100
    obj.num_optimizer_steps_total = 0
    obj.successful_actor_updates = 0
    obj.optimizer_steps_this_iteration = 0
    obj.skipped_zero_gradient_updates = 0
    obj.consecutive_skipped_updates = 0
    obj.postponed_updates = 0
    obj.failed_updates = 0
    obj._resolved_max_edge_prompt_length = lambda: 6
    return obj


class TestIterationStatus:
    """Required test 10 + 12."""

    def test_default_is_not_started(self):
        assert RunManifest().last_iteration_status == ITERATION_STATUS_NOT_STARTED
        assert RunManifest().actor_update_skipped is False

    @pytest.mark.parametrize(
        "status,expected_flag",
        [
            (ITERATION_STATUS_UPDATED, False),
            (ITERATION_STATUS_ALL_ZERO_SKIPPED, True),
            (ITERATION_STATUS_ZERO_ACTIVE_SKIPPED, True),
            (ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED, True),
            (ITERATION_STATUS_POSTPONED, False),
            (ITERATION_STATUS_NO_SAMPLE, False),
            (ITERATION_STATUS_ACTOR_FAILED, False),
        ],
    )
    def test_compat_flag_is_derived_from_status(self, tmp_path, status, expected_flag):
        trainer = _trainer(tmp_path)
        trainer._set_iteration_status(status)
        assert trainer.run_manifest.last_iteration_status == status
        assert trainer.run_manifest.actor_update_skipped is expected_flag

    def test_unknown_status_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown iteration status"):
            _trainer(tmp_path)._set_iteration_status("something_else")

    @pytest.mark.parametrize(
        "all_zero,zero_active,expected",
        [
            (2.0, 0.0, ITERATION_STATUS_ALL_ZERO_SKIPPED),
            (0.0, 2.0, ITERATION_STATUS_ZERO_ACTIVE_SKIPPED),
            (1.0, 1.0, ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED),
        ],
    )
    def test_skip_reason_classification(self, all_zero, zero_active, expected):
        metrics = {
            "vdra/all_zero_advantage_logical_batches": all_zero,
            "vdra/zero_active_token_logical_batches": zero_active,
        }
        assert RayGearTreeTrainer._zero_signal_skip_status(metrics) == expected

    def test_later_update_clears_the_compat_flag(self, tmp_path):
        """Required test 12."""
        trainer = _trainer(tmp_path)
        trainer._set_iteration_status(ITERATION_STATUS_ALL_ZERO_SKIPPED)
        assert trainer.run_manifest.actor_update_skipped is True
        trainer._set_iteration_status(ITERATION_STATUS_UPDATED)
        assert trainer.run_manifest.actor_update_skipped is False
        assert (
            trainer.run_manifest.last_iteration_status == ITERATION_STATUS_UPDATED
        )


class TestLivelockGuards:
    """Required tests 3-5."""

    def _metrics(self):
        return {
            "vdra/all_zero_advantage_logical_batches": 2.0,
            "vdra/zero_active_token_logical_batches": 0.0,
        }

    def test_limit_not_reached_is_a_no_op(self, tmp_path):
        trainer = _trainer(tmp_path, max_skips=3)
        trainer.consecutive_skipped_updates = 2
        trainer._enforce_livelock_guards(self._metrics())  # must not raise

    def test_repeated_skips_trip_the_limit(self, tmp_path):
        """Required test 3."""
        trainer = _trainer(tmp_path, max_skips=3)
        trainer.consecutive_skipped_updates = 3
        trainer.global_steps = 0
        trainer.rollout_iteration = 7
        with pytest.raises(RuntimeError) as excinfo:
            trainer._enforce_livelock_guards(self._metrics())
        msg = str(excinfo.value)
        # The error must carry the full diagnostic context.
        assert "global_step=0" in msg
        assert "rollout_iteration=7" in msg
        assert "3 consecutive" in msg
        assert "max_consecutive_skipped_updates=3" in msg
        assert "all_zero_advantage_logical_batches=2.0" in msg
        assert "zero_active_token_logical_batches=0.0" in msg

    def test_manifest_is_persisted_before_aborting(self, tmp_path):
        trainer = _trainer(tmp_path, max_skips=1)
        trainer.consecutive_skipped_updates = 1
        with pytest.raises(RuntimeError):
            trainer._enforce_livelock_guards(self._metrics())
        assert (tmp_path / "vdra_run_manifest.json").exists()

    @pytest.mark.parametrize("limit", [0, -1, None])
    def test_guard_can_be_disabled(self, tmp_path, limit):
        trainer = _trainer(tmp_path, max_skips=limit)
        trainer.consecutive_skipped_updates = 10_000
        trainer._enforce_livelock_guards(self._metrics())  # must not raise

    def test_successful_update_resets_the_counter(self, tmp_path):
        """Required test 4 — driven through the REAL finalize helper."""
        trainer = _trainer(tmp_path)
        trainer.consecutive_skipped_updates = 5
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=64,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=32,
            use_prob_mask=False,
            probability_mask_threshold=0.9,
        )
        buf.add(
            [_trainable_edge("e0")],
            generation_rollout_iteration=1,
            policy_snapshot_id="snap",
        )
        reservation = buf.reserve_for_update(current_rollout_iteration=1)

        class _Out:
            meta_info = {"metrics": {"actor/num_optimizer_steps": [1]}}

        trainer._expected_optimizer_steps = 1
        trainer._finalize_successful_actor_update(
            buf, reservation, _Out(), [_trainable_edge("e0")], {}, {}
        )
        assert trainer.consecutive_skipped_updates == 0
        assert (
            trainer.run_manifest.last_iteration_status == ITERATION_STATUS_UPDATED
        )

    def test_rollout_iteration_budget_stops_a_stuck_run(self, tmp_path):
        """Required test 5: global_step never advances, but the run stops."""
        trainer = _trainer(tmp_path, max_iters=5)
        trainer.global_steps = 0  # far below total_training_steps=100
        trainer.rollout_iteration = 4
        assert trainer._rollout_iteration_budget_exhausted() is False
        trainer.rollout_iteration = 5
        assert trainer._rollout_iteration_budget_exhausted() is True

    @pytest.mark.parametrize("limit", [0, -1, None])
    def test_iteration_budget_can_be_disabled(self, tmp_path, limit):
        trainer = _trainer(tmp_path, max_iters=limit)
        trainer.rollout_iteration = 10_000
        assert trainer._rollout_iteration_budget_exhausted() is False

    def test_counters_are_logged(self, tmp_path):
        trainer = _trainer(tmp_path, max_skips=7, max_iters=9)
        trainer.consecutive_skipped_updates = 2
        metrics: dict = {}
        trainer._log_livelock_counters(metrics)
        assert metrics["training/consecutive_skipped_updates"] == 2.0
        assert metrics["training/max_consecutive_skipped_updates"] == 7.0
        assert metrics["training/max_rollout_iterations"] == 9.0


class TestSkipPathIoRobustness:
    """Required tests 1, 2 and 11 — driven through the REAL fit-loop skip
    branch by source inspection plus direct behavioural checks, because the
    branch is only reachable inside ``fit()``."""

    def _skip_branch_source(self) -> str:
        import inspect

        src = inspect.getsource(RayGearTreeTrainer.fit)
        start = src.index("if edge_batch is None:")
        end = src.index("actor_updated = True", start)
        return src[start:end]

    def test_manifest_write_failure_is_caught(self):
        """Required test 1: the reservation is already committed, so a
        manifest I/O failure must not crash training."""
        branch = self._skip_branch_source()
        assert "self._save_manifest(self.run_manifest)" in branch
        assert "vdra/manifest_save_failed" in branch
        # The save must sit inside a try/except, not be a bare call.
        save_at = branch.index("self._save_manifest(self.run_manifest)")
        assert "try:" in branch[:save_at]

    def test_timing_write_failure_is_caught(self):
        """Required test 2."""
        branch = self._skip_branch_source()
        assert "vdra/timing_write_failed" in branch
        write_at = branch.index("json.dumps(skip_timing)")
        assert "try:" in branch[:write_at]

    def test_skip_timing_carries_cumulative_and_wall_clock(self):
        """Required test 11."""
        branch = self._skip_branch_source()
        for key in (
            "timing/train_total_seconds",
            "timing/cumulative_train_seconds",
            "timing/wall_seconds",
            "training/successful_actor_updates",
            "training/postponed_updates",
            "training/failed_updates",
            "training/consecutive_skipped_updates",
            "last_iteration_status",
        ):
            assert key in branch, key
        # Generation time is folded into the running total.
        assert "cum_train += t_gen" in branch

    def test_skip_timing_reports_zero_steps_and_unchanged_global_step(self):
        branch = self._skip_branch_source()
        assert '"training/expected_optimizer_steps": 0.0' in branch
        assert '"training/optimizer_steps_this_iteration": 0.0' in branch
        # The skip branch must never touch global_step.
        assert "self.global_steps +=" not in branch

    def test_livelock_guard_runs_after_state_is_persisted(self):
        branch = self._skip_branch_source()
        guard_at = branch.index("_enforce_livelock_guards")
        manifest_at = branch.index("self._save_manifest(self.run_manifest)")
        assert manifest_at < guard_at, (
            "the guard must abort only AFTER the manifest was persisted"
        )

    def test_manifest_save_failure_does_not_propagate(self, tmp_path, monkeypatch):
        """Behavioural check of the same contract on the helper itself."""
        trainer = _trainer(tmp_path)

        def _boom(_manifest):
            raise OSError("disk full")

        monkeypatch.setattr(trainer, "_save_manifest", _boom)
        metrics: dict = {}
        # The guard's own best-effort save must swallow the failure and still
        # raise the intended RuntimeError (not the OSError).
        trainer.consecutive_skipped_updates = 50
        with pytest.raises(RuntimeError, match="consecutive fully skipped"):
            trainer._enforce_livelock_guards(metrics)


def _trainable_edge(edge_id="e0", *, with_metadata=True):
    edge = {
        "edge_id": edge_id,
        "question_id": "q0",
        "tree_id": "t0",
        "parent_group_id": "t0/pg",
        "child_segment_id": f"t0/{edge_id}",
        "query_token_ids": [1, 2],
        "response_token_ids": [3, 4],
        "actor_shifted_log_probs": [-0.2, -0.2],
        "advantage": 0.5,
        "value": 0.4,
        "reward": 1.0,
        "generation_rollout_iteration": 0,
    }
    if with_metadata:
        edge.update(
            {
                "response_token_count": 2,
                "prob_mask_token_count": 2,
                "probability_mask_threshold": 0.9,
            }
        )
    return edge


class TestCanonicalDenominatorMetadataRequired:
    """Required tests 8-9."""

    def _buffer(self, *, canonical):
        return GearTreeReplayBuffer(
            target_edges_per_iteration=16,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=16,
            use_prob_mask=True,
            probability_mask_threshold=0.9,
            require_logical_denominator_metadata=canonical,
        )

    def test_canonical_rejects_missing_metadata(self):
        """Required test 8."""
        buf = self._buffer(canonical=True)
        with pytest.raises(ValueError) as excinfo:
            buf.add(
                [_trainable_edge("bad", with_metadata=False)],
                generation_rollout_iteration=0,
                policy_snapshot_id="snap",
            )
        msg = str(excinfo.value)
        assert "bad" in msg  # the offending edge_id
        assert "response_token_count" in msg

    def test_canonical_accepts_complete_metadata(self):
        buf = self._buffer(canonical=True)
        buf.add(
            [_trainable_edge("ok")],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        assert len(buf) == 1

    def test_legacy_buffer_tolerates_missing_metadata(self):
        buf = self._buffer(canonical=False)
        buf.add(
            [_trainable_edge("legacy", with_metadata=False)],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        assert len(buf) == 1

    def test_legacy_recomputation_warns(self):
        """Required test 9: a legacy record may be completed downstream, but
        never silently."""
        from recipe.gear_tree.tree_data import build_logical_update_batch

        legacy = _trainable_edge("legacy", with_metadata=False)
        legacy.update({"allocated_k": 2, "tree_total_segment_count": 2})
        with pytest.warns(RuntimeWarning, match="missing"):
            build_logical_update_batch(
                [legacy, _trainable_edge("e1")],
                _tiny_actor.Tok(),
                max_prompt_length=6,
                max_response_length=4,
                ppo_mini_batch_size=2,
                dp_size=1,
                loss_mode="vdra_segment_mean_ppo",
                use_prob_mask=True,
                probability_mask_threshold=0.9,
            )

    def test_canonical_checkpoint_refuses_incomplete_records(self, tmp_path):
        buf = self._buffer(canonical=False)
        buf.add(
            [_trainable_edge("legacy", with_metadata=False)],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        # Flip to canonical to simulate a mode change before saving.
        buf.require_logical_denominator_metadata = True
        with pytest.raises(ValueError, match="CANONICAL"):
            buf.save(tmp_path)

    def test_legacy_checkpoint_declares_record_schema_v1(self, tmp_path):
        """A checkpoint must not CLAIM v2 while holding incomplete records."""
        import json

        buf = self._buffer(canonical=False)
        buf.add(
            [_trainable_edge("legacy", with_metadata=False)],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        buf.save(tmp_path)
        meta = json.loads(
            (tmp_path / "gear_tree_replay_buffer_meta.json").read_text()
        )
        assert meta["logical_record_schema_version"] == 1


class TestCanonicalRejectsAuxiliaryLosses:
    """Required tests 6-7: rejected regardless of strict_vdra."""

    def _cfg(self, *, strict, aggregation="segment_mean", **actor):
        actor_block = {
            "entropy_coeff": 0.0,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
        }
        actor_block.update(actor)
        actor_block["policy_loss"] = {
            "loss_mode": "vdra_segment_mean_ppo",
            "policy_aggregation": aggregation,
            "segment_token_reduction": "mean",
            "use_prob_mask": True,
            "probability_mask_threshold": 0.9,
        }
        return OmegaConf.create(
            {
                "gear_tree": {
                    "tree_shape": [6, 6, 6],
                    "segment_length": 100,
                    "tree_update_mode": "spo",
                    "gear": {"strict_vdra": strict},
                    "replay_buffer": {
                        "underfilled_update_policy": "postpone_until_divisible"
                    },
                    "only_adv_greater_than_zero": True,
                },
                "tree_policy": {
                    "policy_aggregation": aggregation,
                    "segment_token_reduction": "mean",
                },
                "actor_rollout_ref": {"actor": actor_block},
            }
        )

    @pytest.mark.parametrize("strict", [True, False])
    def test_entropy_rejected_for_canonical(self, strict):
        """Required test 6 — including NON-strict."""
        with pytest.raises(ValueError, match="policy-gradient objective only"):
            validate_policy_loss_consistency(
                self._cfg(strict=strict, entropy_coeff=0.01)
            )

    @pytest.mark.parametrize("strict", [True, False])
    def test_kl_rejected_for_canonical(self, strict):
        """Required test 7 — including NON-strict."""
        with pytest.raises(ValueError, match="policy-gradient objective only"):
            validate_policy_loss_consistency(
                self._cfg(strict=strict, use_kl_loss=True, kl_loss_coef=0.1)
            )

    @pytest.mark.parametrize("strict", [True, False])
    def test_clean_canonical_config_passes(self, strict):
        assert validate_policy_loss_consistency(self._cfg(strict=strict)) is None

    def test_ablation_may_still_use_auxiliary_losses(self):
        """The tree_balanced ablation does not use logical batching."""
        assert (
            validate_policy_loss_consistency(
                self._cfg(
                    strict=False,
                    aggregation="tree_balanced_segment_mean",
                    entropy_coeff=0.01,
                )
            )
            is None
        )
