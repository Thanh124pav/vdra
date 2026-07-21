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

import inspect
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
    ITERATION_STATUS_RUNNING,
    ITERATION_STATUS_UPDATED,
    ITERATION_STATUS_ZERO_ACTIVE_SKIPPED,
    RunManifest,
    validate_main_run,
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


def _trainer(tmp_path, *, max_skips=50, max_iters=None, use_new_limit=False):
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
            "max_consecutive_nonprogress_iterations": (
                max_skips if use_new_limit else None
            ),
            "max_consecutive_skipped_updates": (
                None if use_new_limit else max_skips
            ),
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
    obj.consecutive_nonprogress_iterations = 0
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
            (ITERATION_STATUS_RUNNING, False),
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


class TestManifestResumeLifecycle:
    """Remaining canonical-run resume manifest requirements."""

    def test_resume_loads_existing_manifest_and_preserves_history(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.replay_buffer = trainer._new_replay_buffer()
        trainer.global_steps = 3
        trainer.rollout_iteration = 4
        trainer.num_optimizer_steps_total = 99
        ckpt = tmp_path / "global_step_3"
        ckpt.mkdir()
        trainer._restored_checkpoint_dir = str(ckpt)
        manifest = trainer._build_run_manifest()
        manifest.global_step = 3
        manifest.rollout_iteration = 4
        manifest.group_integrity_failures = 5
        manifest.segment_count_failures = 2
        manifest.replay_batch_failures = 1
        manifest.parent_split_count = 7
        manifest.tree_split_count = 8
        manifest.num_optimizer_steps_total = 99
        manifest.stored_old_log_probs_used = True
        manifest.segment_count_invariants_passed = False
        manifest.optimizer_step_accounting_valid = False
        manifest.unique_tree_ids_verified = True
        manifest.save(ckpt / "vdra_run_manifest.json")

        loaded = trainer._load_or_build_run_manifest()
        assert loaded.group_integrity_failures == 5
        assert loaded.segment_count_failures == 2
        assert loaded.replay_batch_failures == 1
        assert loaded.parent_split_count == 7
        assert loaded.tree_split_count == 8
        assert loaded.num_optimizer_steps_total == 99
        assert loaded.stored_old_log_probs_used is True
        assert loaded.segment_count_invariants_passed is False
        assert loaded.optimizer_step_accounting_valid is False
        assert loaded.unique_tree_ids_verified is True

    def test_resume_manifest_config_mismatch_fails_fast(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.replay_buffer = trainer._new_replay_buffer()
        trainer.global_steps = 3
        ckpt = tmp_path / "global_step_3"
        ckpt.mkdir()
        trainer._restored_checkpoint_dir = str(ckpt)
        manifest = trainer._build_run_manifest()
        manifest.global_step = 3
        manifest.num_optimizer_steps_total = 0
        manifest.probability_mask_threshold = 0.75
        manifest.save(ckpt / "vdra_run_manifest.json")

        with pytest.raises(ValueError, match="probability_mask_threshold"):
            trainer._load_or_build_run_manifest()

    def test_no_checkpoint_manifest_builds_fresh(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.global_steps = 3
        manifest = trainer._load_or_build_run_manifest()
        assert isinstance(manifest, RunManifest)
        assert manifest.group_integrity_failures == 0

    def test_resume_loads_checkpoint_manifest_not_newer_root(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.replay_buffer = trainer._new_replay_buffer()
        trainer.global_steps = 3
        trainer.rollout_iteration = 4
        trainer.num_optimizer_steps_total = 8
        ckpt = tmp_path / "global_step_3"
        ckpt.mkdir()
        trainer._restored_checkpoint_dir = str(ckpt)

        checkpoint_manifest = trainer._build_run_manifest()
        checkpoint_manifest.global_step = 3
        checkpoint_manifest.rollout_iteration = 4
        checkpoint_manifest.num_optimizer_steps_total = 8
        checkpoint_manifest.group_integrity_failures = 1
        checkpoint_manifest.save(ckpt / "vdra_run_manifest.json")

        root_manifest = trainer._build_run_manifest()
        root_manifest.global_step = 999
        root_manifest.rollout_iteration = 999
        root_manifest.num_optimizer_steps_total = 999
        root_manifest.group_integrity_failures = 77
        root_manifest.save(trainer._manifest_path())

        loaded = trainer._load_or_build_run_manifest()
        assert loaded.global_step == 3
        assert loaded.rollout_iteration == 4
        assert loaded.group_integrity_failures == 1

    def test_resume_manifest_counter_mismatch_fails_fast(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.replay_buffer = trainer._new_replay_buffer()
        trainer.global_steps = 3
        trainer.rollout_iteration = 4
        trainer.num_optimizer_steps_total = 8
        ckpt = tmp_path / "global_step_3"
        ckpt.mkdir()
        trainer._restored_checkpoint_dir = str(ckpt)
        manifest = trainer._build_run_manifest()
        manifest.global_step = 3
        manifest.rollout_iteration = 3
        manifest.num_optimizer_steps_total = 8
        manifest.save(ckpt / "vdra_run_manifest.json")
        with pytest.raises(ValueError, match="counter mismatch"):
            trainer._load_or_build_run_manifest()

    def test_canonical_resume_without_checkpoint_manifest_fails(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.replay_buffer = trainer._new_replay_buffer()
        trainer._restored_checkpoint_dir = str(tmp_path / "global_step_0")
        with pytest.raises(FileNotFoundError, match="checkpoint-scoped"):
            trainer._load_or_build_run_manifest()

    def test_noncanonical_missing_manifest_marks_provenance_missing(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.config.actor_rollout_ref.actor.policy_loss[
            "policy_aggregation"
        ] = "tree_balanced_segment_mean"
        trainer.replay_buffer = trainer._new_replay_buffer()
        trainer._restored_checkpoint_dir = str(tmp_path / "global_step_0")
        manifest = trainer._load_or_build_run_manifest()
        assert manifest.manifest_resume_provenance_missing is True
        assert validate_main_run(manifest) is not None


class TestLivelockGuards:
    """Required tests 3-5."""

    def _metrics(self):
        return {
            "vdra/all_zero_advantage_logical_batches": 2.0,
            "vdra/zero_active_token_logical_batches": 0.0,
        }


    def test_deprecated_skip_limit_alias_maps_to_nonprogress_limit(self, tmp_path):
        trainer = _trainer(tmp_path, max_skips=11)
        assert trainer._resolve_max_nonprogress_iterations() == 11

    def test_new_nonprogress_limit_is_primary(self, tmp_path):
        trainer = _trainer(tmp_path, max_skips=13, use_new_limit=True)
        assert trainer._resolve_max_nonprogress_iterations() == 13

    def test_conflicting_old_and_new_guard_fields_fail_fast(self, tmp_path):
        trainer = _trainer(tmp_path, max_skips=13, use_new_limit=True)
        trainer.config.gear_tree["max_consecutive_skipped_updates"] = 12
        with pytest.raises(ValueError, match="deprecated alias"):
            trainer._resolve_max_nonprogress_iterations()

    @pytest.mark.parametrize(
        "status",
        [
            ITERATION_STATUS_NO_SAMPLE,
            ITERATION_STATUS_POSTPONED,
            ITERATION_STATUS_ALL_ZERO_SKIPPED,
            ITERATION_STATUS_ZERO_ACTIVE_SKIPPED,
            ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED,
        ],
    )
    def test_nonprogress_statuses_increment_the_counter(self, tmp_path, status):
        trainer = _trainer(tmp_path)
        trainer._set_nonprogress_counter(4)
        trainer._set_iteration_status(status)
        trainer._increment_nonprogress_counter()
        assert trainer.consecutive_nonprogress_iterations == 5
        assert trainer.consecutive_skipped_updates == 5

    def test_limit_not_reached_is_a_no_op(self, tmp_path):
        trainer = _trainer(tmp_path, max_skips=3)
        trainer._set_nonprogress_counter(2)
        trainer._enforce_livelock_guards(self._metrics())  # must not raise

    def test_repeated_skips_trip_the_limit(self, tmp_path):
        """Required test 3."""
        trainer = _trainer(tmp_path, max_skips=3)
        trainer._set_nonprogress_counter(3)
        trainer.global_steps = 0
        trainer.rollout_iteration = 7
        with pytest.raises(RuntimeError) as excinfo:
            trainer._enforce_livelock_guards(self._metrics())
        msg = str(excinfo.value)
        # The error must carry the full diagnostic context.
        assert "global_step=0" in msg
        assert "rollout_iteration=7" in msg
        assert "3 consecutive nonprogress" in msg
        assert "last_iteration_status=" in msg
        assert "buffer_size=" in msg
        assert "postponed_updates=" in msg
        assert "max_consecutive_nonprogress_iterations=3" in msg
        assert "all_zero_advantage_logical_batches=2.0" in msg
        assert "zero_active_token_logical_batches=0.0" in msg

    def test_manifest_is_persisted_before_aborting(self, tmp_path):
        trainer = _trainer(tmp_path, max_skips=1)
        trainer._set_nonprogress_counter(1)
        with pytest.raises(RuntimeError):
            trainer._enforce_livelock_guards(self._metrics())
        assert (tmp_path / "vdra_run_manifest.json").exists()

    @pytest.mark.parametrize("limit", [0, -1, None])
    def test_guard_can_be_disabled(self, tmp_path, limit):
        trainer = _trainer(tmp_path, max_skips=limit)
        trainer._set_nonprogress_counter(10_000)
        trainer._enforce_livelock_guards(self._metrics())  # must not raise

    def test_successful_update_resets_the_counter(self, tmp_path):
        """Required test 4 — driven through the REAL finalize helper."""
        trainer = _trainer(tmp_path)
        trainer._set_nonprogress_counter(5)
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
        assert trainer.consecutive_nonprogress_iterations == 0
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
        trainer._set_nonprogress_counter(2)
        metrics: dict = {}
        trainer._log_livelock_counters(metrics)
        assert metrics["training/consecutive_nonprogress_iterations"] == 2.0
        assert metrics["training/max_consecutive_nonprogress_iterations"] == 7.0
        assert metrics["training/consecutive_skipped_updates"] == 2.0
        assert metrics["training/max_consecutive_skipped_updates"] == 7.0
        assert metrics["training/max_rollout_iterations"] == 9.0


class TestSkipPathIoRobustness:
    """Required tests 1, 2 and 11 — driven through the REAL fit-loop skip
    branch by source inspection plus direct behavioural checks, because the
    branch is only reachable inside ``fit()``."""

    def _skip_branch_source(self) -> str:
        src = inspect.getsource(RayGearTreeTrainer.fit)
        start = src.index("if edge_batch is None:")
        end = src.index("actor_updated = True", start)
        return src[start:end]

    def _nonprogress_helper_source(self) -> str:
        return inspect.getsource(RayGearTreeTrainer._record_nonprogress_iteration)

    def test_manifest_write_failure_is_caught(self):
        """Required test 1: the reservation is already committed, so a
        manifest I/O failure must not crash training."""
        helper = self._nonprogress_helper_source()
        assert "self._save_manifest_best_effort(metrics)" in helper
        assert "vdra/manifest_save_failed" in inspect.getsource(
            RayGearTreeTrainer._save_manifest_best_effort
        )

    def test_timing_write_failure_is_caught(self):
        """Required test 2."""
        helper = self._nonprogress_helper_source()
        assert "self._append_timing_row(timing_path, row, metrics)" in helper
        assert "vdra/timing_write_failed" in inspect.getsource(
            RayGearTreeTrainer._append_timing_row
        )

    def test_skip_timing_carries_cumulative_and_wall_clock(self):
        """Required test 11."""
        branch = self._skip_branch_source()
        helper = self._nonprogress_helper_source()
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
            assert key in branch or key in helper, key
        # Generation time is folded into the running total.
        assert "updated_cum_train" in helper


    def test_fit_failure_wrapper_sets_and_persists_failed_before_actor(self):
        src = inspect.getsource(RayGearTreeTrainer.fit)
        assert "self._set_iteration_status(ITERATION_STATUS_RUNNING)" in src
        assert "== ITERATION_STATUS_RUNNING" in src
        assert "ITERATION_STATUS_FAILED_BEFORE_ACTOR" in src
        assert "self._save_manifest_best_effort(metrics)" in src

    def test_skip_timing_reports_zero_steps_and_unchanged_global_step(self):
        branch = self._skip_branch_source()
        helper = self._nonprogress_helper_source()
        assert '"training/expected_optimizer_steps": 0.0' in helper
        assert '"training/optimizer_steps_this_iteration": 0.0' in helper
        # The skip branch must never touch global_step.
        assert "self.global_steps +=" not in branch

    def test_livelock_guard_runs_after_state_is_persisted(self):
        helper = self._nonprogress_helper_source()
        guard_at = helper.index("_enforce_livelock_guards")
        manifest_at = helper.index("self._save_manifest_best_effort(metrics)")
        assert manifest_at < guard_at, (
            "the guard must abort only AFTER the manifest was persisted"
        )


    def test_normal_timing_write_failure_does_not_propagate(self, tmp_path, monkeypatch):
        trainer = _trainer(tmp_path)

        def _boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("builtins.open", _boom)
        metrics: dict = {}
        trainer._append_timing_row(str(tmp_path / "training_timing.jsonl"), {"x": 1}, metrics)
        assert metrics["vdra/timing_write_failed"] == 1.0

    def test_manifest_save_failure_does_not_propagate(self, tmp_path, monkeypatch):
        """Behavioural check of the same contract on the helper itself."""
        trainer = _trainer(tmp_path)

        def _boom(_manifest):
            raise OSError("disk full")

        monkeypatch.setattr(trainer, "_save_manifest", _boom)
        metrics: dict = {}
        # The guard's own best-effort save must swallow the failure and still
        # raise the intended RuntimeError (not the OSError).
        trainer._set_nonprogress_counter(50)
        with pytest.raises(RuntimeError, match="consecutive nonprogress"):
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


class TestLiveStateResume:
    def _prep_resume(self, tmp_path, monkeypatch, *, live_global_step=0, live_rollout=50):
        from recipe.gear_tree.trainer_state import (
            GearTreeLiveState,
            GearTreeTrainerState,
            save_live_state,
            save_trainer_state,
        )
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        trainer = _trainer(tmp_path)
        trainer.config.trainer.resume_mode = "resume_path"
        trainer.config.trainer.resume_from_path = str(tmp_path / "global_step_0")
        trainer.config.trainer.default_hdfs_dir = None
        trainer.global_steps = 0
        trainer.replay_buffer = trainer._new_replay_buffer()
        ckpt = tmp_path / "global_step_0"
        ckpt.mkdir()
        save_trainer_state(
            ckpt,
            GearTreeTrainerState(
                global_step=0,
                rollout_iteration=0,
                num_optimizer_steps_total=0,
                consecutive_nonprogress_iterations=0,
            ),
        )
        manifest = trainer._build_run_manifest()
        manifest.global_step = 0
        manifest.rollout_iteration = 0
        manifest.num_optimizer_steps_total = 0
        manifest.save(ckpt / "vdra_run_manifest.json")
        save_live_state(
            tmp_path,
            GearTreeLiveState(
                global_step=live_global_step,
                rollout_iteration=live_rollout,
                num_optimizer_steps_total=0,
                successful_actor_updates=0,
                postponed_updates=7,
                failed_updates=0,
                skipped_zero_gradient_updates=0,
                consecutive_nonprogress_iterations=49,
                last_iteration_status=ITERATION_STATUS_POSTPONED,
            ),
        )
        monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", lambda self: 0)
        return trainer

    def test_global_step_zero_live_state_preserves_nonprogress_after_restart(self, tmp_path, monkeypatch):
        trainer = self._prep_resume(tmp_path, monkeypatch)
        trainer._load_checkpoint()
        trainer.run_manifest = trainer._load_or_build_run_manifest()
        trainer._merge_pending_live_state()
        assert trainer.global_steps == 0
        assert trainer.rollout_iteration == 50
        assert trainer.consecutive_nonprogress_iterations == 49
        assert trainer.postponed_updates == 7
        assert trainer.run_manifest.last_iteration_status == ITERATION_STATUS_POSTPONED

    def test_live_state_with_different_global_step_is_rejected(self, tmp_path, monkeypatch):
        trainer = self._prep_resume(tmp_path, monkeypatch, live_global_step=1)
        with pytest.raises(ValueError, match="gear_tree_live_state.json global_step"):
            trainer._load_checkpoint()

    def test_stale_live_state_rollout_is_ignored(self, tmp_path, monkeypatch):
        trainer = self._prep_resume(tmp_path, monkeypatch, live_rollout=-1)
        trainer._load_checkpoint()
        trainer.run_manifest = trainer._load_or_build_run_manifest()
        trainer._merge_pending_live_state()
        assert trainer.rollout_iteration == 0
        assert trainer.consecutive_nonprogress_iterations == 0


class TestNonprogressHelper:
    def test_no_sample_writes_timing_manifest_and_live_state(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.replay_buffer = trainer._new_replay_buffer()
        trainer._expected_optimizer_steps = None
        metrics = {}
        timing_path = str(tmp_path / "training_timing.jsonl")
        out = trainer._record_nonprogress_iteration(
            status=ITERATION_STATUS_NO_SAMPLE,
            timing_path=timing_path,
            t_gen=1.25,
            sample_stats={"buffer/selected_edges": 0},
            metrics=metrics,
            loop_start=0.0,
            cumulative_train_seconds=2.0,
        )
        assert out == pytest.approx(3.25)
        row = json.loads((tmp_path / "training_timing.jsonl").read_text().splitlines()[-1])
        assert row["last_iteration_status"] == ITERATION_STATUS_NO_SAMPLE
        assert row["actor_update_skipped"] is False
        assert row["timing/update_seconds"] == 0.0
        assert row["training/expected_optimizer_steps"] == 0.0
        assert (tmp_path / "vdra_run_manifest.json").exists()
        assert (tmp_path / "gear_tree_live_state.json").exists()

    def test_zero_signal_status_sets_actor_update_skipped_only_for_zero_skip(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.replay_buffer = trainer._new_replay_buffer()
        metrics = {}
        trainer.skipped_zero_gradient_updates = 1
        trainer._record_nonprogress_iteration(
            status=ITERATION_STATUS_ALL_ZERO_SKIPPED,
            timing_path=str(tmp_path / "training_timing.jsonl"),
            t_gen=0.5,
            sample_stats={"buffer/selected_edges": 4},
            metrics=metrics,
            loop_start=0.0,
            cumulative_train_seconds=0.0,
        )
        assert trainer.run_manifest.actor_update_skipped is True
        row = json.loads((tmp_path / "training_timing.jsonl").read_text().splitlines()[-1])
        assert row["actor_update_skipped"] is True
