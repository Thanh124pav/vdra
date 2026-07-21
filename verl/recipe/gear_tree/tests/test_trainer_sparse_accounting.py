"""Trainer-level integration for the sparse accounting + metrics contract.

PLAN.md §2/§3/§5 corrections. These drive the REAL trainer methods
(``_edges_to_update_batch``, ``_resolve_expected_optimizer_steps``,
``_record_iteration_on_manifest``, ``_replay_config``) rather than mirrors,
because the defects they pin were wiring defects between those methods:

* a skipped logical batch must not make a valid mixed update look like an
  optimizer-step accounting mismatch (the stale selected-slot formula);
* the manifest and the timing JSON must report the SAME expected count;
* dummy padding rows must be excluded from the reported row/token metrics;
* ``reset_replay_on_objective_mismatch`` must reach replay restore.
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

from recipe.gear_tree.gear_ray_trainer import RayGearTreeTrainer  # noqa: E402
from recipe.gear_tree.run_manifest import RunManifest  # noqa: E402

try:
    from recipe.gear_tree.tests import _tiny_actor
except ImportError:
    import _tiny_actor

MINI = 4
LP_ACTIVE = -0.2
LP_INACTIVE = -0.02


class _Cfg(dict):
    """Attribute-accessible config stub mirroring OmegaConf access."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - surfaced as AttributeError
            raise AttributeError(item) from exc

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _edge(i: int, adv: float, lps=None, tree: str = "t0") -> dict:
    lps = lps if lps is not None else [LP_ACTIVE, LP_ACTIVE]
    return {
        "edge_id": f"{tree}/e{i}",
        "tree_id": tree,
        "parent_group_id": f"{tree}/pg",
        "child_segment_id": f"{tree}/e{i}",
        "question_id": f"q{i % 3}",
        "allocated_k": 4,
        "sample_multiplicity": 1,
        "tree_total_segment_count": 4,
        "queue_flush_id": "0",
        "queue_released_segment_count": 4,
        "query_token_ids": [1, 2],
        "response_token_ids": [3 + (i % 5), 4],
        "actor_shifted_log_probs": list(lps),
        "advantage": adv,
        "value": 0.4,
        "reward": 1.0,
        "advantage_is_zero": adv == 0.0,
        "response_token_count": 2,
        "prob_mask_token_count": sum(1 for lp in lps if lp < -0.10536),  # < 0.9
        "probability_mask_threshold": 0.9,
        "generation_rollout_iteration": 0,
    }


def _slot(i: int, tree: str = "t0") -> dict:
    return {
        "edge_id": f"{tree}/z{i}",
        "tree_id": tree,
        "parent_group_id": f"{tree}/pg",
        "child_segment_id": f"{tree}/z{i}",
        "question_id": f"q{i % 3}",
        "allocated_k": 4,
        "sample_multiplicity": 1,
        "advantage": 0.0,
        "advantage_is_zero": True,
        "trainable_edge_id": None,
        "response_token_count": 2,
        "prob_mask_token_count": 2,
        "probability_mask_threshold": 0.9,
        "generation_rollout_iteration": 0,
    }


def _trainer(*, use_prob_mask=False, reset_on_mismatch=False, ppo_epochs=1):
    obj = object.__new__(RayGearTreeTrainer)
    obj.tokenizer = _tiny_actor.Tok()
    obj.config = _Cfg(
        data=_Cfg(max_prompt_length=6, max_response_length=4),
        trainer=_Cfg(
            balance_batch=False,
            default_local_dir="/tmp",
            nnodes=1,
            n_gpus_per_node=1,
        ),
        actor_rollout_ref=_Cfg(
            actor=_Cfg(
                ppo_mini_batch_size=MINI,
                ppo_epochs=ppo_epochs,
                ulysses_sequence_parallel_size=1,
                policy_loss={
                    "loss_mode": "vdra_segment_mean_ppo",
                    "policy_aggregation": "segment_mean",
                    "use_prob_mask": use_prob_mask,
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
                "reset_replay_on_objective_mismatch": reset_on_mismatch,
            },
            "gear": {"strict_vdra": False},
        },
    )
    obj.run_manifest = RunManifest()
    obj.rollout_iteration = 1
    obj.global_steps = 0
    obj.num_optimizer_steps_total = 0
    obj.successful_actor_updates = 0
    obj.optimizer_steps_this_iteration = 0
    obj.skipped_zero_gradient_updates = 0
    obj._resolved_max_edge_prompt_length = lambda: 6
    return obj


class TestExpectedStepsIsAuthoritative:
    """Required tests 4-6: the trainer's expectation counts only TRAINABLE
    logical batches, and the stale selected-slot formula never overrides it."""

    def _mixed_all_zero_slots(self):
        # batch 0 trainable | batch 1 all-zero | batch 2 trainable
        return (
            [_edge(0, 0.5), _edge(1, -0.5), _slot(0), _slot(1)]
            + [_slot(2 + i) for i in range(4)]
            + [_edge(2, 0.3), _edge(3, -0.3), _slot(6), _slot(7)]
        )

    def test_mixed_all_zero_batch_expects_only_trainable_steps(self):
        trainer = _trainer()
        metrics: dict = {}
        batch = trainer._edges_to_update_batch(self._mixed_all_zero_slots(), metrics)
        assert batch is not None
        # 3 logical batches, 1 skipped -> 2 trainable.
        assert metrics["vdra/trainable_logical_batches"] == 2.0
        assert metrics["vdra/all_zero_advantage_logical_batches"] == 1.0
        assert trainer._expected_optimizer_steps == 2
        # The stale formula would have said 12 slots / 4 = 3.
        assert trainer._resolve_expected_optimizer_steps(12) == 2

    def test_manifest_accounting_is_valid_for_a_mixed_update(self):
        """The load-bearing regression: 2 actual steps against 3 logical
        batches must NOT be flagged as an accounting mismatch."""
        trainer = _trainer()
        metrics: dict = {}
        trainer._edges_to_update_batch(self._mixed_all_zero_slots(), metrics)
        trainer._record_iteration_on_manifest(
            selected_edges=12, sample_stats={}, actual_optimizer_steps=2
        )
        assert trainer.run_manifest.expected_optimizer_steps_last_iteration == 2
        assert trainer.run_manifest.optimizer_step_accounting_valid is True

    def test_zero_active_token_batch_also_reduces_the_expectation(self):
        """Required test 5: the zero_active_tokens skip reason counts too."""
        trainer = _trainer(use_prob_mask=True)
        inactive = [
            _edge(2, 0.3, lps=[LP_INACTIVE, LP_INACTIVE]),
            _edge(3, -0.3, lps=[LP_INACTIVE, LP_INACTIVE]),
        ]
        slots = (
            [_edge(0, 0.5), _edge(1, -0.5), _slot(0), _slot(1)]
            + inactive
            + [_slot(2), _slot(3)]
        )
        metrics: dict = {}
        batch = trainer._edges_to_update_batch(slots, metrics)
        assert batch is not None
        assert metrics["vdra/zero_active_token_logical_batches"] == 1.0
        assert metrics["vdra/all_zero_advantage_logical_batches"] == 0.0
        assert trainer._expected_optimizer_steps == 1
        trainer._record_iteration_on_manifest(
            selected_edges=8, sample_stats={}, actual_optimizer_steps=1
        )
        assert trainer.run_manifest.optimizer_step_accounting_valid is True

    def test_ppo_epochs_multiplies_the_expectation(self):
        trainer = _trainer(ppo_epochs=3)
        metrics: dict = {}
        trainer._edges_to_update_batch(self._mixed_all_zero_slots(), metrics)
        assert trainer._expected_optimizer_steps == 2 * 3

    def test_legacy_fallback_never_overrides_the_canonical_value(self):
        trainer = _trainer()
        metrics: dict = {}
        trainer._edges_to_update_batch(self._mixed_all_zero_slots(), metrics)
        # Even asking with a different slot count returns the canonical value.
        assert trainer._resolve_expected_optimizer_steps(999) == 2

    def test_fallback_applies_only_without_logical_batches(self):
        trainer = _trainer()
        trainer._expected_optimizer_steps = None
        assert trainer._resolve_expected_optimizer_steps(12) == 3  # 12/4 * 1
        # An indivisible count has no well-defined expectation.
        assert trainer._resolve_expected_optimizer_steps(11) is None


class TestDummyFreeMetrics:
    """Required test 11: dummy padding rows count in no reported metric."""

    def test_dummy_rows_are_excluded_from_row_and_token_counts(self):
        trainer = _trainer()
        trainer.config.trainer["n_gpus_per_node"] = 2  # force dp_size = 2
        # 3 trainable rows in one logical batch -> padded to 4 with 1 dummy.
        slots = [_edge(0, 0.5), _edge(1, -0.5), _edge(2, 0.3), _slot(0)]
        metrics: dict = {}
        batch = trainer._edges_to_update_batch(slots, metrics)
        assert batch is not None
        assert metrics["vdra/dummy_rows"] == 1.0

        is_dummy = batch.batch["is_dummy"]
        real_row_mask = ~is_dummy.bool()
        trainable_tensor_rows = int(real_row_mask.sum())
        dummy_rows = int(is_dummy.sum())
        real_response_tokens = int(
            batch.batch["response_mask"][real_row_mask].sum()
        )

        assert dummy_rows == 1
        assert trainable_tensor_rows == 3
        # The raw tensor length includes the dummy row — it is NOT an edge count.
        assert len(batch) == 4
        # Every real row has 2 response tokens; the dummy's token must not count.
        assert real_response_tokens == 6
        assert int(batch.batch["response_mask"].sum()) > real_response_tokens


class TestReplayResetConfigPlumbing:
    """Required test 8: the YAML flag must reach GearTreeReplayBuffer.load."""

    def test_replay_config_exposes_the_flag(self):
        assert (
            _trainer(reset_on_mismatch=True)._replay_config()[
                "reset_replay_on_objective_mismatch"
            ]
            is True
        )
        assert (
            _trainer(reset_on_mismatch=False)._replay_config()[
                "reset_replay_on_objective_mismatch"
            ]
            is False
        )

    @pytest.mark.parametrize("flag", [False, True])
    def test_flag_reaches_replay_load(self, flag, tmp_path, monkeypatch):
        from recipe.gear_tree import gear_ray_trainer as grt
        from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer

        trainer = _trainer(reset_on_mismatch=flag)
        trainer.config.trainer["default_local_dir"] = str(tmp_path)
        # The restore path only runs past the first outer update.
        trainer.global_steps = 1
        # A real saved buffer so the restore path is taken.
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=64,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=32,
            use_prob_mask=False,
            probability_mask_threshold=0.9,
        )
        buf.save(tmp_path / "global_step_1")

        captured: dict = {}
        real_load = GearTreeReplayBuffer.load

        def _spy(cls_dir, **kwargs):
            captured.update(kwargs)
            return real_load(cls_dir, **kwargs)

        monkeypatch.setattr(grt.GearTreeReplayBuffer, "load", staticmethod(_spy))
        trainer._restore_or_init_replay_buffer()

        assert captured["reset_replay_on_objective_mismatch"] is flag
        assert captured["expected_use_prob_mask"] is False
        assert captured["expected_probability_mask_threshold"] == 0.9
