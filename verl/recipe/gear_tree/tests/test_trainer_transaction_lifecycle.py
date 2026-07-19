"""PLAN.md M2: reserve -> validate -> tensorize -> actor RPC transaction.

Every pre-actor failure must roll the reservation back and leave the outer
counters untouched; only an actor RPC failure increments ``failed_updates``.
Commit and outer-counter mutation stay in ``fit()`` after a successful RPC.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

from recipe.gear_tree.gear_ray_trainer import RayGearTreeTrainer
from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer
from recipe.gear_tree.run_manifest import RunManifest


class _Cfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1


def _edge(edge_id="e", question_id="q"):
    return {
        "edge_id": edge_id,
        "question_id": question_id,
        "query_token_ids": [5, 6],
        "response_token_ids": [7, 8],
        "actor_shifted_log_probs": [-0.1, -0.2],
        "advantage": 1.0,
        "value": 0.5,
        "reward": 1.0,
        "depth": 0,
        "leaf": False,
        "pruned": False,
        "tree_update_mode": "spo",
    }


def _trainer_with_reservation(n_edges=2):
    """Real trainer methods + real replay buffer, no Ray."""
    trainer = object.__new__(RayGearTreeTrainer)
    trainer.tokenizer = _Tokenizer()
    trainer.config = _Cfg(
        data=_Cfg(max_prompt_length=4, max_response_length=3),
        trainer=_Cfg(balance_batch=False, default_local_dir="/tmp"),
        actor_rollout_ref=_Cfg(
            actor=_Cfg(
                ppo_mini_batch_size=1,
                policy_loss={"loss_mode": "vdra_segment_mean_ppo"},
            )
        ),
        gear_tree={
            "replay_buffer": {
                "target_edges_per_iteration": 512,
                "max_edges_per_question_per_iteration": 32,
                "max_edge_age_iterations": 8,
                "sampling_seed": 0,
            },
            "gear": {"strict_vdra": False},
        },
    )
    trainer.global_steps = 0
    trainer.rollout_iteration = 1
    trainer.successful_actor_updates = 0
    trainer.postponed_updates = 0
    trainer.failed_updates = 0
    trainer.optimizer_steps_this_iteration = 0
    trainer.num_optimizer_steps_total = 0
    trainer.run_manifest = RunManifest()

    buffer = GearTreeReplayBuffer(
        target_edges_per_iteration=512,
        max_edge_age_iterations=8,
        max_edges_per_question_per_iteration=32,
        sampling_seed=0,
    )
    buffer.add(
        [_edge(f"e{i}") for i in range(n_edges)],
        generation_rollout_iteration=1,
        policy_snapshot_id="global_step:0",
    )
    trainer.replay_buffer = buffer
    reservation = buffer.reserve_for_update(current_rollout_iteration=1)
    sampled_edges = [dict(edge) for edge in reservation.edges]
    return trainer, buffer, reservation, sampled_edges


def _assert_rolled_back_and_counters_untouched(trainer, buffer, n_edges):
    assert buffer._reserved == {}
    assert len(buffer) == n_edges
    assert trainer.global_steps == 0
    assert trainer.successful_actor_updates == 0
    assert trainer.postponed_updates == 0
    assert trainer.num_optimizer_steps_total == 0


class TestPreActorFailures:
    def test_replay_validation_failure_rolls_back_without_counters(self):
        trainer, buffer, reservation, sampled = _trainer_with_reservation()
        sampled[0].pop("advantage")
        with pytest.raises(ValueError, match="advantage"):
            trainer._execute_reserved_actor_update(
                buffer, reservation, sampled, {}, manifest_strict=True
            )
        _assert_rolled_back_and_counters_untouched(trainer, buffer, 2)
        assert trainer.failed_updates == 0
        assert trainer.run_manifest.replay_batch_failures == 1
        assert trainer.run_manifest.no_truncation is False

    def test_tensorization_failure_rolls_back_without_counters(self):
        trainer, buffer, reservation, sampled = _trainer_with_reservation()

        def _boom(sampled_edges, metrics):
            raise RuntimeError("tensorization failed")

        trainer._edges_to_update_batch = _boom
        with pytest.raises(RuntimeError, match="tensorization failed"):
            trainer._execute_reserved_actor_update(
                buffer, reservation, sampled, {}, manifest_strict=True
            )
        _assert_rolled_back_and_counters_untouched(trainer, buffer, 2)
        assert trainer.failed_updates == 0
        assert trainer.run_manifest.replay_batch_failures == 0
        assert trainer.run_manifest.no_truncation is False

    def test_overlength_row_is_a_real_tensorization_failure(self):
        trainer, buffer, reservation, sampled = _trainer_with_reservation()
        sampled[0]["query_token_ids"] = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="max_prompt_length"):
            trainer._execute_reserved_actor_update(
                buffer, reservation, sampled, {}, manifest_strict=True
            )
        _assert_rolled_back_and_counters_untouched(trainer, buffer, 2)
        assert trainer.failed_updates == 0

    def test_actor_rpc_failure_rolls_back_and_counts_failed_update(self):
        trainer, buffer, reservation, sampled = _trainer_with_reservation()

        def _rpc_fail(edge_batch):
            raise RuntimeError("actor RPC died")

        trainer.actor_rollout_wg = SimpleNamespace(update_actor=_rpc_fail)
        with pytest.raises(RuntimeError, match="actor RPC died"):
            trainer._execute_reserved_actor_update(
                buffer, reservation, sampled, {}, manifest_strict=True
            )
        assert buffer._reserved == {}
        assert len(buffer) == 2
        assert trainer.global_steps == 0
        assert trainer.successful_actor_updates == 0
        assert trainer.failed_updates == 1
        # Tensorization succeeded before the RPC, so the observed
        # no-truncation event did happen for this batch.
        assert trainer.run_manifest.no_truncation is True
        # Rolled-back edges must be reservable again.
        again = buffer.reserve_for_update(current_rollout_iteration=1)
        assert sorted(again.edge_ids) == ["e0", "e1"]


class TestSuccessPath:
    def test_success_leaves_commit_to_fit(self):
        trainer, buffer, reservation, sampled = _trainer_with_reservation()
        seen = {}

        def _rpc_ok(edge_batch):
            seen["rows"] = len(edge_batch)
            return SimpleNamespace(
                meta_info={"metrics": {"actor/num_optimizer_steps": [2]}}
            )

        trainer.actor_rollout_wg = SimpleNamespace(update_actor=_rpc_ok)
        edge_batch, actor_output, t_update = (
            trainer._execute_reserved_actor_update(
                buffer, reservation, sampled, {}, manifest_strict=True
            )
        )
        assert seen["rows"] == 2
        assert actor_output.meta_info["metrics"][
            "actor/num_optimizer_steps"
        ] == [2]
        assert t_update >= 0.0
        assert trainer.run_manifest.no_truncation is True
        # The helper never commits or mutates outer counters — fit() does.
        assert set(buffer._reserved) == {"e0", "e1"}
        assert trainer.global_steps == 0
        assert trainer.successful_actor_updates == 0
        assert trainer.failed_updates == 0
        removed = buffer.commit(reservation)
        assert sorted(removed) == ["e0", "e1"]
        assert len(buffer) == 0


class TestFitWiring:
    def test_fit_routes_update_through_transaction_helper(self):
        source = inspect.getsource(RayGearTreeTrainer.fit)
        assert "_execute_reserved_actor_update(" in source
        assert "self.actor_rollout_wg.update_actor(" not in source

    def test_helper_does_not_commit_or_mutate_outer_counters(self):
        source = inspect.getsource(
            RayGearTreeTrainer._execute_reserved_actor_update
        )
        assert ".commit(" not in source
        assert "self.global_steps +=" not in source
        assert "self.successful_actor_updates +=" not in source
        assert "self.num_optimizer_steps_total +=" not in source
