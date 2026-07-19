"""PLAN.md P0.E: trainer counters survive checkpoint/resume exactly.

* ``gear_tree_trainer_state.json`` round-trips all counters.
* Restored replay ages are non-negative and expire on schedule when
  ``rollout_iteration`` is restored alongside the buffer.
* A legacy checkpoint without the state file loads as ``None`` so the
  trainer can reset replay instead of silently creating negative ages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer
from recipe.gear_tree.trainer_state import (
    TRAINER_STATE_FILENAME,
    GearTreeTrainerState,
    load_trainer_state,
    save_trainer_state,
    trainer_state_path,
)


def _edge(edge_id: str, question_id: str = "q0") -> dict:
    return {
        "edge_id": edge_id,
        "question_id": question_id,
        "query_token_ids": [1],
        "response_token_ids": [2, 3],
        "actor_shifted_log_probs": [-0.1, -0.2],
        "advantage": 1.0,
        "value": 0.5,
        "reward": 0.5,
    }


class TestStateRoundTrip:
    def test_exact_counter_equality(self, tmp_path):
        state = GearTreeTrainerState(
            global_step=400,
            rollout_iteration=100,
            num_optimizer_steps_total=400,
            successful_actor_updates=97,
            postponed_updates=2,
            failed_updates=1,
        )
        save_trainer_state(tmp_path, state)
        loaded = load_trainer_state(tmp_path)
        assert loaded == state
        assert loaded.global_step == 400
        assert loaded.rollout_iteration == 100
        assert loaded.num_optimizer_steps_total == 400

    def test_m1_counter_units_survive_resume(self, tmp_path):
        """PLAN.md M1 completion table: five successful outer updates that
        each report four internal PPO optimizer batches persist and restore
        as global_step=5 / num_optimizer_steps_total=20 — the units must not
        change across a save/load boundary.
        """
        state = GearTreeTrainerState(
            global_step=5,
            rollout_iteration=5,
            num_optimizer_steps_total=20,
            successful_actor_updates=5,
        )
        save_trainer_state(tmp_path, state)
        loaded = load_trainer_state(tmp_path)
        assert loaded.global_step == 5
        assert loaded.rollout_iteration == 5
        assert loaded.num_optimizer_steps_total == 20
        assert loaded.successful_actor_updates == 5

    def test_state_file_name_and_location(self, tmp_path):
        save_trainer_state(tmp_path, GearTreeTrainerState(global_step=8))
        assert (tmp_path / TRAINER_STATE_FILENAME).exists()
        assert trainer_state_path(tmp_path).name == "gear_tree_trainer_state.json"

    def test_legacy_checkpoint_without_state_returns_none(self, tmp_path):
        assert load_trainer_state(tmp_path) is None

    def test_unknown_keys_are_ignored(self, tmp_path):
        payload = GearTreeTrainerState(global_step=12).to_dict()
        payload["future_field"] = 7
        trainer_state_path(tmp_path).write_text(json.dumps(payload))
        loaded = load_trainer_state(tmp_path)
        assert loaded is not None
        assert loaded.global_step == 12


class TestRestoredReplayAges:
    def _buffer(self) -> GearTreeReplayBuffer:
        return GearTreeReplayBuffer(
            target_edges_per_iteration=512,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=1000,
            sampling_seed=0,
        )

    def test_restored_ages_are_non_negative_and_expire_on_schedule(self, tmp_path):
        buf = self._buffer()
        # A long-running trainer: edges generated at rollout iterations
        # 93..100, checkpointed at rollout_iteration=100 / global_step=400.
        for it in range(93, 101):
            buf.add(
                [_edge(f"it{it}-e{i}") for i in range(2)],
                generation_rollout_iteration=it,
                policy_snapshot_id="s0",
            )
        buf.save(tmp_path)
        save_trainer_state(
            tmp_path,
            GearTreeTrainerState(
                global_step=400,
                rollout_iteration=100,
                num_optimizer_steps_total=400,
            ),
        )

        restored = GearTreeReplayBuffer.load(tmp_path)
        state = load_trainer_state(tmp_path)
        cri = state.rollout_iteration
        assert cri == 100

        ages = [
            cri - int(e["generation_rollout_iteration"])
            for e in restored.edges()
        ]
        assert ages and all(a >= 0 for a in ages), ages

        # Nothing is old enough to expire at the restored iteration...
        expired = restored.expire(current_rollout_iteration=cri)
        assert expired == []
        # ...and the oldest edges (generation 93, age 8) expire exactly one
        # iteration later.
        expired = restored.expire(current_rollout_iteration=cri + 1)
        assert sorted(expired) == [f"it93-e{i}" for i in range(2)]

    def test_resume_without_state_would_go_negative(self, tmp_path):
        """Documents the failure mode option A protects against: a reset
        rollout_iteration with restored high-generation edges yields
        negative ages and edges that can never expire."""
        buf = self._buffer()
        buf.add(
            [_edge("e0")], generation_rollout_iteration=98,
            policy_snapshot_id="s0",
        )
        buf.save(tmp_path)
        restored = GearTreeReplayBuffer.load(tmp_path)
        reset_rollout_iteration = 0
        age = reset_rollout_iteration - 98
        assert age < 0
        assert restored.expire(current_rollout_iteration=0) == []


class TestTrainerWiring:
    def test_trainer_saves_and_restores_counter_state(self):
        source = (
            Path(__file__).resolve().parents[1] / "gear_ray_trainer.py"
        ).read_text()
        # Save side: state written into the checkpoint dir after the base
        # checkpoint.
        assert "save_trainer_state(" in source
        # Load side: rollout_iteration restored from the state file, folder
        # consistency asserted, legacy checkpoints flagged.
        assert "load_trainer_state(" in source
        assert "self.rollout_iteration = int(state.rollout_iteration)" in source
        assert "_legacy_checkpoint_without_state" in source
        # Legacy path resets replay instead of restoring it.
        assert "buffer/legacy_checkpoint_reset" in source
