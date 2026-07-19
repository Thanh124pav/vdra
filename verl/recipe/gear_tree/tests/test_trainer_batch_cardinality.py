"""PLAN.md P0.D: exact sample size and optimizer-batch divisibility.

Canonical mode allows only selected counts divisible by
``ppo_mini_batch_size`` (128/256/384/512 with the shipped config), postpones
everything else, and treats a count above the target as a sampler bug. The
expected-step formula is valid only after divisibility is enforced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe.gear_tree.replay_buffer import (
    GearTreeReplayBuffer,
    expected_optimizer_steps,
    reserve_replay_edges,
    should_postpone_sampled_update,
)


def _postpone(n: int, *, policy: str = "postpone_until_divisible") -> bool:
    return should_postpone_sampled_update(
        selected_count=n,
        target_edges_per_iteration=512,
        ppo_mini_batch_size=128,
        underfilled_update_policy=policy,
    )


class TestPostponeDecision:
    @pytest.mark.parametrize("n", [128, 256, 384, 512])
    def test_divisible_counts_run(self, n):
        assert _postpone(n) is False

    @pytest.mark.parametrize("n", [4, 130, 258, 511])
    def test_non_divisible_counts_postpone(self, n):
        assert _postpone(n) is True

    def test_empty_batch_never_postpones(self):
        assert _postpone(0) is False

    @pytest.mark.parametrize("n", [513, 516, 600, 769])
    def test_over_target_is_a_sampler_bug(self, n):
        with pytest.raises(AssertionError, match="exceeding"):
            _postpone(n)

    def test_use_available_ablation_runs_non_divisible_counts(self):
        assert _postpone(258, policy="use_available") is False

    def test_use_available_still_rejects_over_target(self):
        with pytest.raises(AssertionError):
            _postpone(516, policy="use_available")

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="underfilled_update_policy"):
            _postpone(128, policy="run_anyway")


class TestExpectedOptimizerSteps:
    @pytest.mark.parametrize(
        "n,steps", [(512, 4), (384, 3), (256, 2), (128, 1)]
    )
    def test_divisible_counts(self, n, steps):
        assert (
            expected_optimizer_steps(
                selected_count=n, ppo_mini_batch_size=128, ppo_epochs=1
            )
            == steps
        )

    def test_ppo_epochs_multiplier(self):
        assert (
            expected_optimizer_steps(
                selected_count=256, ppo_mini_batch_size=128, ppo_epochs=3
            )
            == 6
        )

    @pytest.mark.parametrize("n", [4, 130, 258, 516])
    def test_non_divisible_counts_raise(self, n):
        with pytest.raises(ValueError, match="divisible"):
            expected_optimizer_steps(
                selected_count=n, ppo_mini_batch_size=128
            )


class TestEndToEndCardinality:
    def _edge(self, edge_id: str, question_id: str) -> dict:
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

    def test_516_candidates_select_512_run_4_steps(self):
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=512,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=1000,
            sampling_seed=0,
        )
        edges = []
        for q in range(4):
            edges.extend(
                self._edge(f"q{q}-e{i}", f"q{q}") for i in range(129)
            )
        assert len(edges) == 516
        buf.add(edges, generation_rollout_iteration=1, policy_snapshot_id="s0")
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        n = len(reservation.edges)
        assert n == 512
        assert not should_postpone_sampled_update(
            selected_count=n,
            target_edges_per_iteration=512,
            ppo_mini_batch_size=128,
        )
        assert (
            expected_optimizer_steps(
                selected_count=n, ppo_mini_batch_size=128, ppo_epochs=1
            )
            == 4
        )

    def test_130_candidates_are_postponed(self):
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=512,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=1000,
            sampling_seed=0,
        )
        buf.add(
            [self._edge(f"e{i}", "q0") for i in range(130)],
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert len(reservation.edges) == 130
        assert should_postpone_sampled_update(
            selected_count=len(reservation.edges),
            target_edges_per_iteration=512,
            ppo_mini_batch_size=128,
        )
        # Canonical flow: postponed reservations roll back unchanged.
        buf.rollback(reservation)
        assert len(buf) == 130
        assert len(buf._reserved) == 0


class TestTrainerWiring:
    def test_trainer_delegates_to_production_helpers(self):
        """The trainer must use the extracted production helpers, not a
        re-implemented (and previously buggy) inline condition."""
        source = (
            Path(__file__).resolve().parents[1] / "gear_ray_trainer.py"
        ).read_text()
        assert "should_postpone_sampled_update(" in source
        assert "expected_optimizer_steps(" in source
        # The old buggy guard allowed oversized non-divisible batches:
        assert "< target and" not in source
