"""PLAN.md P0.A: canonical strict main must use EDGE-level reservation.

Covers the production dispatch function ``reserve_replay_edges`` (the exact
function ``RayGearTreeTrainer.fit`` calls), the per-question auto cap on the
edge path, the hard 512 target cap, and the complete-tree ablation's
no-overshoot packing.
"""

from __future__ import annotations

import yaml
from pathlib import Path

import pytest

from recipe.gear_tree.replay_buffer import (
    GearTreeReplayBuffer,
    VALID_REPLAY_SAMPLING_UNITS,
    reserve_replay_edges,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "gear_tree_trainer.yaml"
)


def _edge(edge_id: str, question_id: str, tree_id: str = "t0") -> dict:
    return {
        "edge_id": edge_id,
        "question_id": question_id,
        "tree_id": tree_id,
        "query_token_ids": [1],
        "response_token_ids": [2, 3],
        "actor_shifted_log_probs": [-0.1, -0.2],
        "advantage": 1.0,
        "value": 0.5,
        "reward": 0.5,
    }


class _SpyBuffer(GearTreeReplayBuffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.edge_reservations = 0
        self.tree_reservations = 0

    def reserve_for_update(self, **kwargs):
        self.edge_reservations += 1
        return super().reserve_for_update(**kwargs)

    def reserve_complete_trees_for_update(self, **kwargs):
        self.tree_reservations += 1
        return super().reserve_complete_trees_for_update(**kwargs)


def _spy_buffer(**overrides) -> _SpyBuffer:
    kwargs = dict(
        target_edges_per_iteration=512,
        max_edge_age_iterations=8,
        max_edges_per_question_per_iteration=1000,
        sampling_seed=0,
    )
    kwargs.update(overrides)
    return _SpyBuffer(**kwargs)


class TestStrictMainDispatch:
    def test_main_config_declares_edge_unit_and_strict_integrity(self):
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        replay = cfg["gear_tree"]["replay_buffer"]
        assert replay["replay_sampling_unit"] == "edge"
        assert cfg["tree_policy"]["strict_group_integrity"] is True

    def test_edge_unit_calls_reserve_for_update_only(self):
        buf = _spy_buffer()
        buf.add(
            [_edge(f"e{i}", "q0") for i in range(8)],
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert buf.edge_reservations == 1
        assert buf.tree_reservations == 0
        assert len(reservation.edges) == 8

    def test_complete_tree_unit_calls_tree_reservation_only(self):
        buf = _spy_buffer(replay_sampling_unit="complete_tree")
        buf.add(
            [_edge(f"e{i}", "q0") for i in range(8)],
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf,
            replay_sampling_unit="complete_tree",
            current_rollout_iteration=1,
        )
        assert buf.tree_reservations == 1
        assert buf.edge_reservations == 0
        assert reservation.edges

    def test_unknown_unit_raises(self):
        buf = _spy_buffer()
        with pytest.raises(ValueError, match="replay_sampling_unit"):
            reserve_replay_edges(
                buf, replay_sampling_unit="tree", current_rollout_iteration=1
            )

    def test_trainer_fit_uses_the_dispatcher(self):
        """The production trainer must contain no strictness-gated
        reservation branch — only the ``reserve_replay_edges`` dispatch.

        Checked textually because importing ``gear_ray_trainer`` requires the
        full verl/ray/torchdata stack that the CPU suite does not install.
        """
        trainer_source = (
            Path(__file__).resolve().parents[1] / "gear_ray_trainer.py"
        ).read_text()
        assert "reserve_replay_edges(" in trainer_source
        assert "reserve_complete_trees_for_update" not in trainer_source
        stripped = trainer_source.replace("reserve_replay_edges", "")
        assert ".reserve_for_update(" not in stripped


class TestEdgePathCaps:
    def test_666_tree_contributes_at_most_33_edges_per_question(self):
        buf = _spy_buffer(
            max_edges_per_question_per_iteration="auto",
            tree_shape=[6, 6, 6],
            trees_per_question=1,
        )
        assert buf.max_edges_per_question_per_iteration == 33
        buf.add(
            [_edge(f"e{i}", "q0") for i in range(258)],
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert len(reservation.edges) == 33
        assert buf.tree_reservations == 0

    def test_888_tree_contributes_at_most_73_edges_per_question(self):
        buf = _spy_buffer(
            max_edges_per_question_per_iteration="auto",
            tree_shape=[8, 8, 8],
            trees_per_question=1,
        )
        assert buf.max_edges_per_question_per_iteration == 73
        buf.add(
            [_edge(f"e{i}", "q0") for i in range(584)],
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert len(reservation.edges) == 73

    def test_516_candidates_select_exactly_512(self):
        buf = _spy_buffer()
        edges = []
        for q in range(4):
            edges.extend(_edge(f"q{q}-e{i}", f"q{q}") for i in range(129))
        assert len(edges) == 516
        buf.add(edges, generation_rollout_iteration=1, policy_snapshot_id="s0")
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert len(reservation.edges) == 512
        assert len(set(reservation.edge_ids)) == 512

    def test_edge_sampler_never_duplicates_rows(self):
        buf = _spy_buffer()
        buf.add(
            [_edge(f"e{i}", "q0") for i in range(600)],
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert len(reservation.edges) == 512
        assert len(set(reservation.edge_ids)) == len(reservation.edge_ids)


class TestCompleteTreeAblationNeverOvershoots:
    def _tree(self, tree_id: str, question_id: str, n: int) -> list[dict]:
        return [
            _edge(f"{tree_id}/e{i}", question_id, tree_id=tree_id)
            for i in range(n)
        ]

    def test_two_258_trees_never_return_516(self):
        buf = _spy_buffer(
            replay_sampling_unit="complete_tree",
            max_edges_per_question_per_iteration=1000,
        )
        buf.add(
            self._tree("t0", "q0", 258) + self._tree("t1", "q1", 258),
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf,
            replay_sampling_unit="complete_tree",
            current_rollout_iteration=1,
        )
        # 258 + 258 = 516 > 512: only one whole tree fits; never overshoot.
        assert len(reservation.edges) == 258
        assert reservation.stats["buffer/skipped_oversized_trees"] == 1

    def test_two_256_trees_fill_the_target_exactly(self):
        buf = _spy_buffer(
            replay_sampling_unit="complete_tree",
            max_edges_per_question_per_iteration=1000,
        )
        buf.add(
            self._tree("t0", "q0", 256) + self._tree("t1", "q1", 256),
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf,
            replay_sampling_unit="complete_tree",
            current_rollout_iteration=1,
        )
        assert len(reservation.edges) == 512

    def test_reservation_is_transactional(self):
        buf = _spy_buffer()
        buf.add(
            [_edge(f"e{i}", "q0") for i in range(10)],
            generation_rollout_iteration=1,
            policy_snapshot_id="s0",
        )
        reservation = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        assert len(buf._reserved) == 10
        buf.rollback(reservation)
        assert len(buf._reserved) == 0
        assert len(buf) == 10
        reservation2 = reserve_replay_edges(
            buf, replay_sampling_unit="edge", current_rollout_iteration=1
        )
        buf.commit(reservation2)
        assert len(buf) == 0


VALID_UNITS_DOC = VALID_REPLAY_SAMPLING_UNITS  # re-export guard for the gate
