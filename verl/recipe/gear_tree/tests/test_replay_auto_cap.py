"""PLAN.md P0.2 — auto per-question replay cap tests.

The main config uses ``max_edges_per_question_per_iteration: auto`` which the
buffer resolves at startup from ``tree_shape`` and ``max_edge_age_iterations``
using

    E_max = sum_{d=1..D} prod_{l=1..d} b_l
    C_question = ceil(R * E_max / max_edge_age_iterations)

Regressions to a stale hard-coded 32, or dropping the ``R`` factor when the
config generates more than one tree per question per iteration, are rejected
here.
"""

from __future__ import annotations

import math

import pytest

from recipe.gear_tree.replay_buffer import (
    GearTreeReplayBuffer,
    compute_max_edges_per_question,
)


# ---------- pure formula ----------------------------------------------------

def test_formula_666_age8_R1_is_33():
    assert compute_max_edges_per_question(
        [6, 6, 6], trees_per_question=1, max_edge_age_iterations=8
    ) == 33


def test_formula_888_age8_R1_is_73():
    assert compute_max_edges_per_question(
        [8, 8, 8], trees_per_question=1, max_edge_age_iterations=8
    ) == 73


def test_formula_scales_linear_in_R():
    base = compute_max_edges_per_question(
        [6, 6, 6], trees_per_question=1, max_edge_age_iterations=8
    )
    r2 = compute_max_edges_per_question(
        [6, 6, 6], trees_per_question=2, max_edge_age_iterations=8
    )
    # ceil(2*258/8) = ceil(64.5) = 65 vs base=33; must strictly grow.
    assert r2 > base
    assert r2 == math.ceil(2 * 258 / 8)


def test_formula_smaller_age_window_larger_cap():
    small = compute_max_edges_per_question(
        [6, 6, 6], trees_per_question=1, max_edge_age_iterations=4
    )
    large = compute_max_edges_per_question(
        [6, 6, 6], trees_per_question=1, max_edge_age_iterations=8
    )
    assert small > large


def test_formula_rejects_zero_age():
    with pytest.raises(ValueError):
        compute_max_edges_per_question(
            [6, 6, 6], trees_per_question=1, max_edge_age_iterations=0
        )


def test_formula_rejects_empty_shape():
    with pytest.raises(ValueError):
        compute_max_edges_per_question(
            [], trees_per_question=1, max_edge_age_iterations=8
        )


# ---------- buffer wiring ---------------------------------------------------

def _buffer_auto(tree_shape, *, age=8, R=1, target=512):
    return GearTreeReplayBuffer(
        target_edges_per_iteration=target,
        max_edge_age_iterations=age,
        max_edges_per_question_per_iteration="auto",
        tree_shape=tree_shape,
        trees_per_question=R,
    )


def test_buffer_auto_666_age8_resolves_to_33():
    buf = _buffer_auto([6, 6, 6], age=8, R=1)
    assert buf.max_edges_per_question_per_iteration == 33
    assert buf.resolved_max_edges_per_question_per_iteration == 33
    assert buf.max_edges_per_question_cap_source == "auto"


def test_buffer_auto_888_age8_resolves_to_73():
    buf = _buffer_auto([8, 8, 8], age=8, R=1)
    assert buf.max_edges_per_question_per_iteration == 73


def test_buffer_auto_rejects_missing_tree_shape():
    with pytest.raises(ValueError, match="tree_shape"):
        GearTreeReplayBuffer(
            target_edges_per_iteration=512,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration="auto",
        )


def test_buffer_override_records_source():
    buf = GearTreeReplayBuffer(
        target_edges_per_iteration=64,
        max_edge_age_iterations=4,
        max_edges_per_question_per_iteration=16,
        tree_shape=[6, 6, 6],
    )
    assert buf.max_edges_per_question_per_iteration == 16
    assert buf.max_edges_per_question_cap_source == "override"


# ---------- sampling under the resolved cap --------------------------------

def _edge(edge_id, question_id="q0"):
    return {
        "edge_id": str(edge_id),
        "question_id": str(question_id),
        "query_token_ids": [0],
        "response_token_ids": [1],
        "actor_shifted_log_probs": [0.0],
        "advantage": 0.0,
        "value": 0.0,
        "reward": 0.0,
    }


def test_sampled_per_question_never_exceeds_resolved_cap():
    # Small config so we can enumerate everything.
    buf = GearTreeReplayBuffer(
        target_edges_per_iteration=32,
        max_edge_age_iterations=4,
        max_edges_per_question_per_iteration="auto",
        tree_shape=[4, 4],  # E_max = 4 + 16 = 20; cap = ceil(20/4) = 5
    )
    assert buf.max_edges_per_question_per_iteration == 5
    # Stuff the buffer with 15 edges for one question.
    buf.add(
        [_edge(i, question_id="q0") for i in range(15)],
        generation_rollout_iteration=0,
        policy_snapshot_id="snap",
    )
    sampled, stats = buf.sample_for_update(current_rollout_iteration=1)
    # Per-question cap must clamp to 5, no matter that target is 32.
    assert stats["buffer/edges_per_question_max"] <= 5
    assert stats["buffer/resolved_max_edges_per_question_per_iteration"] == 5
    assert len(sampled) == 5


def test_replay_age_uses_rollout_iteration_not_global_step():
    buf = _buffer_auto([4, 4], age=3, R=1)
    # 3 edges added at iterations 0/1/2 in one buffer.
    for i, gri in enumerate([0, 1, 2]):
        buf.add(
            [_edge(f"e{i}", question_id="q")],
            generation_rollout_iteration=gri,
            policy_snapshot_id="snap",
        )
    # Now advance to iteration 3 — edge at iter 0 should expire (3-0>=3).
    sampled, _ = buf.sample_for_update(current_rollout_iteration=3)
    ids = sorted(e["edge_id"] for e in sampled)
    assert "e0" not in ids
    assert set(ids) == {"e1", "e2"}
