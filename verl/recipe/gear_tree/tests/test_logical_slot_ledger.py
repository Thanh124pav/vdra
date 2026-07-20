"""PLAN.md §1.2 (user decision 2026-07-21): logical-slot ledger.

Sparse tensor execution must not change the mathematical objective:

* advantages are computed from the COMPLETE realized sibling set, before any
  filtering (required tests 1-2);
* an exact-zero-advantage child becomes a metadata-only LOGICAL SLOT
  (``trainable_edge_id=None``) that keeps its identity, question, and
  pre-filter ``response_token_count`` — never its tensor payload;
* replay stores, caps, ages, reserves and counts SLOTS exactly like
  trainable edges, so ``target_edges_per_iteration`` stays a logical-slot
  target (required test 5);
* no parent-group proportional attribution exists anywhere — slots carry
  their own bookkeeping (required test 6).
"""

from __future__ import annotations

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()

from recipe.gear_tree.replay_buffer import (  # noqa: E402
    GearTreeReplayBuffer,
    is_ledger_slot,
)
from recipe.gear_tree.tree_advantage import extract_edges_from_tree  # noqa: E402
from recipe.gear_tree.tree_data import normalize_generated_edges  # noqa: E402


def _tree4(rewards=(0.8, 0.2, 0.5, 0.5)):
    """One parent, four realized children; mean baseline 0.5 gives
    advantages [+0.3, -0.3, 0, 0]. Child i carries (i % 3) + 1 tokens."""
    return {
        "reward": 0.5,
        "reward_std": 0.25,
        "full_text": "Q",
        "_request_object": {"_treetune__idx": 7, "problem": "1+1"},
        "children": [
            {
                "text": f" s{i}",
                "full_text": f"Q s{i}",
                "reward": r,
                "reward_std": 0.0,
                "leaf": True,
                "response_token_ids": [11, 12, 13][: (i % 3) + 1],
                "actor_shifted_log_probs": [-0.1, -0.2, -0.3][: (i % 3) + 1],
            }
            for i, r in enumerate(rewards)
        ],
    }


def _extract(**kwargs):
    return extract_edges_from_tree(_tree4(), tree_update_mode="spo", **kwargs)


class TestExtractionLedger:
    def test_zero_children_become_metadata_only_slots(self):
        records = _extract(only_adv_greater_than_zero=True, emit_zero_slots=True)
        slots = [r for r in records if is_ledger_slot(r)]
        rows = [r for r in records if not is_ledger_slot(r)]
        assert len(records) == 4
        assert len(slots) == 2 and len(rows) == 2
        for slot in slots:
            assert slot["advantage"] == 0.0
            assert slot["advantage_is_zero"] is True
            assert slot["trainable_edge_id"] is None
            assert slot["response_token_count"] > 0
            for payload in ("query_token_ids", "response_token_ids", "actor_shifted_log_probs"):
                assert payload not in slot
        # Children 2 and 3 have (2%3)+1 = 3 and (3%3)+1 = 1 tokens.
        assert sorted(s["response_token_count"] for s in slots) == [1, 3]
        assert sorted(r["advantage"] for r in rows) == pytest.approx([-0.3, 0.3])

    def test_advantages_identical_with_sparse_execution_on_or_off(self):
        """Required tests 1-2: the baseline uses the COMPLETE sibling set,
        so enabling sparse execution never changes any advantage."""
        dense = _extract(only_adv_greater_than_zero=False)
        sparse = _extract(only_adv_greater_than_zero=True, emit_zero_slots=True)
        dense_by_child = {e["child_segment_id"]: e for e in dense}
        assert len(dense_by_child) == 4
        for rec in sparse:
            ref = dense_by_child[rec["child_segment_id"]]
            assert rec["advantage"] == pytest.approx(ref["advantage"])
            assert rec["value"] == pytest.approx(ref["value"])

    def test_summary_counts_slots_separately(self):
        records = _extract(only_adv_greater_than_zero=True, emit_zero_slots=True)
        summary = records[0]["tree_summary"]
        assert summary["tree_total_segment_count"] == 4
        assert summary["retained_edge_count"] == 2
        assert summary["ledger_slot_count"] == 2

    def test_without_emit_flag_zero_rows_still_drop(self):
        """Pre-flip behavior preserved: the legacy drop path is unchanged
        until the canonical trainer enables the ledger."""
        records = _extract(only_adv_greater_than_zero=True)
        assert len(records) == 2
        assert not any(is_ledger_slot(r) for r in records)

    def test_dense_mode_keeps_full_zero_rows(self):
        records = _extract(only_adv_greater_than_zero=False)
        assert len(records) == 4
        assert not any(is_ledger_slot(r) for r in records)
        zero_rows = [r for r in records if r["advantage_is_zero"]]
        assert len(zero_rows) == 2
        for row in zero_rows:
            assert row["response_token_ids"]
            assert row["response_token_count"] == len(row["response_token_ids"])


def _stamp_snapshot(records, snapshot="snap"):
    for rec in records:
        rec["policy_snapshot_id"] = snapshot
    return records


class TestNormalizeLedger:
    def test_slots_get_edge_ids_and_rows_point_at_themselves(self):
        records = _stamp_snapshot(
            _extract(only_adv_greater_than_zero=True, emit_zero_slots=True)
        )
        normalized = normalize_generated_edges(records, snapshot_id="snap")
        slots = [r for r in normalized if is_ledger_slot(r)]
        rows = [r for r in normalized if not is_ledger_slot(r)]
        assert len(slots) == 2 and len(rows) == 2
        for slot in slots:
            assert slot["edge_id"]
            assert slot["trainable_edge_id"] is None
        for row in rows:
            assert row["trainable_edge_id"] == row["edge_id"]
        assert len({r["edge_id"] for r in normalized}) == 4

    def test_slot_with_nonzero_advantage_rejected(self):
        records = _stamp_snapshot(
            _extract(only_adv_greater_than_zero=True, emit_zero_slots=True)
        )
        slot = next(r for r in records if is_ledger_slot(r))
        slot["advantage"] = 0.25
        with pytest.raises(ValueError, match="zero"):
            normalize_generated_edges(records, snapshot_id="snap")


def _slot(edge_id, question_id="q", token_count=3, step=0, active=None, threshold=0.9):
    return {
        "edge_id": str(edge_id),
        "question_id": str(question_id),
        "tree_id": f"tree-{question_id}",
        "parent_group_id": f"tree-{question_id}/p0",
        "policy_snapshot_id": "snap",
        "generation_step": int(step),
        "advantage": 0.0,
        "advantage_is_zero": True,
        "trainable_edge_id": None,
        "response_token_count": int(token_count),
        # PLAN.md §3/§4: a zero slot's active-token count is stamped at
        # extraction and can never be recomputed later.
        "prob_mask_token_count": int(
            token_count if active is None else active
        ),
        "probability_mask_threshold": float(threshold),
        "sample_multiplicity": 1,
    }


def _edge(edge_id, question_id="q", step=0, advantage=1.0):
    return {
        "edge_id": str(edge_id),
        "question_id": str(question_id),
        "generation_step": int(step),
        "policy_snapshot_id": "snap",
        "query_token_ids": [1, 2],
        "response_token_ids": [3, 4],
        "actor_shifted_log_probs": [-0.1, -0.2],
        "advantage": advantage,
        "value": 0.5,
        "reward": 1.0,
        "depth": 0,
        "leaf": False,
        "pruned": False,
        "tree_update_mode": "spo",
    }


def _buffer(**kwargs):
    cfg = {
        "target_edges_per_update": 4,
        "max_edges_per_question": 32,
        "max_edge_age": 8,
        "sampling_seed": 13,
    }
    cfg.update(kwargs)
    return GearTreeReplayBuffer(**cfg)


class TestReplaySlots:
    def test_reservation_counts_slots_toward_the_logical_target(self):
        """Required test 5: the 512-style target stays a LOGICAL count even
        when fewer rows will be tensorized."""
        buf = _buffer(target_edges_per_update=4)
        buf.add(
            [_edge(f"e{i}", question_id=f"q{i}") for i in range(3)]
            + [_slot(f"z{i}", question_id=f"qz{i}") for i in range(3)],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        reservation = buf.reserve_for_update(current_rollout_iteration=1)
        assert len(reservation.edges) == 4  # slots + edges together hit the cap

    def test_per_question_cap_counts_slots(self):
        buf = _buffer(target_edges_per_update=100, max_edges_per_question=2)
        buf.add(
            [_edge("e0", question_id="one"), _edge("e1", question_id="one")]
            + [_slot("z0", question_id="one"), _slot("z1", question_id="one")],
            generation_rollout_iteration=0,
            policy_snapshot_id="snap",
        )
        sampled, _ = buf.sample_for_update(current_rollout_iteration=1, remove=False)
        assert len(sampled) == 2

    def test_slot_expiry_uses_rollout_iteration_age(self):
        buf = _buffer(max_edge_age=2)
        buf.add(
            [_slot("z0")], generation_rollout_iteration=0, policy_snapshot_id="snap"
        )
        assert buf.expire(current_rollout_iteration=1) == []
        assert buf.expire(current_rollout_iteration=2) == ["z0"]

    def test_duplicate_slot_id_rejected(self):
        buf = _buffer()
        buf.add([_slot("dup")], generation_rollout_iteration=0, policy_snapshot_id="snap")
        with pytest.raises(ValueError, match="duplicate"):
            buf.add(
                [_slot("dup")], generation_rollout_iteration=0, policy_snapshot_id="snap"
            )

    def test_slot_with_payload_rejected(self):
        bad = _slot("z0")
        bad["response_token_ids"] = [1, 2, 3]
        with pytest.raises(ValueError, match="metadata-only"):
            _buffer().add(
                [bad], generation_rollout_iteration=0, policy_snapshot_id="snap"
            )

    def test_slot_with_nonzero_advantage_rejected(self):
        bad = _slot("z0")
        bad["advantage"] = 0.5
        with pytest.raises(ValueError, match="zero advantage"):
            _buffer().add(
                [bad], generation_rollout_iteration=0, policy_snapshot_id="snap"
            )

    def test_slot_without_token_count_rejected(self):
        bad = _slot("z0")
        bad.pop("response_token_count")
        with pytest.raises(ValueError, match="missing required fields"):
            _buffer().add(
                [bad], generation_rollout_iteration=0, policy_snapshot_id="snap"
            )

    def test_slots_carry_no_attribution_fields(self):
        """Required test 6: sibling groups may split across reservations —
        slots own their bookkeeping; nothing redistributes 'zero mass' onto
        nonzero siblings."""
        records = _extract(only_adv_greater_than_zero=True, emit_zero_slots=True)
        for rec in records:
            for forbidden in (
                "denominator_segment_share",
                "denominator_token_share",
                "group_zero_token_count",
                "group_nonzero_segment_count",
            ):
                assert forbidden not in rec