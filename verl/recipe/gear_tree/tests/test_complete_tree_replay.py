"""PLAN.md P0.N6: complete-tree replay + group-aware microbatch packing.

Verifies:
* ``reserve_complete_trees_for_update`` never returns a partial parent group;
* one tree is never split across two reservations;
* pack_edges_into_microbatches keeps parent groups intact and same-tree edges
  contiguous;
* full-batch vs group-packed VDRA loss computations agree.
"""

from __future__ import annotations

import pytest
import torch
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

from recipe.gear_tree.policy_loss import (
    compute_policy_loss_vdra_node_balanced,
    hierarchical_reference_reduction,
)
from recipe.gear_tree.replay_buffer import (
    GearTreeReplayBuffer,
    pack_edges_into_microbatches,
)


def _make_edge(
    *,
    edge_id: str,
    tree_id: str,
    parent_group_id: str,
    child_index: int,
    allocated_k: int,
    question_id: str = "q0",
    sample_multiplicity: int = 1,
    reward: float = 0.5,
) -> dict:
    return {
        "edge_id": edge_id,
        "question_id": question_id,
        "tree_id": tree_id,
        "parent_group_id": parent_group_id,
        "child_segment_id": edge_id,
        "child_index": child_index,
        "allocated_k": allocated_k,
        "sample_multiplicity": sample_multiplicity,
        "queue_flush_id": 0,
        "query_token_ids": [1],
        "response_token_ids": [2, 3],
        "actor_shifted_log_probs": [-0.1, -0.2],
        "advantage": 1.0,
        "value": 0.8,
        "reward": reward,
    }


def _fresh_iid_tree(tree_id: str, question_id: str, k: int = 3) -> list[dict]:
    parent_group = f"{tree_id}/pg"
    return [
        _make_edge(
            edge_id=f"{tree_id}/e{i}",
            tree_id=tree_id,
            parent_group_id=parent_group,
            child_index=i,
            allocated_k=k,
            question_id=question_id,
        )
        for i in range(k)
    ]


def test_reserve_complete_trees_never_returns_partial_parent_group():
    buf = GearTreeReplayBuffer(
        target_edges_per_update=6,
        max_edges_per_question=100,
        max_edge_age=10,
        sampling_seed=1,
    )
    buf.add(_fresh_iid_tree("t0", "q0", k=3), generation_step=0, policy_snapshot_id="s0")
    buf.add(_fresh_iid_tree("t1", "q0", k=4), generation_step=0, policy_snapshot_id="s0")

    reservation = buf.reserve_complete_trees_for_update(current_step=0)
    # Any parent_group_id in the reservation must have every one of its rows.
    by_parent: dict[str, list[dict]] = {}
    for edge in reservation.edges:
        by_parent.setdefault(edge["parent_group_id"], []).append(edge)
    for pgid, rows in by_parent.items():
        assert len(rows) == rows[0]["allocated_k"], (pgid, rows)


def test_reserve_complete_trees_never_splits_a_tree():
    buf = GearTreeReplayBuffer(
        target_edges_per_update=3,
        max_edges_per_question=100,
        max_edge_age=10,
        sampling_seed=1,
    )
    buf.add(_fresh_iid_tree("t0", "q0", k=3), generation_step=0, policy_snapshot_id="s0")
    buf.add(_fresh_iid_tree("t1", "q0", k=3), generation_step=0, policy_snapshot_id="s0")

    reservation = buf.reserve_complete_trees_for_update(current_step=0)
    trees_seen = {e["tree_id"] for e in reservation.edges}
    # Whatever trees appeared must be fully present.
    for tid in trees_seen:
        rows = [e for e in reservation.edges if e["tree_id"] == tid]
        assert len(rows) == 3, (tid, len(rows))


def test_rollback_frees_the_reserved_tree():
    buf = GearTreeReplayBuffer(
        target_edges_per_update=3,
        max_edges_per_question=100,
        max_edge_age=10,
    )
    buf.add(_fresh_iid_tree("t0", "q0", k=3), generation_step=0, policy_snapshot_id="s0")
    reservation = buf.reserve_complete_trees_for_update(current_step=0)
    assert len(buf._reserved) == 3
    buf.rollback(reservation)
    assert len(buf._reserved) == 0
    # The tree is still available for a subsequent reservation.
    reservation2 = buf.reserve_complete_trees_for_update(current_step=1)
    assert len(reservation2.edges) == 3


def test_pack_microbatches_keeps_parent_groups_intact():
    edges = _fresh_iid_tree("t0", "q0", k=3) + _fresh_iid_tree("t1", "q0", k=2)
    packed = pack_edges_into_microbatches(edges, micro_batch_size=3)
    # Every parent_group_id must live in exactly one microbatch.
    parents_per_batch = [
        {e["parent_group_id"] for e in mb} for mb in packed
    ]
    all_parents = set().union(*parents_per_batch)
    for pgid in all_parents:
        occurrences = sum(1 for parents in parents_per_batch if pgid in parents)
        assert occurrences == 1, (pgid, occurrences)


def test_full_batch_vs_packed_microbatches_give_the_same_loss():
    # Uniform k=2, two trees, all rows fresh_iid so every child has its own
    # scalar loss. The node-balanced VDRA loss reduces token -> child ->
    # parent -> tree -> batch; packing into microbatches that keep parents
    # intact must not change the result.
    class _Cfg:
        clip_ratio = 0.2

        def get(self, k, default=None):
            return {"use_prob_mask": False, "ratio_threshold": 100.0}.get(k, default)

    # Build per-row losses: 6 rows, 2 trees x 3 parent-siblings.
    losses = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0, 1.0])
    per_token = losses.unsqueeze(1)
    old = torch.zeros_like(per_token)
    new = old.clone()
    adv = -per_token
    mask = torch.ones_like(per_token)
    parents = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    trees = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.int64)
    mults = torch.ones_like(losses)

    full_loss, *_ = compute_policy_loss_vdra_node_balanced(
        old_log_prob=old,
        log_prob=new,
        advantages=adv,
        response_mask=mask,
        config=_Cfg(),
        parent_group_ids=parents,
        tree_group_ids=trees,
        sample_multiplicity=mults,
    )
    ref = hierarchical_reference_reduction(losses, parents, trees, mults)
    assert torch.allclose(full_loss, ref)

    # "Pack" the batch by splitting the two trees into separate microbatches
    # and averaging *their* tree-losses. This must match the batch mean.
    tree0_loss, *_ = compute_policy_loss_vdra_node_balanced(
        old_log_prob=old[:4],
        log_prob=new[:4],
        advantages=adv[:4],
        response_mask=mask[:4],
        config=_Cfg(),
        parent_group_ids=parents[:4],
        tree_group_ids=trees[:4],
        sample_multiplicity=mults[:4],
    )
    tree1_loss, *_ = compute_policy_loss_vdra_node_balanced(
        old_log_prob=old[4:],
        log_prob=new[4:],
        advantages=adv[4:],
        response_mask=mask[4:],
        config=_Cfg(),
        parent_group_ids=parents[4:],
        tree_group_ids=trees[4:],
        sample_multiplicity=mults[4:],
    )
    packed_mean = 0.5 * (tree0_loss + tree1_loss)
    assert torch.allclose(packed_mean, full_loss, atol=1e-6)


def test_pack_microbatches_places_oversize_parent_group_alone():
    edges = _fresh_iid_tree("t0", "q0", k=5)
    packed = pack_edges_into_microbatches(edges, micro_batch_size=3)
    # 5-row parent group exceeds mb size; must sit alone in its own microbatch.
    assert len(packed) == 1
    assert len(packed[0]) == 5


def test_reserve_complete_trees_respects_per_question_cap_on_whole_trees():
    buf = GearTreeReplayBuffer(
        target_edges_per_update=100,
        max_edges_per_question=4,  # only 1 tree of size 3 will fit
        max_edge_age=10,
    )
    buf.add(_fresh_iid_tree("t0", "q0", k=3), generation_step=0, policy_snapshot_id="s0")
    buf.add(_fresh_iid_tree("t1", "q0", k=3), generation_step=0, policy_snapshot_id="s0")
    reservation = buf.reserve_complete_trees_for_update(current_step=0)
    # Only the first tree fits under the per-question cap.
    trees_in_reservation = {e["tree_id"] for e in reservation.edges}
    assert len(trees_in_reservation) == 1
