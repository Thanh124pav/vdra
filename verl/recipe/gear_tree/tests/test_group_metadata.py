"""PLAN.md P0.N1/N2/N4: canonical grouping metadata propagation.

Verifies:
* extract_edges_from_tree assigns stable tree_id, parent_group_id,
  child_segment_id, child_index, allocated_k, sample_multiplicity, and
  queue_flush_id;
* the resulting edges carry a tree_summary aggregate;
* tree_data.edges_to_dataproto lifts the metadata into the row-level int64
  group tensors (tree_group_ids, parent_group_ids, queue_group_ids,
  allocated_k) and a float32 sample_multiplicity;
* validate_group_integrity accepts a well-formed fresh_iid tree and rejects a
  partial parent group.
"""

from __future__ import annotations

import pytest
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

import torch

from recipe.gear_tree.tree_advantage import extract_edges_from_tree
from recipe.gear_tree.tree_data import (
    group_tensors_for_edges,
    validate_group_integrity,
)


def _tree_two_parents():
    """Root -> {A, B}. A -> {A1, A2}. B is a leaf.

    Under SPO local-advantage, all four intermediate segments are trainable.
    """
    return {
        "reward": 0.5,
        "reward_std": 0.25,
        "full_text": "Q",
        "_request_object": {
            "_treetune__idx": 42,
            "problem": "1+1",
            "policy_snapshot_id": "snap:1",
        },
        "gear_segment_id": "root",
        "vdra_allocated_k": 2,
        "children": [
            {
                "text": " A",
                "full_text": "Q A",
                "reward": 0.8,
                "reward_std": 0.05,
                "leaf": False,
                "gear_segment_id": "root/0/0",
                "vdra_allocated_k": 2,
                "response_token_ids": [11, 12],
                "actor_shifted_log_probs": [-0.1, -0.2],
                "children": [
                    {
                        "text": " A1",
                        "full_text": "Q A A1",
                        "reward": 0.9,
                        "leaf": True,
                        "gear_segment_id": "root/0/0/1/0",
                        "response_token_ids": [21],
                        "actor_shifted_log_probs": [-0.3],
                    },
                    {
                        "text": " A2",
                        "full_text": "Q A A2",
                        "reward": 0.7,
                        "leaf": True,
                        "gear_segment_id": "root/0/0/1/1",
                        "response_token_ids": [22],
                        "actor_shifted_log_probs": [-0.4],
                    },
                ],
            },
            {
                "text": " B",
                "full_text": "Q B",
                "reward": 0.2,
                "leaf": True,
                "gear_segment_id": "root/0/1",
                "response_token_ids": [13],
                "actor_shifted_log_probs": [-0.5],
            },
        ],
    }


def test_extract_edges_assigns_canonical_group_metadata():
    edges = extract_edges_from_tree(_tree_two_parents(), only_adv_greater_than_zero=False)
    # 3 edges: A, A1, A2, B (root->A, root->B, A->A1, A->A2).
    assert len(edges) == 4
    # All edges share one tree_id.
    tree_ids = {e["tree_id"] for e in edges}
    assert len(tree_ids) == 1
    # Root's children share one parent_group_id; A's children share another.
    root_children = [e for e in edges if e["parent_segment_id"] == "root"]
    assert {e["parent_group_id"] for e in root_children} == {
        root_children[0]["parent_group_id"]
    }
    # Their allocated_k must be 2 (root has vdra_allocated_k=2).
    assert {e["allocated_k"] for e in root_children} == {2}
    a_children = [e for e in edges if e["parent_segment_id"] == "root/0/0"]
    assert len(a_children) == 2
    assert {e["parent_group_id"] for e in a_children} == {
        a_children[0]["parent_group_id"]
    }
    assert {e["allocated_k"] for e in a_children} == {2}
    # Sample multiplicity defaults to 1 under fresh_iid.
    assert all(e["sample_multiplicity"] == 1 for e in edges)
    # Every edge carries a tree_summary aggregate.
    summary = edges[0]["tree_summary"]
    assert summary["trainable_child_count"] == 4
    assert summary["expanded_parent_group_count"] == 2
    # Child indices within each parent are contiguous starting at 0.
    a_indices = sorted(e["child_index"] for e in a_children)
    assert a_indices == [0, 1]


def test_group_tensors_are_row_level_int64():
    edges = extract_edges_from_tree(_tree_two_parents(), only_adv_greater_than_zero=False)
    tensors = group_tensors_for_edges(edges)
    for key in ("tree_group_ids", "parent_group_ids", "queue_group_ids", "allocated_k"):
        assert tensors[key].dtype == torch.int64, key
        assert tensors[key].shape == (len(edges),), key
    assert tensors["sample_multiplicity"].dtype == torch.float32
    # Two parent groups -> two distinct parent_group_ids.
    unique_parents = torch.unique(tensors["parent_group_ids"]).numel()
    assert unique_parents == 2
    # One tree -> one distinct tree_group_id.
    assert torch.unique(tensors["tree_group_ids"]).numel() == 1


def test_validate_group_integrity_accepts_well_formed_fresh_iid_tree():
    edges = extract_edges_from_tree(_tree_two_parents(), only_adv_greater_than_zero=False)
    stats = validate_group_integrity(edges, strict_fresh_iid=True)
    assert stats["vdra/group_integrity_failures"] == 0
    assert stats["vdra/fresh_iid_parent_groups"] == 2


def test_validate_group_integrity_allows_zero_filtered_subset():
    # Zero-filter contract: rows stamped with realized_child_count ==
    # allocated_k may legitimately be a strict subset of the realized
    # children (exact-zero advantages are removed), so a partial retained
    # group must PASS as long as the pre-filter facts hold.
    edges = extract_edges_from_tree(_tree_two_parents(), only_adv_greater_than_zero=False)
    a_edges = [e for e in edges if e["parent_segment_id"] == "root/0/0"]
    partial_edges = [e for e in edges if e is not a_edges[0]]
    stats = validate_group_integrity(partial_edges, strict_fresh_iid=True)
    assert stats["vdra/group_integrity_failures"] == 0


def test_validate_group_integrity_rejects_construction_shortfall():
    # A parent that genuinely realized fewer children than allocated_k at
    # construction time must still fail — the pre-filter realized stamp
    # (or, for unstamped legacy rows, the retained count) is compared
    # against the allocation.
    edges = extract_edges_from_tree(_tree_two_parents(), only_adv_greater_than_zero=False)
    a_edges = [e for e in edges if e["parent_segment_id"] == "root/0/0"]
    for e in a_edges:
        e["realized_child_count"] = 1
    with pytest.raises(ValueError, match="allocated_k"):
        validate_group_integrity(edges, strict_fresh_iid=True)
    # Legacy rows without the stamp: dropping a sibling means realized falls
    # back to the retained count and the shortfall is still detected.
    edges = extract_edges_from_tree(_tree_two_parents(), only_adv_greater_than_zero=False)
    for e in edges:
        e.pop("realized_child_count", None)
    a_edges = [e for e in edges if e["parent_segment_id"] == "root/0/0"]
    partial_edges = [e for e in edges if e is not a_edges[0]]
    with pytest.raises(ValueError, match="allocated_k"):
        validate_group_integrity(partial_edges, strict_fresh_iid=True)


def test_validate_group_integrity_rejects_retained_overflow():
    # More retained rows than allocated_k is always a defect.
    edges = extract_edges_from_tree(_tree_two_parents(), only_adv_greater_than_zero=False)
    a_edges = [e for e in edges if e["parent_segment_id"] == "root/0/0"]
    extra = dict(a_edges[0])
    with pytest.raises(ValueError, match="exceeds allocated_k"):
        validate_group_integrity(edges + [extra], strict_fresh_iid=True)


def test_validate_group_integrity_ignores_multiplicity_for_weighted_reuse():
    edges = extract_edges_from_tree(_tree_two_parents(), only_adv_greater_than_zero=False)
    # Convert parent A's group into a weighted_reuse group with representative
    # multiplicity 2 and 3 — partial fresh_iid check must not fire.
    a_edges = [e for e in edges if e["parent_segment_id"] == "root/0/0"]
    a_edges[0]["sample_multiplicity"] = 2
    a_edges[1]["sample_multiplicity"] = 3
    stats = validate_group_integrity(edges, strict_fresh_iid=True)
    assert stats["vdra/weighted_reuse_parent_groups"] == 1
    assert stats["vdra/fresh_iid_parent_groups"] == 1
