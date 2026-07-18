import pytest
import torch

from verl import DataProto
from recipe.gear_tree.tree_advantage import (
    add_tree_advantage_tensors,
    extract_edges_from_tree,
    token_fields_for_edges,
)


def _tree():
    return {
        "reward": 0.5,
        "reward_std": 0.25,
        "full_text": "Q",
        "_request_object": {"_treetune__idx": 7, "problem": "1+1"},
        "children": [
            {
                "text": " step",
                "full_text": "Q step",
                "reward": 0.8,
                "reward_std": 0.0,
                "leaf": True,
                "response_token_ids": [11, 12, 13],
                "actor_shifted_log_probs": [-0.1, -0.2, -0.3],
            }
        ],
    }


@pytest.mark.parametrize(
    "mode,expected_adv,expected_value",
    [
        ("spo", 0.3, 0.8),
        ("treepo_style_ablation", 0.3, 0.8),
        ("treerl_style_ablation", 0.52, 1.02),
        # PLAN.md P0.N3: legacy aliases stay supported for vendor parity.
        ("treepo_original", 0.3, 0.8),
        ("treerl_original", 0.52, 1.02),
    ],
)
def test_extract_edges_matches_vendor_update_modes(mode, expected_adv, expected_value):
    edges = extract_edges_from_tree(
        _tree(),
        only_adv_greater_than_zero=False,
        tree_update_mode=mode,
        treepo_global_weight=0.5,
        treerl_gamma=0.9,
    )
    assert len(edges) == 1
    assert edges[0]["question_id"] == 7
    assert edges[0]["query_text"] == "Q"
    assert edges[0]["response_text"] == " step"
    assert edges[0]["response_token_ids"] == [11, 12, 13]
    assert edges[0]["advantage"] == pytest.approx(expected_adv)
    assert edges[0]["value"] == pytest.approx(expected_value)


def test_pruned_edge_is_dropped_by_default():
    # PLAN.md P0.1: administrative pruned=True placeholder rows must NOT enter
    # DataProto or the parent denominator. Default emit_pruned_edges=False.
    tree = _tree()
    tree["children"][0]["pruned"] = True
    edges = extract_edges_from_tree(tree, only_adv_greater_than_zero=False)
    assert edges == []


def test_pruned_edge_kept_when_emit_flag_true():
    # Diagnostics-only: callers can still opt in to inspecting pruned edges.
    tree = _tree()
    tree["children"][0]["pruned"] = True
    edges = extract_edges_from_tree(
        tree, only_adv_greater_than_zero=False, emit_pruned_edges=True
    )
    assert len(edges) == 1
    assert edges[0]["pruned"] is True
    assert edges[0]["advantage"] == 0.0


def test_token_fields_broadcast_edge_scalars_to_valid_response_tokens():
    edge = extract_edges_from_tree(_tree(), only_adv_greater_than_zero=False)[0]
    response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    tensors = token_fields_for_edges([edge], response_mask)

    assert torch.equal(tensors["advantages"], torch.tensor([[0.3, 0.3, 0.3, 0.0]]))
    assert torch.equal(tensors["values"], torch.tensor([[0.8, 0.8, 0.8, 0.0]]))
    assert torch.equal(tensors["returns"], torch.tensor([[0.8, 0.8, 0.8, 0.0]]))
    assert torch.equal(tensors["token_level_rewards"], torch.tensor([[0.0, 0.0, 0.8, 0.0]]))
    assert torch.equal(tensors["old_log_probs"], torch.tensor([[-0.1, -0.2, -0.3, 0.0]]))


def test_add_tree_advantage_tensors_mutates_dataproto_batch():
    edge = extract_edges_from_tree(_tree(), only_adv_greater_than_zero=False)[0]
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[11, 12, 13, 0]]),
            "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
        }
    )
    out = add_tree_advantage_tensors(data, [edge])
    assert out is data
    assert "advantages" in data.batch.keys()
    assert "old_log_probs" in data.batch.keys()
    assert data.batch["advantages"][0, 0].item() == pytest.approx(0.3)

def _tree_with_alloc(k: int, mark_multiplicity: int = 1):
    """Build a small tree stamped with vdra_allocated_k=k on the root."""
    root = {
        "reward": 0.5,
        "reward_std": 0.25,
        "full_text": "Q",
        "_request_object": {"_treetune__idx": 42, "problem": "1+1"},
        "vdra_allocated_k": k,
        "children": [
            {
                "text": f" seg{i}",
                "full_text": f"Q seg{i}",
                "reward": 0.4 + 0.1 * i,
                "reward_std": 0.0,
                "leaf": True,
                "response_token_ids": [10 + i, 11 + i, 12 + i],
                "actor_shifted_log_probs": [-0.1, -0.2, -0.3],
                "sample_multiplicity": mark_multiplicity,
            }
            for i in range(k)
        ],
    }
    return root


def test_zero_advantage_realized_child_remains_in_parent_group():
    # PLAN.md P0.1 acceptance: a zero-advantage realized child stays in the
    # parent group (allocated_k rows preserved).
    tree = _tree_with_alloc(k=3)
    # Force one child's reward to equal the parent so its advantage is 0.
    tree["children"][1]["reward"] = 0.5
    edges = extract_edges_from_tree(
        tree, only_adv_greater_than_zero=False, tree_update_mode="spo"
    )
    assert len(edges) == 3
    zero_adv_rows = [e for e in edges if float(e["advantage"]) == 0.0]
    assert len(zero_adv_rows) == 1


def test_strict_fresh_iid_rejects_partial_realized_group():
    # PLAN.md P0.1 acceptance: dropping a realized child breaks the invariant.
    tree = _tree_with_alloc(k=3)
    tree["children"].pop()  # only 2 realized rows for allocated_k=3
    with pytest.raises(ValueError, match="fresh_iid"):
        extract_edges_from_tree(
            tree, only_adv_greater_than_zero=False, strict_fresh_iid=True
        )


def test_strict_fresh_iid_rejects_multiplicity_other_than_one():
    tree = _tree_with_alloc(k=2, mark_multiplicity=3)
    with pytest.raises(ValueError, match="sample_multiplicity"):
        extract_edges_from_tree(
            tree, only_adv_greater_than_zero=False, strict_fresh_iid=True
        )
