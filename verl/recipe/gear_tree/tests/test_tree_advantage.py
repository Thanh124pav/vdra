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


def test_pruned_edge_is_emitted_with_zero_advantage():
    tree = _tree()
    tree["children"][0]["pruned"] = True
    edges = extract_edges_from_tree(tree, only_adv_greater_than_zero=False)
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