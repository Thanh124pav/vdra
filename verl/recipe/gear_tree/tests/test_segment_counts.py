"""PLAN.md P0.2: pre-filter segment counts on tree extraction.

Every realized (non-pruned) segment increments ``tree_total_segment_count``
and the corresponding ``queue_released_segment_count[q]`` regardless of
whether ``only_adv_greater_than_zero`` filters it from the retained edges.
"""

from __future__ import annotations

from recipe.gear_tree.tree_advantage import extract_edges_from_tree


def _tree(num_children=4, zero_advantages_mask=None, queue_ids=None):
    # Root reward 0, child rewards mix zero/nonzero. Each child is a leaf
    # segment so we only have one parent (root).
    children = []
    for i in range(num_children):
        reward = 0.0
        if zero_advantages_mask is None or not zero_advantages_mask[i]:
            reward = 1.0
        child = {
            "reward": reward,
            "gear_segment_id": f"c{i}",
            "response_token_ids": [i + 1],
            "actor_shifted_log_probs": [-0.1],
            "leaf": True,
            "text": "x",
        }
        if queue_ids is not None:
            child["vdra_queue_flush_id"] = queue_ids[i]
        children.append(child)
    return {
        "_request_object": {
            "_treetune__idx": "q1",
            "policy_snapshot_id": "snap",
            "rollout_iteration": 3,
            "tree_instance_id": "T-test-1",
        },
        "reward": 0.0,
        "full_text": "root",
        "full_token_ids": [1],
        "vdra_allocated_k": num_children,
        "vdra_queue_flush_id": queue_ids[0] if queue_ids else 0,
        "children": children,
    }


def test_tree_total_segment_count_matches_realized_children_no_filter():
    tree = _tree(num_children=4)
    edges = extract_edges_from_tree(tree)
    assert len(edges) == 4
    assert all(e["tree_total_segment_count"] == 4 for e in edges)
    summary = edges[0]["tree_summary"]
    assert summary["tree_total_segment_count"] == 4


def test_tree_total_segment_count_survives_zero_advantage_filter():
    # PLAN.md P0.2 acceptance: contributions [2, 0, 0, 0] -> 1 retained row,
    # tree_total_segment_count == 4, contribution == 2 / 4 handled by loss.
    tree = _tree(num_children=4, zero_advantages_mask=[False, True, True, True])
    edges = extract_edges_from_tree(tree, only_adv_greater_than_zero=True)
    assert len(edges) == 1
    assert edges[0]["tree_total_segment_count"] == 4
    summary = edges[0]["tree_summary"]
    assert summary["tree_total_segment_count"] == 4


def test_queue_release_counts_sum_to_tree_total():
    # PLAN.md P0.2: sum_q queue_released_segment_count[q] == N_seg(T).
    tree = _tree(
        num_children=4,
        queue_ids=["q0", "q0", "q1", "q1"],
    )
    edges = extract_edges_from_tree(tree)
    summary = edges[0]["tree_summary"]
    total = summary["tree_total_segment_count"]
    queue_totals = summary["queue_released_segment_count"]
    assert sum(queue_totals.values()) == total
    assert queue_totals["q0"] == 2
    assert queue_totals["q1"] == 2


def test_pruned_placeholder_does_not_change_tree_total():
    # PLAN.md P0.2 acceptance: pruned=True placeholder is not counted.
    children = [
        {"reward": 1.0, "gear_segment_id": "c0", "response_token_ids": [1],
         "actor_shifted_log_probs": [-0.1], "leaf": True, "text": "x"},
        {"reward": 0.0, "gear_segment_id": "c1", "response_token_ids": [2],
         "actor_shifted_log_probs": [-0.1], "leaf": True, "text": "y",
         "pruned": True},
    ]
    tree = {
        "_request_object": {
            "_treetune__idx": "q1",
            "policy_snapshot_id": "snap",
            "rollout_iteration": 4,
            "tree_instance_id": "T-test-pruned",
        },
        "reward": 0.0,
        "full_text": "root",
        "full_token_ids": [1],
        "vdra_allocated_k": 1,
        "vdra_queue_flush_id": 0,
        "children": children,
    }
    edges = extract_edges_from_tree(tree)
    assert len(edges) == 1
    assert edges[0]["tree_total_segment_count"] == 1


def test_zero_advantage_filter_allows_partial_fresh_iid():
    # PLAN.md P0.2 rule: retained_row_count <= realized_child_count == allocated_k.
    # Under strict_fresh_iid this must NOT raise even after filtering.
    tree = _tree(num_children=4, zero_advantages_mask=[False, True, True, True])
    edges = extract_edges_from_tree(
        tree, only_adv_greater_than_zero=True, strict_fresh_iid=True
    )
    assert len(edges) == 1  # only the first child retained
    # allocated_k stays at 4.
    assert edges[0]["allocated_k"] == 4
