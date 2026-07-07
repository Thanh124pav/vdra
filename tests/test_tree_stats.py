"""Tests for `aggregate_tree_stats` — the single source of truth for
the aggregate `gear/share_rate` / `gear/prune_rate` published by
`_construct_tree`.

Regression target: the local_value_share code path used to drop expanded
nodes from the denominator, so `share_rate` and `prune_rate` got inflated
and disagreed with the per-depth breakdown.
"""

from __future__ import annotations

from treetune.gear.logging_helpers import (
    aggregate_tree_stats,
    per_depth_action_counts,
)


def _make_mixed_tree():
    # 1 expand at root, 1 expand + 1 share at depth 1, 1 expand + 1 prune at
    # depth 2.  Total nodes with an action = 5.
    return {
        "gear_action": "expand",
        "gear_depth": 0,
        "children": [
            {
                "gear_action": "expand",
                "gear_depth": 1,
                "children": [
                    {"gear_action": "expand", "gear_depth": 2},
                    {"gear_action": "prune", "gear_depth": 2},
                ],
            },
            {"gear_action": "share", "gear_depth": 1},
        ],
    }


def test_aggregate_tree_stats_counts_all_actions():
    stats = aggregate_tree_stats(_make_mixed_tree())
    assert stats["gear/expanded_count"] == 3
    assert stats["gear/shared_count"] == 1
    assert stats["gear/pruned_count"] == 1
    # share_rate = 1 / 5, prune_rate = 1 / 5
    assert stats["gear/share_rate"] == 1 / 5
    assert stats["gear/prune_rate"] == 1 / 5


def test_aggregate_consistent_with_per_depth():
    tree = _make_mixed_tree()
    agg = aggregate_tree_stats(tree)
    per_depth = per_depth_action_counts(tree)

    sum_expand = sum(v for k, v in per_depth.items() if k.endswith("/expand_count"))
    sum_share = sum(v for k, v in per_depth.items() if k.endswith("/share_count"))
    sum_prune = sum(v for k, v in per_depth.items() if k.endswith("/prune_count"))
    assert agg["gear/expanded_count"] == sum_expand
    assert agg["gear/shared_count"] == sum_share
    assert agg["gear/pruned_count"] == sum_prune


def test_local_value_share_path_keeps_expanded_in_denominator():
    # Tree that mimics the local_value_share path output: most nodes stay
    # EXPAND, only one sibling is flipped to SHARE.  Old aggregate stats
    # (shared / (shared + pruned)) used to return 1.0; the new walk-based
    # impl correctly returns 1/4.
    tree = {
        "gear_action": "expand",
        "children": [
            {"gear_action": "expand"},
            {"gear_action": "expand"},
            {"gear_action": "share"},
        ],
    }
    stats = aggregate_tree_stats(tree)
    assert stats["gear/expanded_count"] == 3
    assert stats["gear/shared_count"] == 1
    assert stats["gear/share_rate"] == 0.25
    assert stats["gear/prune_rate"] == 0.0


def test_empty_tree_returns_empty_dict():
    # When no node carries an action label we should return {} — matches the
    # previous "if total else {}" behaviour so downstream demos rendering
    # silently skips the header.
    assert aggregate_tree_stats({}) == {}
    assert aggregate_tree_stats({"children": [{}]}) == {}
