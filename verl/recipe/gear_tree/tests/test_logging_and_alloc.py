"""Tests for tree logging and the modified budget-allocation formula."""

import json
import math
from pathlib import Path

from recipe.gear_tree.gear_core.gear.budget_allocation import allocate_branch_factors
from recipe.gear_tree.tree_logging import basic_tree_stats, TreeDemoLogger
from recipe.gear_tree.tree_rollout import SegmentSample, build_tree


def _mk_tree():
    calls = [0]

    def seg(prompt_ids, bf, mt):
        calls[0] += 1
        out = []
        for b in range(bf):
            finish = "stop" if (mt is None or (calls[0] + b) % 2 == 0) else "length"
            out.append(SegmentSample(token_ids=[10 + b, 11], text=f" s{b}", finish_reason=finish, logprobs=[-0.2, -0.3]))
        return out

    def grade(q, r, d):
        return float(len(r) % 2)

    return build_tree("Q", [1, 2], {"problem": "p", "answer": "1", "_treetune__idx": 0},
                      tree_shape=[2, 2], M=4, segment_fn=seg, grade_fn=grade)


def test_basic_tree_stats():
    tree = _mk_tree()
    s = basic_tree_stats(tree)
    assert s["num_nodes"] >= 3
    assert s["max_depth"] >= 1
    assert "per_depth" in s and s["per_depth"]
    assert s["tree_construction_seconds"] is not None


def test_demo_logger_writes_files(tmp_path: Path):
    logger = TreeDemoLogger(tmp_path, demo_examples_per_tree=2, full_tree_every_n_trees=1, full_tree_max_trees=2)
    tree = _mk_tree()
    logger.log_tree(tree, question_id=0)
    logger.close()
    demos = (tmp_path / "demos.jsonl").read_text().strip().splitlines()
    assert len(demos) == 1
    rec = json.loads(demos[0])
    assert rec["tree_idx"] == 1 and "stats" in rec
    assert (tmp_path / "demos.md").exists()
    # full-tree example dumped (every_n=1 => first tree).
    assert list((tmp_path / "full_trees").glob("tree_*.json"))


def test_allocation_uses_sigma_squared():
    # VDRA priority weight is exactly sqrt(C_s).
    nodes = [
        {"gear_segment_id": "a", "vdra_default_k": 1, "vdra_predicted_k": 6, "vdra_dispersion_C": 0.25},  # weight sqrt(0.25)=0.5
        {"gear_segment_id": "b", "vdra_default_k": 1, "vdra_predicted_k": 6, "vdra_dispersion_C": 1.0},   # weight sqrt(1.0)=1.0
    ]
    summ = allocate_branch_factors(nodes, total_budget=6)
    assert summ.weights["a"] == math.sqrt(0.25)
    assert summ.weights["b"] == math.sqrt(1.0)
    # budget split 1:2 -> a=2, b=4.
    assert summ.allocations["a"] == 2
    assert summ.allocations["b"] == 4


def test_allocation_uses_positive_dispersion_without_threshold():
    nodes = [{"gear_segment_id": "a", "vdra_default_k": 1, "vdra_predicted_k": 4, "vdra_dispersion_C": 0.1}]
    summ = allocate_branch_factors(nodes, total_budget=4)
    assert summ.weights["a"] == math.sqrt(0.1)
    assert summ.allocations["a"] == 4
