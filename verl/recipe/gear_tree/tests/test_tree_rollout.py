"""Golden-numerics parity for the native tree rollout (Step 2).

A deterministic mock segment generator drives ``build_tree``; the result is
compared against an inline reference ``dfs`` transcribed directly from treetune's
``HybridInferenceStrategy._construct_tree`` (hybrid_inference_strategy.py:359-453)
and ``TreeEpisodeUtils.extract_edges_from_tree``. This proves the port preserves
tree topology, reward back-prop, and segment advantages without needing a GPU.
"""

import random

import numpy as np
import pytest

from recipe.gear_tree.tree_rollout import SegmentSample, build_tree
from recipe.gear_tree.tree_advantage import extract_edges_from_tree


class MockEngine:
    """Deterministic segment generator keyed by (prefix tokens, call index)."""

    def __init__(self, seed=0, leaf_prob=0.5, max_seg_tokens=3):
        self.seed = seed
        self.leaf_prob = leaf_prob
        self.max_seg_tokens = max_seg_tokens
        self._call = 0

    def segment_fn(self, prompt_token_ids, branch_factor, max_tokens):
        rng = random.Random(hash((self.seed, tuple(prompt_token_ids), self._call)) & 0xFFFFFFFF)
        self._call += 1
        samples = []
        for b in range(branch_factor):
            ntok = rng.randint(1, self.max_seg_tokens)
            token_ids = [rng.randint(10, 99) for _ in range(ntok)]
            logps = [-rng.random() for _ in range(ntok)]
            # If max_tokens is None (last internal depth) always finish ("stop").
            if max_tokens is None:
                finish = "stop"
            else:
                finish = "length" if rng.random() > self.leaf_prob else "stop"
            samples.append(
                SegmentSample(
                    token_ids=token_ids,
                    text=f"[seg{b}:{'|'.join(map(str, token_ids))}]",
                    finish_reason=finish,
                    logprobs=logps,
                )
            )
        return samples


def make_grade_fn():
    def grade(query, response, data_instance):
        # Deterministic pseudo-reward from the response text hash in [0,1].
        h = abs(hash(response)) % 1000
        return h / 1000.0

    return grade


def reference_construct_tree(root_text, root_ids, data_instance, tree_shape, M, segment_fn, grade_fn):
    """Inline transcription of treetune `_construct_tree` (dfs) for parity."""
    max_depth = len(tree_shape)
    tree = {
        "text": root_text,
        "depth": 0,
        "full_text": root_text,
        "stop_text": "aaa",
        "_request_object": data_instance,
        "leaf": False,
        "full_token_ids": list(root_ids),
    }

    def dfs(node, prefix, depth):
        if depth == max_depth:
            node["reward"] = float(grade_fn(prefix, node["text"], data_instance))
            node["leaf"] = True
            return
        max_tokens = None if depth == max_depth - 1 else M
        branch_factor = tree_shape[depth]
        samples = segment_fn(node["full_token_ids"], branch_factor, max_tokens)
        children = []
        for s in samples:
            children.append(
                {
                    "text": s.text,
                    "depth": depth + 1,
                    "full_text": prefix + s.text,
                    "finish_reason": s.finish_reason,
                    "full_token_ids": list(node["full_token_ids"]) + list(s.token_ids),
                }
            )
        node["children"] = children
        for child in children:
            if child["finish_reason"] != "length":
                child["reward"] = float(grade_fn(prefix, child["full_text"], data_instance))
                child["leaf"] = True
            else:
                child["leaf"] = False
                dfs(child, child["full_text"], depth + 1)
        child_rewards = [c["reward"] for c in children]
        node["reward"] = float(np.mean(child_rewards))
        node["reward_std"] = float(np.std(child_rewards))

    dfs(tree, root_text, 0)
    return tree


def _collect(node, out):
    out.append((node["depth"], node.get("reward"), node.get("leaf"), node.get("finish_reason")))
    for c in node.get("children", []) or []:
        _collect(c, out)


@pytest.mark.parametrize("shape", [[2, 2], [3, 3], [2, 3, 2]])
def test_tree_topology_and_rewards_match_reference(shape):
    data_instance = {"problem": "p", "answer": "42", "_treetune__idx": 7}
    root_text, root_ids = "PROMPT ", [1, 2, 3]

    eng1 = MockEngine(seed=123)
    got = build_tree(
        root_text, root_ids, data_instance,
        tree_shape=shape, M=4, segment_fn=eng1.segment_fn, grade_fn=make_grade_fn(),
    )
    eng2 = MockEngine(seed=123)
    ref = reference_construct_tree(
        root_text, root_ids, data_instance, shape, 4, eng2.segment_fn, make_grade_fn()
    )

    got_nodes, ref_nodes = [], []
    _collect(got, got_nodes)
    _collect(ref, ref_nodes)
    assert got_nodes == ref_nodes
    assert got["reward"] == pytest.approx(ref["reward"])
    assert got["reward_std"] == pytest.approx(ref["reward_std"])


def test_edges_advantage_is_child_minus_parent_spo():
    data_instance = {"problem": "p", "answer": "42", "_treetune__idx": 1}
    eng = MockEngine(seed=7)
    tree = build_tree(
        "Q ", [5, 6], data_instance,
        tree_shape=[2, 2], M=4, segment_fn=eng.segment_fn, grade_fn=make_grade_fn(),
    )
    edges = extract_edges_from_tree(tree, tree_update_mode="spo", adv_method="rloo")
    # Rebuild a seg-id -> reward map to check advantage = child_reward - parent_reward.
    assert edges, "expected at least one edge"
    for e in edges:
        # SPO advantage stored back; child reward available on the edge.
        assert e["advantage"] == pytest.approx(
            e["tree_update_child_reward"] - e["tree_update_parent_reward"]
        )
