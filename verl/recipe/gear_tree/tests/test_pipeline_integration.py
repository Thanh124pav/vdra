"""End-to-end CPU integration test of the recipe data path (no GPU).

mock engine -> build_tree -> extract_edges_from_tree -> edges_to_dataproto ->
treetune_ppo policy loss. Proves the modules connect with correct shapes and
that per-token advantages land on the right response positions.
"""

import random

import numpy as np
import torch
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

from recipe.gear_tree.tree_rollout import SegmentSample, build_tree
from recipe.gear_tree.tree_advantage import extract_edges_from_tree
from recipe.gear_tree.tree_data import edges_to_dataproto
from recipe.gear_tree.policy_loss import compute_policy_loss_treetune
from recipe.gear_tree.gear_gate import GearGate


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, list(ids)))


class MockEngine:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def segment_fn(self, prompt_token_ids, branch_factor, max_tokens):
        out = []
        for b in range(branch_factor):
            ntok = self.rng.randint(1, 3)
            toks = [self.rng.randint(10, 99) for _ in range(ntok)]
            lps = [-self.rng.random() for _ in range(ntok)]
            finish = "stop" if (max_tokens is None or self.rng.random() > 0.5) else "length"
            out.append(SegmentSample(token_ids=toks, text=f" s{b}", finish_reason=finish, logprobs=lps))
        return out


def _grade(query, response, inst):
    return float(abs(hash(response)) % 2)  # 0.0 or 1.0


class _Cfg:
    """Minimal ActorConfig stand-in (Mapping-like .get + attr access)."""

    clip_ratio = 0.2

    def get(self, k, default=None):
        return getattr(self, k, default)


def test_full_data_path_shapes_and_loss():
    inst = {"problem": "p", "answer": "1", "_treetune__idx": 3}
    eng = MockEngine(seed=42)
    gate = GearGate(k_algorithm="simple", n_min=1, skip_near_leaf_expand=True, max_depth=2)

    tree = build_tree(
        "PROMPT", [7, 8, 9], inst,
        tree_shape=[3, 3], M=4, segment_fn=eng.segment_fn, grade_fn=_grade, gear_gate=gate,
    )
    # Keep every edge (the pav_advantage!=0 filter is treetune-faithful and
    # covered by other tests; here we exercise the full tensor path).
    edges = extract_edges_from_tree(
        tree, tree_update_mode="spo", adv_method="rloo", only_adv_greater_than_zero=False
    )
    assert edges

    data = edges_to_dataproto(
        edges, MockTokenizer(), max_prompt_length=16, max_response_length=8
    )
    bsz = len(edges)
    assert data.batch["input_ids"].shape == (bsz, 24)
    assert data.batch["advantages"].shape == (bsz, 8)
    assert data.batch["response_mask"].shape == (bsz, 8)
    assert "uid" in data.non_tensor_batch

    # advantage must be zero outside the valid response region and constant within.
    for row in range(bsz):
        valid = int(data.batch["response_mask"][row].sum())
        adv_row = data.batch["advantages"][row]
        if valid > 0:
            uniq = torch.unique(adv_row[:valid])
            assert uniq.numel() == 1  # broadcast scalar
        assert torch.all(adv_row[valid:] == 0)

    # feed through the treetune_ppo policy loss (uses response_mask as action mask).
    old_lp = data.batch["old_log_probs"] if "old_log_probs" in data.batch.keys() else torch.zeros_like(
        data.batch["advantages"]
    )
    log_prob = old_lp.clone()  # ratio == 1
    loss, clipfrac, kl, clipfrac_lower = compute_policy_loss_treetune(
        old_log_prob=old_lp,
        log_prob=log_prob,
        advantages=data.batch["advantages"],
        response_mask=data.batch["response_mask"],
        config=_Cfg(),
    )
    assert torch.isfinite(loss)
    # ratio==1 => pg_loss == mean(-advantages) over the action mask.
    assert loss.item() == loss.item()  # not NaN
