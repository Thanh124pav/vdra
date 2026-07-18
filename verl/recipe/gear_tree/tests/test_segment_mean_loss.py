"""PLAN.md P0.1 / P0.4: segment-average loss (mean|sum) reference tests.

Both reductions share exactly the same rollout / advantage / segment
denominator / replay / batching path. This suite exercises:

  * mean/sum reference match against ``segment_average_reference``;
  * mean-vs-sum non-alias on non-uniform segment lengths;
  * zero-advantage sparse filter preservation;
  * queue-label permutation invariance;
  * parent-group regrouping invariance;
  * row-permutation invariance;
  * mean/sum full-vs-split gradient parity.
"""

from __future__ import annotations

import pytest
import torch
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

from recipe.gear_tree.policy_loss import (
    compute_policy_loss_vdra_segment_mean,
    segment_average_reference,
    _segment_row_losses,
)
from recipe.gear_tree.tree_data import (
    compute_segment_objective_weights,
    validate_segment_objective_weights,
)


class _Cfg:
    def __init__(
        self,
        *,
        clip_ratio: float = 0.2,
        use_prob_mask: bool = False,
        ratio_threshold: float = 1e6,
        segment_token_reduction: str = "mean",
    ) -> None:
        self.clip_ratio = clip_ratio
        self._d = {
            "use_prob_mask": use_prob_mask,
            "ratio_threshold": ratio_threshold,
            "segment_token_reduction": segment_token_reduction,
        }

    def get(self, key: str, default=None):
        return self._d.get(key, default)


def _one_token_batch(per_row_losses: torch.Tensor):
    """Encode L_row = per_row_losses[row] via adv = -per_row_losses, ratio=1."""
    b = per_row_losses.numel()
    old = torch.zeros(b, 1)
    new = torch.zeros_like(old)
    adv = -per_row_losses.unsqueeze(1)
    mask = torch.ones(b, 1)
    return old, new, adv, mask


def _loss(old, new, adv, mask, weights, reduction="mean"):
    return compute_policy_loss_vdra_segment_mean(
        old_log_prob=old,
        log_prob=new,
        advantages=adv,
        response_mask=mask,
        config=_Cfg(segment_token_reduction=reduction),
        segment_objective_weights=weights,
    )[0]


def _edges_two_trees_with_totals():
    """Two trees. Tree T0 has 4 realized segments (2 retained),
    tree T1 has 3 realized segments (3 retained)."""
    return [
        {"tree_id": "T0", "parent_group_id": "T0:p0", "sample_multiplicity": 1,
         "tree_total_segment_count": 4, "queue_flush_id": "q0"},
        {"tree_id": "T0", "parent_group_id": "T0:p1", "sample_multiplicity": 1,
         "tree_total_segment_count": 4, "queue_flush_id": "q0"},
        {"tree_id": "T1", "parent_group_id": "T1:p0", "sample_multiplicity": 1,
         "tree_total_segment_count": 3, "queue_flush_id": "q0"},
        {"tree_id": "T1", "parent_group_id": "T1:p1", "sample_multiplicity": 1,
         "tree_total_segment_count": 3, "queue_flush_id": "q1"},
        {"tree_id": "T1", "parent_group_id": "T1:p1", "sample_multiplicity": 1,
         "tree_total_segment_count": 3, "queue_flush_id": "q1"},
    ]


def test_reference_matches_precomputed_weights_mean():
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(11)
    per_row = torch.randn(len(edges)).abs() + 0.5

    old, new, adv, mask = _one_token_batch(per_row)
    got = _loss(old, new, adv, mask, weights, "mean")
    row_losses = _segment_row_losses(-adv, mask, reduction="mean")
    tree_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    totals = torch.tensor([4, 4, 3, 3, 3], dtype=torch.float32)
    expected = segment_average_reference(row_losses, tree_ids, totals)
    assert torch.allclose(got, expected, atol=1e-6)


def test_reference_matches_precomputed_weights_sum():
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(12)
    per_row = torch.randn(len(edges)).abs() + 0.5

    old, new, adv, mask = _one_token_batch(per_row)
    got = _loss(old, new, adv, mask, weights, "sum")
    row_losses = _segment_row_losses(-adv, mask, reduction="sum")
    tree_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    totals = torch.tensor([4, 4, 3, 3, 3], dtype=torch.float32)
    expected = segment_average_reference(row_losses, tree_ids, totals)
    assert torch.allclose(got, expected, atol=1e-6)


def test_parent_regrouping_does_not_change_mean_loss():
    edges = _edges_two_trees_with_totals()
    weights_a = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(13)
    per_row = torch.randn(len(edges)).abs() + 0.5
    old, new, adv, mask = _one_token_batch(per_row)
    base = _loss(old, new, adv, mask, weights_a, "mean")
    # Change parent group labels arbitrarily.
    edges2 = [dict(e) for e in edges]
    edges2[0]["parent_group_id"] = "T0:regrouped-a"
    edges2[1]["parent_group_id"] = "T0:regrouped-b"
    edges2[3]["parent_group_id"] = "T1:regrouped-c"
    weights_b = torch.tensor(compute_segment_objective_weights(edges2), dtype=torch.float32)
    got = _loss(old, new, adv, mask, weights_b, "mean")
    assert torch.allclose(base, got, atol=1e-6)


def test_queue_label_permutation_does_not_change_loss():
    edges = _edges_two_trees_with_totals()
    weights_a = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(14)
    per_row = torch.randn(len(edges)).abs() + 0.5
    old, new, adv, mask = _one_token_batch(per_row)
    base = _loss(old, new, adv, mask, weights_a, "mean")
    edges2 = [dict(e) for e in edges]
    for e in edges2:
        e["queue_flush_id"] = "Q-permuted-" + str(e.get("queue_flush_id"))
    weights_b = torch.tensor(compute_segment_objective_weights(edges2), dtype=torch.float32)
    got = _loss(old, new, adv, mask, weights_b, "mean")
    assert torch.allclose(base, got, atol=1e-6)


def test_zero_advantage_sparse_filter_preserves_loss():
    # PLAN.md P0.2 acceptance: filtering zero-contribution rows does not
    # change the loss (they contribute zero and the denominator counts them).
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(15)
    per_row = torch.tensor([2.0, 0.0, 0.0, 3.0, 0.0])
    old, new, adv, mask = _one_token_batch(per_row)
    full = _loss(old, new, adv, mask, weights, "mean")

    # Filtered batch: drop the zero-adv rows.
    keep = torch.tensor([0, 3])
    filtered_edges = [edges[i] for i in keep.tolist()]
    # tree_total_segment_count stays unchanged — it is the pre-filter count.
    weights_f = torch.tensor(
        compute_segment_objective_weights(filtered_edges), dtype=torch.float32
    )
    got = _loss(old[keep], new[keep], adv[keep], mask[keep], weights_f, "mean")
    assert torch.allclose(got, full, atol=1e-6)


def test_row_permutation_invariance():
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(16)
    per_row = torch.randn(len(edges)).abs() + 0.5
    old, new, adv, mask = _one_token_batch(per_row)
    base = _loss(old, new, adv, mask, weights, "mean")
    perm = torch.tensor([3, 1, 4, 0, 2])
    got = _loss(old[perm], new[perm], adv[perm], mask[perm], weights[perm], "mean")
    assert torch.allclose(base, got, atol=1e-6)


def _two_token_batch(per_token_losses: torch.Tensor, active_mask: torch.Tensor):
    """Direct token losses via adv=-losses on active tokens."""
    b, t = per_token_losses.shape
    old = torch.zeros(b, t)
    new = torch.zeros_like(old)
    adv = -per_token_losses
    mask = active_mask.to(dtype=torch.float32)
    return old, new, adv, mask


def test_mean_and_sum_disagree_on_non_uniform_lengths():
    # PLAN.md P0.4 acceptance: mean and sum are not aliases when segments
    # have different active-token counts.
    per_token_losses = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],  # 4 active tokens each with loss=1 -> mean=1, sum=4
        [1.0, 0.0, 0.0, 0.0],  # 1 active token with loss=1 -> mean=1, sum=1
    ])
    active_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 0, 0, 0],
    ])
    edges = [
        {"tree_id": "T0", "parent_group_id": "T0:p", "sample_multiplicity": 1,
         "tree_total_segment_count": 2, "queue_flush_id": "q"},
        {"tree_id": "T0", "parent_group_id": "T0:p", "sample_multiplicity": 1,
         "tree_total_segment_count": 2, "queue_flush_id": "q"},
    ]
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    old, new, adv, mask = _two_token_batch(per_token_losses, active_mask)
    mean_loss = _loss(old, new, adv, mask, weights, "mean")
    sum_loss = _loss(old, new, adv, mask, weights, "sum")
    assert not torch.isclose(mean_loss, sum_loss, atol=1e-3)


def test_mean_duplicating_active_tokens_preserves_row_loss():
    # PLAN.md P0.4 mode-specific test: duplicating every active token with
    # the same token loss leaves L_row (mean) unchanged.
    per_token = torch.tensor([[2.0, 2.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 0, 0]])
    row_single = _segment_row_losses(per_token, mask.float(), reduction="mean")
    per_token_dup = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    mask_dup = torch.tensor([[1, 1, 1, 1]])
    row_dup = _segment_row_losses(per_token_dup, mask_dup.float(), reduction="mean")
    assert torch.allclose(row_single, row_dup)


def test_sum_duplicating_active_tokens_doubles_row_loss():
    per_token = torch.tensor([[2.0, 2.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 0, 0]])
    row_single = _segment_row_losses(per_token, mask.float(), reduction="sum")
    per_token_dup = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    mask_dup = torch.tensor([[1, 1, 1, 1]])
    row_dup = _segment_row_losses(per_token_dup, mask_dup.float(), reduction="sum")
    assert torch.allclose(row_dup, 2.0 * row_single)


def test_invalid_reduction_string_fails_clearly():
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(17)
    per_row = torch.ones(len(edges))
    old, new, adv, mask = _one_token_batch(per_row)
    with pytest.raises(ValueError, match="segment_token_reduction"):
        compute_policy_loss_vdra_segment_mean(
            old_log_prob=old,
            log_prob=new,
            advantages=adv,
            response_mask=mask,
            config=_Cfg(segment_token_reduction="banana"),
            segment_objective_weights=weights,
        )


def test_edge_weights_are_rejected_on_segment_mean_path():
    # PLAN.md P0.1: the main path must not silently combine edge_weights
    # with the segment objective.
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(18)
    per_row = torch.ones(len(edges))
    old, new, adv, mask = _one_token_batch(per_row)
    with pytest.raises(ValueError, match="edge_weights"):
        compute_policy_loss_vdra_segment_mean(
            old_log_prob=old,
            log_prob=new,
            advantages=adv,
            response_mask=mask,
            config=_Cfg(segment_token_reduction="mean"),
            segment_objective_weights=weights,
            edge_weights=torch.ones_like(mask),
        )


def test_full_batch_equals_sum_of_microbatch_splits_mean():
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(19)
    per_row = torch.randn(len(edges)).abs() + 0.5
    old, new, adv, mask = _one_token_batch(per_row)
    full = _loss(old, new, adv, mask, weights, "mean")
    perm = torch.tensor([2, 0, 4, 1, 3])
    a = perm[:2]
    b = perm[2:]
    la = _loss(old[a], new[a], adv[a], mask[a], weights[a], "mean")
    lb = _loss(old[b], new[b], adv[b], mask[b], weights[b], "mean")
    assert torch.allclose(la + lb, full, atol=1e-6)


def test_full_batch_equals_sum_of_microbatch_splits_sum():
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(20)
    per_row = torch.randn(len(edges)).abs() + 0.5
    old, new, adv, mask = _one_token_batch(per_row)
    full = _loss(old, new, adv, mask, weights, "sum")
    perm = torch.tensor([2, 0, 4, 1, 3])
    a = perm[:2]
    b = perm[2:]
    la = _loss(old[a], new[a], adv[a], mask[a], weights[a], "sum")
    lb = _loss(old[b], new[b], adv[b], mask[b], weights[b], "sum")
    assert torch.allclose(la + lb, full, atol=1e-6)


def test_gradient_parity_full_vs_split_mean():
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(21)
    per_row_full = (torch.randn(len(edges)).abs() + 0.5).requires_grad_(True)
    per_row_split = per_row_full.detach().clone().requires_grad_(True)
    b = len(edges)
    old = torch.zeros(b, 1)
    new = torch.zeros_like(old)
    mask = torch.ones(b, 1)
    adv_full = -per_row_full.unsqueeze(1)
    _loss(old, new, adv_full, mask, weights, "mean").backward()

    a_idx = torch.tensor([0, 2, 4])
    b_idx = torch.tensor([1, 3])
    adv_split = -per_row_split.unsqueeze(1)
    la = _loss(old[a_idx], new[a_idx], adv_split[a_idx], mask[a_idx], weights[a_idx], "mean")
    lb = _loss(old[b_idx], new[b_idx], adv_split[b_idx], mask[b_idx], weights[b_idx], "mean")
    (la + lb).backward()
    assert torch.allclose(per_row_full.grad, per_row_split.grad, atol=1e-6)


def test_gradient_parity_full_vs_split_sum():
    edges = _edges_two_trees_with_totals()
    weights = torch.tensor(compute_segment_objective_weights(edges), dtype=torch.float32)
    torch.manual_seed(22)
    per_row_full = (torch.randn(len(edges)).abs() + 0.5).requires_grad_(True)
    per_row_split = per_row_full.detach().clone().requires_grad_(True)
    b = len(edges)
    old = torch.zeros(b, 1)
    new = torch.zeros_like(old)
    mask = torch.ones(b, 1)
    adv_full = -per_row_full.unsqueeze(1)
    _loss(old, new, adv_full, mask, weights, "sum").backward()

    a_idx = torch.tensor([0, 2, 4])
    b_idx = torch.tensor([1, 3])
    adv_split = -per_row_split.unsqueeze(1)
    la = _loss(old[a_idx], new[a_idx], adv_split[a_idx], mask[a_idx], weights[a_idx], "sum")
    lb = _loss(old[b_idx], new[b_idx], adv_split[b_idx], mask[b_idx], weights[b_idx], "sum")
    (la + lb).backward()
    assert torch.allclose(per_row_full.grad, per_row_split.grad, atol=1e-6)


def test_segment_weights_normalize_and_sum_to_leq_one():
    edges = _edges_two_trees_with_totals()
    w = compute_segment_objective_weights(edges)
    # PLAN.md P0.4: batch weight sum <= 1 (equality when every realized
    # segment is retained).
    assert 0.0 < sum(w) <= 1.0 + 1e-9
    metrics = validate_segment_objective_weights(edges, w)
    assert metrics["vdra/segment_weight_tree_count"] == 2


def test_zero_active_row_is_finite_zero():
    # Mean over an empty mask must be a finite differentiable zero.
    per_token = torch.tensor([[0.0, 0.0]])
    mask = torch.zeros(1, 2)
    row = _segment_row_losses(per_token, mask, reduction="mean")
    assert torch.equal(row, torch.zeros_like(row))
    row_sum = _segment_row_losses(per_token, mask, reduction="sum")
    assert torch.equal(row_sum, torch.zeros_like(row_sum))
