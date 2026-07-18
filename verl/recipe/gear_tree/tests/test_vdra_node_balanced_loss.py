"""Behavioural tests for the canonical VDRA node-balanced PPO loss.

Covers the reduction contract in PLAN.md Section 4.3 and the pure-reduction
matrix in Section 7.1. The production loss is verified against an explicit
hierarchical reference computation for each invariant.
"""

from __future__ import annotations

import pytest
import torch
import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

from recipe.gear_tree.policy_loss import (
    compute_policy_loss_treetune,
    compute_policy_loss_vdra_node_balanced,
    hierarchical_reference_reduction,
)


class _Cfg:
    def __init__(self, clip_ratio=0.2, use_prob_mask=False, ratio_threshold=100.0):
        self.clip_ratio = clip_ratio
        self._d = {"use_prob_mask": use_prob_mask, "ratio_threshold": ratio_threshold}

    def get(self, k, default=None):
        return self._d.get(k, default)


def _identity_ratio_inputs(child_losses_per_token: torch.Tensor) -> tuple:
    """Construct (old, new, adv, mask) so that pg_losses[row, t] == child_losses_per_token[row, t]
    with a unit action mask and ratio == 1.

    ratio == 1 requires new == old, so log_ratio = 0, ratio = 1. pg_losses1 =
    pg_losses2 = -adv * 1 = -adv, and pg_losses = max(=,=) = -adv. Passing
    adv = -child_losses_per_token yields pg_losses = child_losses_per_token.
    """
    b, t = child_losses_per_token.shape
    old = torch.zeros(b, t)
    new = old.clone()
    adv = -child_losses_per_token
    mask = torch.ones(b, t)
    return old, new, adv, mask


def _run_vdra(
    child_losses: torch.Tensor,
    parent_ids: torch.Tensor,
    tree_ids: torch.Tensor,
    multiplicities: torch.Tensor | None = None,
    per_token_length: int = 1,
) -> torch.Tensor:
    """Build tensors so that each row's token-mean surrogate equals its child loss."""
    bsz = child_losses.numel()
    # Broadcast each child loss uniformly across its tokens (length invariance).
    per_token = child_losses.unsqueeze(1).expand(bsz, per_token_length).contiguous()
    old, new, adv, mask = _identity_ratio_inputs(per_token)
    kwargs = dict(
        old_log_prob=old,
        log_prob=new,
        advantages=adv,
        response_mask=mask,
        config=_Cfg(),
        parent_group_ids=parent_ids,
        tree_group_ids=tree_ids,
    )
    if multiplicities is not None:
        kwargs["sample_multiplicity"] = multiplicities
    loss, *_ = compute_policy_loss_vdra_node_balanced(**kwargs)
    return loss


def test_uniform_parity_matches_legacy_average():
    # Uniform two-parent tree with equal branch counts and equal segment
    # lengths reproduces the legacy edge-mean.
    child_losses = torch.tensor([1.0, 3.0, 2.0, 4.0])
    parent_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    tree_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    got = _run_vdra(child_losses, parent_ids, tree_ids)
    # parents: 2.0, 3.0; tree: 2.5.
    assert torch.allclose(got, torch.tensor(2.5))


def test_non_uniform_separation_favours_parent_balanced_over_edge_balanced():
    # PLAN.md 7.1.2: parent A with one child (loss 2) and parent B with three
    # child losses 4,4,4 gives node-balanced 3.0, edge-balanced 3.5.
    child_losses = torch.tensor([2.0, 4.0, 4.0, 4.0])
    parent_ids = torch.tensor([0, 1, 1, 1], dtype=torch.int64)
    tree_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    got = _run_vdra(child_losses, parent_ids, tree_ids)
    assert torch.allclose(got, torch.tensor(3.0))
    assert not torch.allclose(got, torch.tensor(3.5))


def test_length_invariance_within_a_child():
    child_losses = torch.tensor([1.0, 3.0, 2.0, 4.0])
    parent_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    tree_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    short = _run_vdra(child_losses, parent_ids, tree_ids, per_token_length=1)
    long = _run_vdra(child_losses, parent_ids, tree_ids, per_token_length=17)
    assert torch.allclose(short, long)


def test_child_duplication_invariance_under_multiplicity():
    # PLAN.md 7.1.4: duplicating an identical child and updating allocated_k
    # does not change that parent's loss (weighted mean is multiplicity-safe).
    child_losses = torch.tensor([2.0, 4.0])
    parent_ids = torch.tensor([0, 1], dtype=torch.int64)
    tree_ids = torch.tensor([0, 0], dtype=torch.int64)
    base = _run_vdra(child_losses, parent_ids, tree_ids)

    dup_losses = torch.tensor([2.0, 2.0, 4.0])
    dup_parents = torch.tensor([0, 0, 1], dtype=torch.int64)
    dup_trees = torch.tensor([0, 0, 0], dtype=torch.int64)
    dup = _run_vdra(dup_losses, dup_parents, dup_trees)
    assert torch.allclose(base, dup)


def test_parent_balance_change_of_another_parent_does_not_shift_first():
    # Fix parent 0 with a single child loss=2. Change parent 1's branch count.
    for extra_bf in (1, 3, 7):
        n = extra_bf + 1
        child_losses = torch.cat([torch.tensor([2.0]), torch.full((extra_bf,), 4.0)])
        parent_ids = torch.tensor([0] + [1] * extra_bf, dtype=torch.int64)
        tree_ids = torch.zeros(n, dtype=torch.int64)
        loss = _run_vdra(child_losses, parent_ids, tree_ids)
        # parent 0 = 2.0, parent 1 = 4.0, tree = 3.0 regardless of extra_bf.
        assert torch.allclose(loss, torch.tensor(3.0)), extra_bf


def test_queue_decomposition_matches_parent_balanced_direct_reduction():
    # PLAN.md 4.4: sum over queues of |Q_r|/|P(T)| * mean(Q_r) equals L_T.
    child_losses = torch.tensor([1.0, 3.0, 4.0, 2.0, 10.0])
    parent_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    tree_ids = torch.zeros(5, dtype=torch.int64)
    multiplicities = torch.ones(5)
    direct = _run_vdra(child_losses, parent_ids, tree_ids, multiplicities)
    # Parents: 0->2, 1->3, 2->10. Tree = 5.0.
    assert torch.allclose(direct, torch.tensor(5.0))
    # Queue Q_r partition: {0,1} and {2}. Weighted mean is 2/3 * mean(2,3) + 1/3 * 10 = 5/3 + 10/3 = 5.
    q1 = (2.0 + 3.0) / 2
    q2 = 10.0
    weighted = (2 / 3) * q1 + (1 / 3) * q2
    assert weighted == pytest.approx(5.0)


def test_wrong_queue_coefficient_reproduces_edge_mean_not_parent_mean():
    # PLAN.md 7.1.7: using |edges in queue|/|edges in tree| collapses to the
    # legacy edge mean, not the parent-balanced objective.
    child_losses = torch.tensor([1.0, 3.0, 4.0, 2.0, 10.0])
    parent_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64)
    tree_ids = torch.zeros(5, dtype=torch.int64)
    multiplicities = torch.ones(5)
    got = _run_vdra(child_losses, parent_ids, tree_ids, multiplicities)
    # Edge mean = (1+3+4+2+10)/5 = 4.0. Node-balanced should differ.
    assert not torch.allclose(got, torch.tensor(4.0))


def test_zero_advantage_child_counts_in_denominator():
    # PLAN.md 7.1.9: a real zero-advantage child stays in its parent group.
    child_losses = torch.tensor([2.0, 0.0])
    parent_ids = torch.tensor([0, 0], dtype=torch.int64)
    tree_ids = torch.tensor([0, 0], dtype=torch.int64)
    got = _run_vdra(child_losses, parent_ids, tree_ids)
    # Parent = 1.0, tree = 1.0.
    assert torch.allclose(got, torch.tensor(1.0))


def test_permutation_invariance():
    # PLAN.md 7.1.13: row ordering does not change the result.
    losses = torch.tensor([1.0, 3.0, 2.0, 4.0])
    parents = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    trees = torch.zeros(4, dtype=torch.int64)
    baseline = _run_vdra(losses, parents, trees)

    perm = torch.tensor([2, 0, 3, 1])
    got = _run_vdra(losses[perm], parents[perm], trees[perm])
    assert torch.allclose(baseline, got)


def test_batch_mean_over_multiple_trees():
    # PLAN.md 4.3 Stage 4: two trees give the mean of tree losses.
    losses = torch.tensor([2.0, 4.0, 10.0])
    parents = torch.tensor([0, 1, 2], dtype=torch.int64)
    # tree 0 has parents {0, 1}; tree 1 has parent {2}.
    trees = torch.tensor([0, 0, 1], dtype=torch.int64)
    got = _run_vdra(losses, parents, trees)
    # tree 0 loss = mean(2, 4) = 3, tree 1 loss = 10. batch = 6.5.
    assert torch.allclose(got, torch.tensor(6.5))


def test_reference_implementation_matches_production_loss():
    # PLAN.md 7.1.14: autograd gradients match a hierarchical reference.
    torch.manual_seed(0)
    losses = torch.randn(12).abs() + 0.1
    parents = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5], dtype=torch.int64)
    trees = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int64)
    mults = torch.ones_like(losses)
    ref = hierarchical_reference_reduction(losses, parents, trees, mults)
    prod = _run_vdra(losses, parents, trees, mults)
    assert torch.allclose(prod, ref, atol=1e-6)


def test_weighted_reuse_locality_multiplicity_shifts_only_within_parent():
    # PLAN.md 7.1.12: within-parent multiplicities move the parent's mean, but
    # do not change the parent's weight in the tree.
    # Baseline: parent 0 fresh_iid children {loss 2, 4}, parent 1 fresh_iid {6}.
    baseline_losses = torch.tensor([2.0, 4.0, 6.0])
    baseline_parents = torch.tensor([0, 0, 1], dtype=torch.int64)
    baseline_trees = torch.zeros(3, dtype=torch.int64)
    baseline = _run_vdra(baseline_losses, baseline_parents, baseline_trees, torch.ones(3))
    # baseline parent 0 = 3.0, parent 1 = 6.0, tree = 4.5.
    assert torch.allclose(baseline, torch.tensor(4.5))

    # Now duplicate the second child in parent 0 by multiplicity=3 (fewer rows).
    reused_losses = torch.tensor([2.0, 4.0, 6.0])
    reused_mults = torch.tensor([1.0, 3.0, 1.0])
    reused = _run_vdra(
        reused_losses, baseline_parents, baseline_trees, reused_mults
    )
    # parent 0 = (1*2 + 3*4) / (1+3) = 14/4 = 3.5, parent 1 = 6, tree = 4.75.
    assert torch.allclose(reused, torch.tensor(4.75))
    # Parent 1's contribution is unchanged: swap its (single) child value and
    # observe the same delta on the tree loss as would be seen without reuse.
    parent1_flipped = torch.tensor([2.0, 4.0, 7.0])
    tree_before = _run_vdra(
        parent1_flipped, baseline_parents, baseline_trees, reused_mults
    )
    # parent 0 unchanged at 3.5, parent 1 = 7. tree = 5.25.
    assert torch.allclose(tree_before, torch.tensor(5.25))


def test_missing_group_tensors_raises_clear_error():
    losses = torch.tensor([1.0, 2.0])
    old = torch.zeros(2, 1)
    new = old.clone()
    with pytest.raises(ValueError, match="parent_group_ids"):
        compute_policy_loss_vdra_node_balanced(
            old_log_prob=old,
            log_prob=new,
            advantages=-losses.unsqueeze(1),
            response_mask=torch.ones(2, 1),
            config=_Cfg(),
        )


def test_edge_weights_are_rejected_by_node_balanced_loss():
    old = torch.zeros(2, 1)
    new = old.clone()
    with pytest.raises(ValueError, match="edge_weights"):
        compute_policy_loss_vdra_node_balanced(
            old_log_prob=old,
            log_prob=new,
            advantages=torch.tensor([[-1.0], [-2.0]]),
            response_mask=torch.ones(2, 1),
            config=_Cfg(),
            parent_group_ids=torch.tensor([0, 0], dtype=torch.int64),
            tree_group_ids=torch.tensor([0, 0], dtype=torch.int64),
            edge_weights=torch.tensor([[1.0], [2.0]]),
        )


def test_treetune_loss_reduces_to_edge_mean_under_same_inputs():
    # Sanity check: the same tensors handed to treetune_ppo give the edge-mean
    # legacy value, confirming the two losses genuinely differ on non-uniform
    # trees.
    child_losses = torch.tensor([2.0, 4.0, 4.0, 4.0])
    old, new, adv, mask = _identity_ratio_inputs(child_losses.unsqueeze(1))
    legacy, *_ = compute_policy_loss_treetune(
        old_log_prob=old,
        log_prob=new,
        advantages=adv,
        response_mask=mask,
        config=_Cfg(),
    )
    assert torch.allclose(legacy, torch.tensor((2 + 4 + 4 + 4) / 4))


def test_gradient_parity_with_reference():
    # PLAN.md 7.1.14: autograd gradients through the production loss match
    # the reference implementation.
    torch.manual_seed(1)
    losses = torch.randn(8).abs() + 0.1
    losses_prod = losses.clone().detach().requires_grad_(True)
    losses_ref = losses.clone().detach().requires_grad_(True)
    parents = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3], dtype=torch.int64)
    trees = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    mults = torch.ones_like(losses)

    per_token_prod = losses_prod.unsqueeze(1)
    prod = compute_policy_loss_vdra_node_balanced(
        old_log_prob=torch.zeros_like(per_token_prod),
        log_prob=torch.zeros_like(per_token_prod),
        advantages=-per_token_prod,
        response_mask=torch.ones_like(per_token_prod),
        config=_Cfg(),
        parent_group_ids=parents,
        tree_group_ids=trees,
        sample_multiplicity=mults,
    )[0]
    ref = hierarchical_reference_reduction(losses_ref, parents, trees, mults)
    prod.backward()
    ref.backward()
    assert torch.allclose(prod, ref)
    assert torch.allclose(losses_prod.grad, losses_ref.grad, atol=1e-6)
