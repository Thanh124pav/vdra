"""REAL two-process distributed gradient parity (batch-slot ablation).

This exercises the preserved verl FSDP/DDP data-path contract on the labeled
``batch_slot_mean_ablation`` loss (``L_B = sum_s L_s / N_B``), whose
distributed scaling behavior is already verified and must NOT change during
the medium stage. Real distributed parity for the CANONICAL tree-segment-mean
weights ``w_s = 1/(N_T * N_seg(T))`` is a Hard-stage (H1) verification item.

Production dispatch semantics being tested:

* the 128-slot optimizer batch is sharded across DP ranks (disjoint rows);
* each rank computes the ablation loss with its LOCAL slot count as
  ``original_optimizer_batch_slot_count`` (dp_actor passes ``len(mini_batch)``);
* the reducer AVERAGES gradients across ranks (DDP semantics).

Parity condition: average-of-rank-gradients == the single-process 128-slot
reference gradient, for both ``segment_token_reduction=mean`` and ``sum``,
with uneven token lengths. This spawns two real ``torch.distributed``
processes over the gloo backend and wraps the model in actual
``DistributedDataParallel`` — no single-process algebra emulation.
"""

from __future__ import annotations

import os

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat import when mp.spawn re-imports this module
    import _test_shims

# mp.spawn children re-import this module (to unpickle ``_worker``) without
# running conftest.py, so the transformers shim must install here too.
_test_shims.install()

torch = pytest.importorskip("torch")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

N_ROWS = 128
N_TOKENS = 6
N_FEATURES = 5
WORLD_SIZE = 2


def _make_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(N_FEATURES, 1, bias=True)


def _make_batch():
    """One fixed 128-row batch with uneven token lengths."""
    g = torch.Generator().manual_seed(42)
    features = torch.randn(N_ROWS, N_TOKENS, N_FEATURES, generator=g)
    old_log_prob = -0.5 + 0.1 * torch.randn(N_ROWS, N_TOKENS, generator=g)
    advantages = torch.randn(N_ROWS, N_TOKENS, generator=g)
    response_mask = torch.zeros(N_ROWS, N_TOKENS)
    lengths = torch.randint(1, N_TOKENS + 1, (N_ROWS,), generator=g)
    for i, k in enumerate(lengths.tolist()):
        response_mask[i, :k] = 1.0
    return features, old_log_prob, advantages, response_mask


def _loss_config(reduction: str):
    from verl.workers.config.actor import ActorConfig, PolicyLossConfig

    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=32,
        policy_loss=PolicyLossConfig(
            loss_mode="vdra_segment_mean_ppo",
            segment_token_reduction=reduction,
            use_prob_mask=False,
            batch_slot_mean_ablation=True,
        ),
    )


def _compute_loss(model, rows, reduction: str):
    from recipe.gear_tree.policy_loss import (
        compute_policy_loss_vdra_segment_mean,
    )

    features, old_log_prob, advantages, response_mask = rows
    log_prob = model(features).squeeze(-1)
    loss, *_ = compute_policy_loss_vdra_segment_mean(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_loss_config(reduction),
        original_optimizer_batch_slot_count=features.shape[0],
    )
    return loss


def _reference_grads(reduction: str):
    """Single-process 128-slot reference (the canonical N_B=128 mean)."""
    model = _make_model()
    loss = _compute_loss(model, _make_batch(), reduction)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def _worker(rank: int, world_size: int, rdv_file: str, reduction: str):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rdv_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _make_model()
        ddp_model = DDP(model)

        features, old_log_prob, advantages, response_mask = _make_batch()
        shard = slice(rank * (N_ROWS // world_size), (rank + 1) * (N_ROWS // world_size))
        local_rows = (
            features[shard],
            old_log_prob[shard],
            advantages[shard],
            response_mask[shard],
        )

        # Production denominator: the LOCAL slot count (dp_actor passes
        # len(mini_batch) on each rank).
        loss = _compute_loss(ddp_model, local_rows, reduction)
        loss.backward()  # DDP averages gradients across ranks here.

        reference = _reference_grads(reduction)
        for got, want in zip(ddp_model.module.parameters(), reference):
            assert torch.allclose(got.grad, want, atol=1e-6), (
                f"rank {rank} ({reduction}): distributed grad "
                f"{got.grad.flatten()[:3]} != single-rank reference "
                f"{want.flatten()[:3]}"
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed gloo backend unavailable",
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_two_process_ddp_matches_single_rank_reference(tmp_path, reduction):
    """Real two-process gloo/DDP run must reproduce the single-rank
    128-slot gradient exactly (PLAN.md P0.I)."""
    rdv = tmp_path / f"rdv-{reduction}"
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, str(rdv), reduction),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed gloo backend unavailable",
)
def test_shards_are_disjoint_and_cover_the_batch():
    """The parity test's sharding mirrors production dispatch: disjoint
    halves covering all 128 rows."""
    shard0 = slice(0, N_ROWS // WORLD_SIZE)
    shard1 = slice(N_ROWS // WORLD_SIZE, N_ROWS)
    rows0 = set(range(*shard0.indices(N_ROWS)))
    rows1 = set(range(*shard1.indices(N_ROWS)))
    assert rows0.isdisjoint(rows1)
    assert rows0 | rows1 == set(range(N_ROWS))


def test_average_reducer_with_global_denominator_is_the_trap():
    """Regression documentation (single process, no spawn): passing the
    GLOBAL 128 as N_B on each rank while the reducer averages would halve
    the gradient — production must pass the LOCAL slot count instead."""
    model = _make_model()
    rows = _make_batch()
    full_loss = _compute_loss(model, rows, "mean")

    half = slice(0, N_ROWS // 2)
    local_rows = tuple(t[half] for t in rows)
    from recipe.gear_tree.policy_loss import (
        compute_policy_loss_vdra_segment_mean,
    )

    features, old_log_prob, advantages, response_mask = local_rows
    log_prob = model(features).squeeze(-1)
    loss_global_nb, *_ = compute_policy_loss_vdra_segment_mean(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=_loss_config("mean"),
        original_optimizer_batch_slot_count=N_ROWS,  # WRONG under averaging
    )
    loss_local_nb = _compute_loss(model, local_rows, "mean")
    # With the global denominator the local loss is half the local-N_B loss:
    assert torch.allclose(loss_global_nb * 2, loss_local_nb, atol=1e-6)
    del full_loss
