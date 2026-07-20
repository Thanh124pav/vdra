"""PLAN.md §1.2 required tests 9 + 11: REAL 2-rank FSDP2 sparse dispatch.

Two guarantees that only a real multi-rank run can establish for the
canonical ``segment_mean`` logical-batch dispatch:

* required test 9 — when a logical batch's trainable rows are fewer than a
  multiple of the DP size, ``build_logical_update_batch`` inserts
  collective-safe DUMMY rows so every rank enters the same number of
  forward/backward collectives; a real 2-process ``fully_shard`` run must
  COMPLETE (no hang) and still reproduce the single-rank gradient;
* required test 11 — a logical batch whose slots are ALL zero-advantage
  carries no trainable rows on any rank, so every rank skips that optimizer
  step consistently and NO parameter drifts.

Runs two real gloo/CPU processes with the actual ``apply_fsdp2`` +
``DataParallelPPOActor`` (FSDP1 is CPU-impossible under torch 2.11).
"""

from __future__ import annotations

import os

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat import when mp.spawn re-imports this module
    import _test_shims

_test_shims.install()

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

try:
    from recipe.gear_tree.tests import _tiny_actor
except ImportError:
    import _tiny_actor

WORLD = 2
MINI = 4
MICRO = 2
LR = 0.05
ATOL = 5e-6


def _edge(i: int, adv: float, tree: str = "t0") -> dict:
    return {
        "edge_id": f"{tree}/e{i}",
        "tree_id": tree,
        "parent_group_id": f"{tree}/pg",
        "child_segment_id": f"{tree}/e{i}",
        "question_id": f"q{i % 3}",
        "allocated_k": 4,
        "sample_multiplicity": 1,
        "tree_total_segment_count": 4,
        "queue_flush_id": "0",
        "queue_released_segment_count": 4,
        "query_token_ids": [1, 2],
        "response_token_ids": [3 + (i % 5), 4 + (i % 3)],
        "actor_shifted_log_probs": [-0.3, -0.4],
        "advantage": adv,
        "value": 0.4,
        "reward": 1.0,
        "advantage_is_zero": adv == 0.0,
    }


def _slot(i: int, n_tokens: int = 2, tree: str = "t0") -> dict:
    return {
        "edge_id": f"{tree}/z{i}",
        "tree_id": tree,
        "parent_group_id": f"{tree}/pg",
        "child_segment_id": f"{tree}/z{i}",
        "question_id": f"q{i % 3}",
        "allocated_k": 4,
        "sample_multiplicity": 1,
        "advantage": 0.0,
        "advantage_is_zero": True,
        "trainable_edge_id": None,
        "response_token_count": int(n_tokens),
        # PLAN.md §3/§4: stamped at extraction, never recomputed later.
        "prob_mask_token_count": int(n_tokens),
        "probability_mask_threshold": 0.9,
    }


# Reservation of 8 logical slots -> two logical batches of MINI=4:
#   batch 0: 3 trainable + 1 zero  -> with dp=2, 3 rows padded to 4 (1 dummy)
#   batch 1: all 4 zero            -> no trainable rows on any rank (skip)
_SLOTS = [
    _edge(0, 0.7),
    _edge(1, -0.5),
    _edge(2, 0.3),
    _slot(0),
    _slot(1),
    _slot(2),
    _slot(3),
    _slot(4),
]


def _build(dp_size: int):
    from recipe.gear_tree.tree_data import build_logical_update_batch

    batch, stats = build_logical_update_batch(
        _SLOTS,
        _tiny_actor.Tok(),
        max_prompt_length=_tiny_actor.MAX_PROMPT,
        max_response_length=_tiny_actor.MAX_RESPONSE,
        ppo_mini_batch_size=MINI,
        dp_size=dp_size,
        loss_mode="vdra_segment_mean_ppo",
    )
    assert batch is not None
    batch.meta_info["temperature"] = 1.0
    batch.meta_info["force_stored_old_log_probs"] = True
    return batch, stats


def _full_params(model) -> dict:
    from torch.distributed.tensor import DTensor

    out = {}
    for n, p in model.named_parameters():
        t = p.detach()
        out[n] = (t.full_tensor() if isinstance(t, DTensor) else t).clone()
    return out


def _worker(rank: int, world: int, rdv: str, out_dir: str):
    _test_shims.install()
    dist.init_process_group(
        backend="gloo", init_method=f"file://{rdv}", rank=rank, world_size=world
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from verl.utils.fsdp_utils import apply_fsdp2

        mesh = init_device_mesh("cpu", (world,))
        model = _tiny_actor.TinyLM()
        apply_fsdp2(model, {"mesh": mesh}, {})
        actor, model, _ = _tiny_actor.make_actor(
            model=model,
            config=_tiny_actor.make_actor_config(
                strategy="fsdp2", mini=MINI, micro=MICRO, aggregation="segment_mean",
                grad_clip=1e9,
            ),
            lr=LR,
        )
        batch, stats = _build(world)
        local = batch.chunk(world)[rank]
        p0 = _full_params(model)
        metrics = actor.update_policy(local)
        p1 = _full_params(model)
        if rank == 0:
            torch.save(
                {
                    "stats": stats,
                    "num_steps": metrics["actor/num_optimizer_steps"],
                    "all_zero": metrics["actor/all_zero_advantage_logical_batches"],
                    "zero_active": metrics["actor/zero_active_token_logical_batches"],
                    "p0": p0,
                    "p1": p1,
                },
                os.path.join(out_dir, "dist.pt"),
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _ref_worker(rank: int, rdv: str, out_dir: str):
    _test_shims.install()
    dist.init_process_group(
        backend="gloo", init_method=f"file://{rdv}", rank=0, world_size=1
    )
    try:
        actor, model, _ = _tiny_actor.make_actor(
            config=_tiny_actor.make_actor_config(
                strategy="fsdp", mini=MINI, micro=MICRO, aggregation="segment_mean",
                grad_clip=1e9,
            ),
            lr=LR,
        )
        batch, _ = _build(1)
        p0 = _full_params(model)
        actor.update_policy(batch)
        p1 = _full_params(model)
        torch.save(
            {"delta": {n: p1[n] - p0[n] for n in p0}},
            os.path.join(out_dir, "ref.pt"),
        )
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo backend unavailable")
    out = tmp_path_factory.mktemp("h1disp")
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        mp.spawn(_worker, args=(WORLD, str(out / "rdv"), str(out)), nprocs=WORLD, join=True)
        mp.spawn(_ref_worker, args=(str(out / "refrdv"), str(out)), nprocs=1, join=True)
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return (
        torch.load(str(out / "dist.pt"), weights_only=False),
        torch.load(str(out / "ref.pt"), weights_only=False),
    )


def test_dummy_padding_does_not_hang_and_matches_single_rank(measured):
    """Required test 9: the 2-rank run with a dummy-padded logical batch
    completes (reaching this assertion proves no collective hung) and its
    4-step parameter delta matches the single-rank reference."""
    dist_data, ref = measured
    # A dummy row WAS needed (3 trainable rows in batch 0, dp=2 -> pad to 4).
    assert dist_data["stats"]["vdra/dummy_rows"] == 1.0
    delta_dist = {n: dist_data["p1"][n] - dist_data["p0"][n] for n in dist_data["p0"]}
    for n, want in ref["delta"].items():
        assert torch.allclose(delta_dist[n], want, atol=ATOL), n


def test_all_zero_logical_batch_is_skipped_with_no_drift(measured):
    """Required test 11: batch 1 (all-zero) takes no optimizer step; only the
    single non-empty batch steps. Parameters still move (from batch 0) but
    the all-zero batch contributes nothing — it is counted, never stepped."""
    dist_data, _ = measured
    assert dist_data["all_zero"] == [1]
    assert dist_data["num_steps"] == [1]
    assert dist_data["stats"]["vdra/all_zero_advantage_logical_batches"] == 1.0
