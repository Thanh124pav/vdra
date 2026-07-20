"""PLAN.md §1.2/§1.3: trainer-side logical batching + REAL actor execution.

Covers the sparse-execution contract's trainer/actor mechanics with the
ACTUAL production entry points:

* ``build_logical_update_batch`` — logical batches formed BEFORE tensor
  filtering; per-batch pre-filter ``M_B``/``T_B`` stamped one value per
  batch; rank-major ordering with collective-safe dummy padding; explicit
  ``(None, stats)`` for a fully-zero reservation;
* ``DataParallelPPOActor.update_policy`` — canonical aggregations group by
  ``logical_batch_index``, reuse the stamped denominators, skip all-zero
  logical batches consistently (required test 10 at actor level), and fail
  fast when the logical structure is missing (required test 12).
"""

from __future__ import annotations

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch.distributed as dist  # noqa: E402

try:
    from recipe.gear_tree.tests import _tiny_actor
except ImportError:
    import _tiny_actor

from recipe.gear_tree.tree_data import build_logical_update_batch  # noqa: E402

MINI = 4


def _edge(i: int, adv: float, n_tokens: int = 2, tree: str = "t0") -> dict:
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
        "response_token_ids": [3 + (i % 5)] * n_tokens,
        "actor_shifted_log_probs": [-0.4] * n_tokens,
        "advantage": adv,
        "value": 0.4,
        "reward": 1.0,
        "advantage_is_zero": adv == 0.0,
    }


def _slot(
    i: int, n_tokens: int = 3, tree: str = "t0", active: int | None = None
) -> dict:
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
        "prob_mask_token_count": int(n_tokens if active is None else active),
        "probability_mask_threshold": 0.9,
    }


def _build(slots, dp_size=1, mini=MINI):
    return build_logical_update_batch(
        slots,
        _tiny_actor.Tok(),
        max_prompt_length=6,
        max_response_length=4,
        ppo_mini_batch_size=mini,
        dp_size=dp_size,
        loss_mode="vdra_segment_mean_ppo",
    )


class TestBuilder:
    def test_denominators_are_prefilter_per_logical_batch(self):
        # Batch 0: 2 trainable (2 tokens each) + 2 zero slots (3 tokens each).
        # Batch 1: 4 trainable rows of 1 token.
        slots = [
            _edge(0, 0.5),
            _edge(1, -0.5),
            _slot(0),
            _slot(1),
            _edge(2, 0.3, n_tokens=1),
            _edge(3, -0.3, n_tokens=1),
            _edge(4, 0.2, n_tokens=1),
            _edge(5, -0.2, n_tokens=1),
        ]
        batch, stats = _build(slots)
        assert batch is not None
        assert batch.meta_info["original_logical_segment_count"] == [4.0, 4.0]
        assert batch.meta_info["original_logical_response_token_count"] == [
            2 + 2 + 3 + 3,
            4.0,
        ]
        assert batch.meta_info["logical_batch_count"] == 2
        # Only trainable rows tensorized; zero slots stay in the denominators.
        assert len(batch) == 6
        assert batch.batch["logical_batch_index"].tolist() == [0, 0, 1, 1, 1, 1]
        assert batch.batch["is_dummy"].sum().item() == 0
        assert stats["vdra/tensor_rows"] == 6.0
        assert stats["vdra/all_zero_advantage_logical_batches"] == 0.0

    def test_all_zero_batch_has_no_rows_and_is_counted(self):
        slots = [
            _edge(0, 0.5),
            _edge(1, -0.5),
            _slot(0),
            _slot(1),
            _slot(2),
            _slot(3),
            _slot(4),
            _slot(5),
        ]
        batch, stats = _build(slots)
        assert batch is not None
        assert stats["vdra/all_zero_advantage_logical_batches"] == 1.0
        # Batch 1 contributes NO tensor rows (skipped consistently later).
        assert batch.batch["logical_batch_index"].tolist() == [0, 0]
        # Its denominators are still stamped (position 1 in the lists).
        assert batch.meta_info["original_logical_segment_count"] == [4.0, 4.0]
        assert batch.meta_info["original_logical_response_token_count"][1] == pytest.approx(
            3 * 4
        )

    def test_fully_zero_reservation_returns_none(self):
        """Required trainer behavior: explicit skipped update, not a failure."""
        batch, stats = _build([_slot(i) for i in range(8)])
        assert batch is None
        assert stats["vdra/skipped_zero_gradient_updates"] == 1.0
        assert stats["vdra/all_zero_advantage_logical_batches"] == 2.0

    def test_dummy_padding_gives_equal_rank_shares(self):
        # Batch 0 has 3 trainable rows; dp=2 pads to 4 with one dummy row.
        slots = [_edge(0, 0.5), _edge(1, -0.5), _edge(2, 0.3), _slot(0)]
        batch, stats = _build(slots, dp_size=2)
        assert batch is not None
        assert len(batch) == 4
        assert stats["vdra/dummy_rows"] == 1.0
        dummy_mask = batch.batch["is_dummy"]
        assert dummy_mask.sum().item() == 1
        # Rank-major order: first half = rank 0's rows, second half rank 1's.
        halves = batch.batch["logical_batch_index"].chunk(2)
        assert halves[0].tolist() == halves[1].tolist()
        # The dummy row carries exactly zero advantage.
        adv = batch.batch["advantages"][dummy_mask.bool()]
        assert torch.all(adv == 0.0)

    def test_missing_token_count_fails_fast(self):
        bad = _slot(0)
        bad.pop("response_token_count")
        with pytest.raises(ValueError, match="response_token_count"):
            _build([_edge(0, 0.5), _edge(1, -0.5), bad, _slot(1)])

    def test_non_divisible_reservation_fails_fast(self):
        with pytest.raises(ValueError, match="divisible"):
            _build([_edge(i, 0.5) for i in range(6)])


@pytest.fixture(scope="module")
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        rdv = tmp_path_factory.mktemp("pg") / "rdv"
        dist.init_process_group(
            backend="gloo", init_method=f"file://{rdv}", rank=0, world_size=1
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _finish_batch(batch):
    batch.meta_info["temperature"] = 1.0
    batch.meta_info["force_stored_old_log_probs"] = True
    return batch


@pytest.mark.usefixtures("single_process_group")
class TestActorLogicalExecution:
    def _run(self, slots, aggregation, mini=MINI, micro=2):
        batch, stats = _build(slots)
        assert batch is not None
        actor, model, _ = _tiny_actor.make_actor(
            config=_tiny_actor.make_actor_config(
                strategy="fsdp", mini=mini, micro=micro, aggregation=aggregation
            )
        )
        metrics = actor.update_policy(_finish_batch(batch))
        return metrics, stats

    def test_segment_mean_steps_once_per_nonempty_logical_batch(self):
        slots = [
            _edge(0, 0.5),
            _edge(1, -0.5),
            _slot(0),
            _slot(1),
            _edge(2, 0.3, n_tokens=1),
            _edge(3, -0.3, n_tokens=1),
            _edge(4, 0.2, n_tokens=1),
            _edge(5, -0.2, n_tokens=1),
        ]
        metrics, _ = self._run(slots, "segment_mean")
        assert metrics["actor/num_optimizer_steps"] == [2]
        assert metrics["actor/all_zero_advantage_logical_batches"] == [0]
        assert metrics["actor/used_stored_old_log_probs"] == [1.0]

    def test_all_zero_logical_batch_skips_its_optimizer_step(self):
        """Required test 10 (actor level): the all-zero logical batch takes
        no forward/backward/optimizer.step and is reported, never stepped."""
        slots = [
            _edge(0, 0.5),
            _edge(1, -0.5),
            _slot(0),
            _slot(1),
            *[_slot(2 + i) for i in range(4)],
        ]
        metrics, stats = self._run(slots, "segment_mean")
        assert metrics["actor/num_optimizer_steps"] == [1]
        assert metrics["actor/all_zero_advantage_logical_batches"] == [1]
        assert stats["vdra/all_zero_advantage_logical_batches"] == 1.0

    def test_token_mean_runs_with_stamped_denominators(self):
        slots = [
            _edge(0, 0.5),
            _edge(1, -0.5),
            _slot(0),
            _slot(1),
        ]
        metrics, _ = self._run(slots, "token_mean")
        assert metrics["actor/num_optimizer_steps"] == [1]
        assert all(torch.isfinite(torch.tensor(metrics["actor/pg_loss"])))

    def test_segment_mean_loss_uses_prefilter_m_b(self):
        """End-to-end (L1 + L2 + 0 + 0)/4 through the REAL actor: doubling
        the zero-slot count halves nothing — the denominator M_B counts the
        slots even though they carry no rows."""
        base = [_edge(0, 0.5), _edge(1, -0.5), _slot(0), _slot(1)]
        batch, _ = _build(base)
        actor, _, _ = _tiny_actor.make_actor(
            config=_tiny_actor.make_actor_config(
                strategy="fsdp", mini=MINI, micro=2, aggregation="segment_mean"
            )
        )
        m1 = actor.update_policy(_finish_batch(batch))
        loss_m4 = sum(m1["actor/pg_loss"])

        # Same two trainable rows, M_B = 8 (six zero slots, mini 8).
        wide = [_edge(0, 0.5), _edge(1, -0.5)] + [_slot(i) for i in range(6)]
        batch8, _ = _build(wide, mini=8)
        actor8, _, _ = _tiny_actor.make_actor(
            config=_tiny_actor.make_actor_config(
                strategy="fsdp", mini=8, micro=2, aggregation="segment_mean"
            )
        )
        m2 = actor8.update_policy(_finish_batch(batch8))
        loss_m8 = sum(m2["actor/pg_loss"])
        assert loss_m4 == pytest.approx(2.0 * loss_m8, rel=1e-5)

    def test_canonical_aggregation_without_logical_structure_fails(self):
        """Required test 12: strict fail-fast when the logical denominator
        metadata is missing — no retained-row fallback."""
        actor, _, _ = _tiny_actor.make_actor(
            config=_tiny_actor.make_actor_config(
                strategy="fsdp", mini=128, micro=64, aggregation="segment_mean"
            )
        )
        plain = _tiny_actor.build_batch(_tiny_actor.make_edges([8] * 16))
        with pytest.raises(ValueError, match="logical"):
            actor.update_policy(plain)