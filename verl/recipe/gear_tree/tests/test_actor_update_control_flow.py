"""PLAN.md P0.K: real DataParallelPPOActor.update_policy control flow.

Exercises the ACTUAL production actor entry point (no rewritten mirror):
512 canonical edges tensorized by the real ``edges_to_dataproto`` are fed to
``DataParallelPPOActor.update_policy`` with a minimal real model and a real
``torch.optim.SGD``. The canonical 512/128 shape must perform exactly four
real ``_optimizer_step()`` calls, report them via
``actor/num_optimizer_steps``, and observe stored-old-log-prob use via
``actor/used_stored_old_log_probs``.

The tiny model / actor / batch builders are shared with the FSDP2 parity
harness via ``_tiny_actor.py``; the edge data and every assertion here are
unchanged from the original extraction source.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch.distributed as dist  # noqa: E402

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _tiny_actor
except ImportError:  # flat rootdir-relative import
    import _tiny_actor

build_batch = _tiny_actor.build_batch
make_actor = _tiny_actor.make_actor
TinyLM = _tiny_actor.TinyLM

N_EDGES = 512
MINI = 128
MICRO = 64


@pytest.fixture(scope="module")
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        rdv = tmp_path_factory.mktemp("pg") / "rdv"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{rdv}",
            rank=0,
            world_size=1,
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _edges(n: int = N_EDGES, advantage: float = 1.0) -> list[dict]:
    return [
        {
            "edge_id": f"t{i // 8}/e{i}",
            "tree_id": f"t{i // 8}",
            "parent_group_id": f"t{i // 8}/pg",
            "child_segment_id": f"t{i // 8}/e{i}",
            "question_id": f"q{i // 32}",
            "allocated_k": 8,
            "sample_multiplicity": 1,
            "tree_total_segment_count": 8,
            "queue_flush_id": "0",
            "queue_released_segment_count": 8,
            "query_token_ids": [1, 2 + (i % 5)],
            "response_token_ids": [3 + (i % 7), 4, 5 + (i % 3)],
            "actor_shifted_log_probs": [-0.5, -0.4, -0.6],
            "advantage": advantage,
            "value": 0.4,
            "reward": 1.0,
        }
        for i in range(n)
    ]


def _build_batch(n: int = N_EDGES, advantage: float = 1.0):
    return build_batch(_edges(n=n, advantage=advantage))


def _make_actor():
    # The tree-balanced ablation path keeps verl's fixed-size mini-batch
    # split, which is exactly the 512/128 -> 4-step control flow this file
    # pins. The canonical aggregations' logical-batch grouping is covered by
    # test_logical_update_batch.py / the FSDP2 parity harness.
    return make_actor(
        config=_tiny_actor.make_actor_config(
            strategy="fsdp",
            mini=MINI,
            micro=MICRO,
            reduction="mean",
            aggregation="tree_balanced_segment_mean",
        )
    )


@pytest.mark.usefixtures("single_process_group")
class TestUpdatePolicyControlFlow:
    def test_512_over_128_performs_four_real_optimizer_steps(self):
        actor, model, _ = _make_actor()

        real_step = actor._optimizer_step
        step_calls: list[int] = []

        def _counting_step():
            step_calls.append(1)
            return real_step()

        actor._optimizer_step = _counting_step
        params_before = [p.detach().clone() for p in model.parameters()]

        metrics = actor.update_policy(_build_batch())

        assert len(step_calls) == 4
        assert metrics["actor/num_optimizer_steps"] == [4]
        # PLAN.md P0.J: the actor OBSERVES stored-old-log-prob use.
        assert metrics["actor/used_stored_old_log_probs"] == [1.0]
        # The four steps were real: parameters moved.
        changed = any(
            not torch.allclose(before, after)
            for before, after in zip(
                params_before, [p.detach() for p in model.parameters()]
            )
        )
        assert changed
        # One grad-norm entry per optimizer step.
        assert len(metrics["actor/grad_norm"]) == 4
        # 2 microbatches (64) per 128-row mini-batch -> 8 loss entries.
        assert len(metrics["actor/pg_loss"]) == 8

    def test_256_over_128_performs_two_steps(self):
        actor, _, _ = _make_actor()
        metrics = actor.update_policy(_build_batch(n=256))
        assert metrics["actor/num_optimizer_steps"] == [2]

    def test_canonical_batch_has_no_objective_weight_tensors(self):
        batch = _build_batch()
        assert "objective_weights" not in batch.batch
        assert "segment_objective_weights" not in batch.batch
        assert "old_log_probs" in batch.batch
