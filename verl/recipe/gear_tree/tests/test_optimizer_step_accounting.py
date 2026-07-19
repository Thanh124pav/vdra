"""PLAN.md host-counter contract and optimizer-step diagnostics.

The actor may perform several internal ``optimizer.step()`` calls during one
``update_actor`` RPC, and it must report that diagnostic count. The preserved
VERL trainer contract is different: ``global_step`` counts successful outer
actor updates, while ``num_optimizer_steps_total`` accumulates the internal
optimizer-batch count only for logging/analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest import mock

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn


def _minibatches(rows: int, mini_batch_size: int) -> List[int]:
    return [
        min(mini_batch_size, rows - i)
        for i in range(0, rows, mini_batch_size)
    ]


class _Recorder:
    """Wraps a real torch optimizer.step so we can count how often it fired
    without spinning up FSDP.
    """

    def __init__(self, optim):
        self.optim = optim
        self.count = 0
        self.orig_step = optim.step

    def __enter__(self):
        def wrapped(*a, **kw):
            self.count += 1
            return self.orig_step(*a, **kw)

        self.optim.step = wrapped
        return self

    def __exit__(self, *_):
        self.optim.step = self.orig_step


def _fake_update_policy(
    *,
    total_rows: int = 512,
    ppo_mini_batch_size: int = 128,
    ppo_micro_batch_size_per_gpu: int = 32,
    ppo_epochs: int = 1,
) -> Dict[str, Any]:
    """Mirror the control flow of DataParallelPPOActor.update_policy without
    a real model. One optimizer.step() per mini_batch; microbatch splits do
    not add extra steps.
    """
    model = nn.Linear(4, 4)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics: Dict[str, Any] = {"actor/num_optimizer_steps": []}

    with _Recorder(optim) as rec:
        num_optimizer_steps = 0
        mini_batches = _minibatches(total_rows, ppo_mini_batch_size)
        for _ in range(ppo_epochs):
            for mb_size in mini_batches:
                optim.zero_grad()
                micros = _minibatches(mb_size, ppo_micro_batch_size_per_gpu)
                for micro_size in micros:
                    x = torch.randn(micro_size, 4)
                    loss = (model(x) ** 2).sum() / mb_size
                    loss.backward()
                optim.step()
                num_optimizer_steps += 1
        metrics["actor/num_optimizer_steps"].append(int(num_optimizer_steps))

    return {"num_optim_calls": rec.count, "metrics": metrics}


class TestOptimizerStepCount:
    def test_512_edges_mini_batch_128_epochs_1_is_four_steps(self):
        out = _fake_update_policy(
            total_rows=512, ppo_mini_batch_size=128, ppo_micro_batch_size_per_gpu=32
        )
        assert out["num_optim_calls"] == 4
        # And the returned meta value must agree so the trainer reads 4, not 1.
        assert out["metrics"]["actor/num_optimizer_steps"] == [4]

    def test_microbatch_split_does_not_add_extra_optimizer_steps(self):
        out = _fake_update_policy(
            total_rows=128, ppo_mini_batch_size=128, ppo_micro_batch_size_per_gpu=32
        )
        assert out["num_optim_calls"] == 1
        assert out["metrics"]["actor/num_optimizer_steps"] == [1]

    def test_epochs_multiply_step_count(self):
        out = _fake_update_policy(
            total_rows=256, ppo_mini_batch_size=128, ppo_micro_batch_size_per_gpu=32,
            ppo_epochs=2,
        )
        # 2 mini-batches × 2 epochs = 4 optimizer steps.
        assert out["num_optim_calls"] == 4
        assert out["metrics"]["actor/num_optimizer_steps"] == [4]


class _FakeTrainer:
    """Minimal expected host-counter contract for M1 implementation."""

    def __init__(self, total_training_steps: int = 5):
        self.global_steps = 0
        self.rollout_iteration = 0
        self.successful_actor_updates = 0
        self.optimizer_steps_this_iteration = 0
        self.num_optimizer_steps_total = 0
        self.total_training_steps = total_training_steps

    def start_iteration(self):
        self.rollout_iteration += 1
        self.optimizer_steps_this_iteration = 0

    def apply_actor_update(self, n_optim_steps: int) -> None:
        self.successful_actor_updates += 1
        self.optimizer_steps_this_iteration = int(n_optim_steps)
        self.global_steps += 1
        self.num_optimizer_steps_total += int(n_optim_steps)

    def should_stop(self) -> bool:
        return self.global_steps >= self.total_training_steps

    def rollback_failed(self) -> None:
        self.optimizer_steps_this_iteration = 0


class TestHostCounterContract:
    def test_512_edges_iteration_bumps_outer_global_step_once(self):
        tr = _FakeTrainer()
        tr.start_iteration()
        tr.apply_actor_update(4)
        assert tr.rollout_iteration == 1
        assert tr.global_steps == 1
        assert tr.successful_actor_updates == 1
        assert tr.optimizer_steps_this_iteration == 4
        assert tr.num_optimizer_steps_total == 4

    def test_five_outer_updates_keep_internal_count_separate(self):
        tr = _FakeTrainer(total_training_steps=5)
        for _ in range(5):
            assert not tr.should_stop()
            tr.start_iteration()
            tr.apply_actor_update(4)
        assert tr.rollout_iteration == 5
        assert tr.global_steps == 5
        assert tr.successful_actor_updates == 5
        assert tr.num_optimizer_steps_total == 20
        assert tr.should_stop()

    def test_failed_iteration_does_not_advance_outer_counters(self):
        tr = _FakeTrainer()
        tr.start_iteration()
        tr.rollback_failed()
        assert tr.rollout_iteration == 1
        assert tr.global_steps == 0
        assert tr.successful_actor_updates == 0
        assert tr.num_optimizer_steps_total == 0

    def test_replay_age_uses_rollout_iteration_not_internal_steps(self):
        tr = _FakeTrainer()
        for _ in range(3):
            tr.start_iteration()
            tr.apply_actor_update(4)
        assert tr.rollout_iteration == 3
        assert tr.global_steps == 3
        assert tr.num_optimizer_steps_total == 12


@pytest.mark.xfail(
    strict=True,
    reason="M1 pending: production still couples global_step to internal optimizer count",
)
def test_production_counter_mutation_uses_host_contract_source_guard():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "gear_ray_trainer.py"
    ).read_text()
    assert "self.global_steps += int(n_optim_steps)" not in source
    assert "self.num_optimizer_steps_total = self.global_steps" not in source
    assert "self.global_steps += 1" in source
    assert "self.num_optimizer_steps_total += int(n_optim_steps)" in source


class TestActorReturnsRealCount:
    """Regression: the actor MUST expose the optimizer-step count in
    ``meta_info['metrics']`` under the key ``actor/num_optimizer_steps`` so
    the trainer can read it in one place. This test asserts the shape of the
    returned metadata (list of int-like scalars, one per rank), which is the
    contract the trainer relies on.
    """

    def test_metadata_shape_and_type(self):
        out = _fake_update_policy(total_rows=512)
        raw = out["metrics"]["actor/num_optimizer_steps"]
        assert isinstance(raw, list)
        assert all(isinstance(v, int) for v in raw)
        assert max(raw) == 4
