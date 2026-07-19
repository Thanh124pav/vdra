"""PLAN.md P0.K: real DataParallelPPOActor.update_policy control flow.

Exercises the ACTUAL production actor entry point (no rewritten mirror):
512 canonical edges tensorized by the real ``edges_to_dataproto`` are fed to
``DataParallelPPOActor.update_policy`` with a minimal real model and a real
``torch.optim.SGD``. The canonical 512/128 shape must perform exactly four
real ``_optimizer_step()`` calls, report them via
``actor/num_optimizer_steps``, and observe stored-old-log-prob use via
``actor/used_stored_old_log_probs``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch.distributed as dist  # noqa: E402
import transformers  # noqa: E402

if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = object

import recipe.gear_tree.policy_loss  # noqa: E402,F401 — registers the VDRA losses
from recipe.gear_tree.tree_data import edges_to_dataproto  # noqa: E402

VOCAB = 32
N_EDGES = 512
MINI = 128
MICRO = 64


class _Tok:
    pad_token_id = 0
    eos_token_id = 0


class TinyLM(torch.nn.Module):
    """Minimal causal-LM-shaped module: returns ``.logits`` like HF models."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, 8)
        self.head = torch.nn.Linear(8, VOCAB)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, use_cache=False, **_):
        logits = self.head(self.embed(input_ids))
        return SimpleNamespace(logits=logits)


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


def _actor_config():
    from verl.workers.config.actor import FSDPActorConfig, PolicyLossConfig

    return FSDPActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_mini_batch_size=MINI,
        ppo_micro_batch_size_per_gpu=MICRO,
        ppo_epochs=1,
        clip_ratio=0.2,
        grad_clip=1.0,
        use_torch_compile=False,
        use_dynamic_bsz=False,
        policy_loss=PolicyLossConfig(
            loss_mode="vdra_segment_mean_ppo",
            segment_token_reduction="mean",
            use_prob_mask=False,
        ),
    )


def _build_batch(n: int = N_EDGES, advantage: float = 1.0):
    batch = edges_to_dataproto(
        _edges(n=n, advantage=advantage),
        _Tok(),
        max_prompt_length=6,
        max_response_length=4,
        include_old_log_probs=True,
        loss_mode="vdra_segment_mean_ppo",
    )
    batch.meta_info["temperature"] = 1.0
    # P0.4: replay edges force the stored PPO denominator.
    batch.meta_info["force_stored_old_log_probs"] = True
    return batch


def _make_actor():
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    model = TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    actor = DataParallelPPOActor(
        config=_actor_config(), actor_module=model, actor_optimizer=optimizer
    )
    return actor, model, optimizer


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
