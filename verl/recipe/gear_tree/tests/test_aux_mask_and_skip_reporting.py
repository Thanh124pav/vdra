"""Auxiliary-mask, step-metric and skipped-iteration reporting contracts.

Required tests 3, 4 and 7 of the remaining-fixes spec:

3. dummy padding rows are excluded from entropy/KL reductions, so a
   non-strict result cannot depend on the DP size;
4. a missing ``actor/num_optimizer_steps`` metric invalidates the accounting
   instead of silently defaulting to 1 — the outer update stays committed;
7. a fully skipped iteration writes a timing row and persists the manifest
   with ``actor_update_skipped=true`` and zero expected/actual steps.
"""

from __future__ import annotations

import json

import pytest

try:  # namespace-package import under PYTHONPATH=verl
    from recipe.gear_tree.tests import _test_shims
except ImportError:  # flat rootdir-relative import
    import _test_shims

_test_shims.install()

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

import torch.distributed as dist  # noqa: E402

from recipe.gear_tree.gear_ray_trainer import RayGearTreeTrainer  # noqa: E402
from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer  # noqa: E402
from recipe.gear_tree.run_manifest import RunManifest  # noqa: E402

try:
    from recipe.gear_tree.tests import _tiny_actor
except ImportError:
    import _tiny_actor

MINI = 4
LP = -0.2


# --------------------------------------------------------------------------
# Required test 3 — entropy/KL must not see dummy padding.
# --------------------------------------------------------------------------


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


def _edge(i: int, adv: float, n_tokens: int = 2) -> dict:
    lps = [LP] * n_tokens
    return {
        "edge_id": f"t0/e{i}",
        "tree_id": "t0",
        "parent_group_id": "t0/pg",
        "child_segment_id": f"t0/e{i}",
        "question_id": f"q{i % 3}",
        "allocated_k": 4,
        "sample_multiplicity": 1,
        "tree_total_segment_count": 4,
        "queue_flush_id": "0",
        "queue_released_segment_count": 4,
        "query_token_ids": [1, 2],
        "response_token_ids": [3 + i, 4],
        "actor_shifted_log_probs": lps,
        "advantage": adv,
        "value": 0.4,
        "reward": 1.0,
        "advantage_is_zero": adv == 0.0,
        "response_token_count": n_tokens,
        "prob_mask_token_count": n_tokens,
        "probability_mask_threshold": 0.9,
        "generation_rollout_iteration": 0,
    }


def _slot(i: int, n_tokens: int = 2) -> dict:
    return {
        "edge_id": f"t0/z{i}",
        "tree_id": "t0",
        "parent_group_id": "t0/pg",
        "child_segment_id": f"t0/z{i}",
        "question_id": f"q{i % 3}",
        "allocated_k": 4,
        "sample_multiplicity": 1,
        "advantage": 0.0,
        "advantage_is_zero": True,
        "trainable_edge_id": None,
        "response_token_count": n_tokens,
        "prob_mask_token_count": n_tokens,
        "probability_mask_threshold": 0.9,
        "generation_rollout_iteration": 0,
    }


def _build(slots, dp_size=1):
    from recipe.gear_tree.tree_data import build_logical_update_batch

    batch, stats = build_logical_update_batch(
        slots,
        _tiny_actor.Tok(),
        max_prompt_length=_tiny_actor.MAX_PROMPT,
        max_response_length=_tiny_actor.MAX_RESPONSE,
        ppo_mini_batch_size=MINI,
        dp_size=dp_size,
        loss_mode="vdra_segment_mean_ppo",
        use_prob_mask=False,
        probability_mask_threshold=0.9,
    )
    assert batch is not None
    batch.meta_info["temperature"] = 1.0
    batch.meta_info["force_stored_old_log_probs"] = True
    return batch, stats


@pytest.mark.usefixtures("single_process_group")
class TestAuxMaskExcludesDummyRows:
    """Required test 3: with entropy enabled (a non-strict ablation), a row
    flagged ``is_dummy`` must contribute NOTHING to the entropy/KL reduction
    — otherwise the result would depend on how many collective-safety
    padding rows the DP size happened to require.

    Driven in a single process by flipping the flag on an otherwise
    identical batch (a real dp=2 padded batch cannot run under world_size=1,
    which dp_actor correctly refuses).
    """

    def _run(self, slots, *, mark_last_dummy=False, monkeypatch=None):
        """Run one update with entropy enabled; return (aux masks, delta)."""
        from verl.workers.actor import dp_actor as dp_actor_mod

        # micro >= row count so the reduction happens once per run.
        cfg = _tiny_actor.make_actor_config(
            strategy="fsdp", mini=MINI, micro=MINI, aggregation="segment_mean"
        )
        # Non-strict ablation: entropy on, so agg_loss actually runs.
        object.__setattr__(cfg, "entropy_coeff", 0.01)
        actor, model, _ = _tiny_actor.make_actor(config=cfg)

        seen_masks = []
        if monkeypatch is not None:
            real_agg = dp_actor_mod.agg_loss

            def _spy(*, loss_mat, loss_mask, loss_agg_mode):
                seen_masks.append(loss_mask.detach().clone())
                return real_agg(
                    loss_mat=loss_mat,
                    loss_mask=loss_mask,
                    loss_agg_mode=loss_agg_mode,
                )

            monkeypatch.setattr(dp_actor_mod, "agg_loss", _spy)

        batch, _ = _build(slots, dp_size=1)
        if mark_last_dummy:
            flags = batch.batch["is_dummy"].clone()
            flags[-1] = 1
            batch.batch["is_dummy"] = flags

        before = [p.detach().clone() for p in model.parameters()]
        actor.update_policy(batch)
        delta = [
            (p.detach() - b) for p, b in zip(model.parameters(), before)
        ]
        return seen_masks, delta

    def test_entropy_reduction_receives_a_dummy_free_mask(self, monkeypatch):
        """The mask handed to the entropy reduction must zero the flagged
        row — the contract, observed directly."""
        four = [_edge(0, 0.5), _edge(1, -0.5), _edge(2, 0.3), _edge(3, -0.7)]
        masks, _ = self._run(four, mark_last_dummy=True, monkeypatch=monkeypatch)
        assert masks, "entropy reduction did not run"
        aux = masks[0]
        assert int(aux[-1].sum()) == 0, "dummy row leaked into the entropy mask"
        assert int(aux[:-1].sum()) > 0, "real rows must still be counted"

    def test_flagged_row_changes_nothing_versus_a_batch_without_it(self):
        """End-to-end: masking the 4th row reproduces the 3-real-row update
        exactly (same M_B, same entropy denominator), and genuinely differs
        from counting all four rows."""
        four = [_edge(0, 0.5), _edge(1, -0.5), _edge(2, 0.3), _edge(3, -0.7)]
        three = [_edge(0, 0.5), _edge(1, -0.5), _edge(2, 0.3), _slot(0)]

        _, delta_all_four = self._run(four)
        _, delta_masked = self._run(four, mark_last_dummy=True)
        _, delta_three = self._run(three)

        for a, b in zip(delta_masked, delta_three):
            assert torch.allclose(a, b, atol=1e-7)
        assert any(
            not torch.allclose(a, b, atol=1e-7)
            for a, b in zip(delta_masked, delta_all_four)
        )

    def test_dummy_row_is_present_but_masked(self):
        """The dummy row DOES carry response-mask tokens — only the EXPLICIT
        mask keeps them out of the reductions (never the probability mask)."""
        slots = [_edge(0, 0.5), _edge(1, -0.5), _edge(2, 0.3), _slot(0)]
        batch, stats = _build(slots, dp_size=2)
        assert stats["vdra/dummy_rows"] == 1.0
        is_dummy = batch.batch["is_dummy"]
        assert int(is_dummy.sum()) == 1
        assert int(batch.batch["response_mask"][is_dummy.bool()].sum()) > 0


# --------------------------------------------------------------------------
# Required tests 4 and 7 — trainer-level reporting.
# --------------------------------------------------------------------------


class _Cfg(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _trainer(tmp_path):
    obj = object.__new__(RayGearTreeTrainer)
    obj.tokenizer = _tiny_actor.Tok()
    obj.config = _Cfg(
        data=_Cfg(max_prompt_length=6, max_response_length=4),
        trainer=_Cfg(
            balance_batch=False,
            default_local_dir=str(tmp_path),
            nnodes=1,
            n_gpus_per_node=1,
        ),
        actor_rollout_ref=_Cfg(
            actor=_Cfg(
                ppo_mini_batch_size=MINI,
                ppo_epochs=1,
                ulysses_sequence_parallel_size=1,
                policy_loss={
                    "loss_mode": "vdra_segment_mean_ppo",
                    "policy_aggregation": "segment_mean",
                    "use_prob_mask": False,
                    "probability_mask_threshold": 0.9,
                },
            )
        ),
        gear_tree={
            "replay_buffer": {
                "target_edges_per_iteration": 64,
                "max_edge_age_iterations": 8,
                "max_edges_per_question_per_iteration": 32,
                "sampling_seed": 0,
            },
            "gear": {"strict_vdra": False},
        },
    )
    obj.run_manifest = RunManifest()
    obj.rollout_iteration = 1
    obj.global_steps = 0
    obj.num_optimizer_steps_total = 0
    obj.successful_actor_updates = 0
    obj.optimizer_steps_this_iteration = 0
    obj.skipped_zero_gradient_updates = 0
    obj.failed_updates = 0
    obj._resolved_max_edge_prompt_length = lambda: 6
    return obj


class _ActorOutput:
    def __init__(self, meta_info):
        self.meta_info = meta_info


class TestMissingStepMetricInvalidatesAccounting:
    """Required test 4."""

    def _finalize(self, tmp_path, actor_metrics):
        trainer = _trainer(tmp_path)
        metrics: dict = {}
        slots = [_edge(0, 0.5), _edge(1, -0.5), _slot(0), _slot(1)]
        trainer._edges_to_update_batch(slots, metrics)
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=64,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=32,
            use_prob_mask=False,
            probability_mask_threshold=0.9,
        )
        buf.add(
            [_edge(9, 0.5)],
            generation_rollout_iteration=1,
            policy_snapshot_id="snap",
        )
        reservation = buf.reserve_for_update(current_rollout_iteration=1)
        trainer._finalize_successful_actor_update(
            buf,
            reservation,
            _ActorOutput({"metrics": actor_metrics}),
            slots,
            {},
            metrics,
        )
        return trainer, metrics

    def test_missing_metric_does_not_default_to_one(self, tmp_path):
        trainer, metrics = self._finalize(tmp_path, {"actor/pg_loss": [0.1]})
        # The outer update MUST stand.
        assert trainer.successful_actor_updates == 1
        assert trainer.global_steps == 1
        # ...but the internal accounting is unknown, never fabricated as 1.
        assert trainer.optimizer_steps_this_iteration == 0
        assert trainer.num_optimizer_steps_total == 0
        assert trainer.run_manifest.optimizer_step_accounting_valid is False
        assert metrics["vdra/actor_metrics_parse_failed"] == 1.0

    def test_present_metric_is_used(self, tmp_path):
        trainer, _ = self._finalize(
            tmp_path,
            {"actor/num_optimizer_steps": [1], "actor/pg_loss": [0.1]},
        )
        assert trainer.optimizer_steps_this_iteration == 1
        assert trainer.num_optimizer_steps_total == 1
        assert trainer.run_manifest.optimizer_step_accounting_valid is True


class TestFullySkippedIterationReporting:
    """Required test 7: the skipped-iteration branch must be auditable."""

    def test_skipped_reservation_yields_none_and_zero_expectation(self, tmp_path):
        trainer = _trainer(tmp_path)
        metrics: dict = {}
        # Every slot is a zero slot -> no trainable rows at all.
        batch = trainer._edges_to_update_batch([_slot(i) for i in range(8)], metrics)
        assert batch is None
        assert metrics["vdra/skipped_zero_gradient_updates"] == 1.0
        assert metrics["vdra/all_zero_advantage_logical_batches"] == 2.0

    def test_manifest_records_the_skip_with_zero_steps(self, tmp_path):
        trainer = _trainer(tmp_path)
        metrics: dict = {}
        trainer._edges_to_update_batch([_slot(i) for i in range(8)], metrics)
        # The fit loop marks the skip and records observed facts.
        trainer._expected_optimizer_steps = 0
        trainer.run_manifest.actor_update_skipped = True
        trainer._record_iteration_on_manifest(
            selected_edges=8, sample_stats={}, actual_optimizer_steps=0
        )
        assert trainer.run_manifest.actor_update_skipped is True
        assert trainer.run_manifest.expected_optimizer_steps_last_iteration == 0
        # 0 expected == 0 performed: the accounting is VALID for a skip.
        assert trainer.run_manifest.optimizer_step_accounting_valid is True
        # Outer counters untouched.
        assert trainer.global_steps == 0
        assert trainer.num_optimizer_steps_total == 0

    def test_manifest_persists_the_skip_flag(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.run_manifest.actor_update_skipped = True
        trainer._save_manifest(trainer.run_manifest)
        saved = json.loads(
            (tmp_path / "vdra_run_manifest.json").read_text(encoding="utf-8")
        )
        assert saved["actor_update_skipped"] is True

    def test_a_real_update_clears_the_skip_flag(self, tmp_path):
        trainer = _trainer(tmp_path)
        trainer.run_manifest.actor_update_skipped = True
        metrics: dict = {}
        slots = [_edge(0, 0.5), _edge(1, -0.5), _slot(0), _slot(1)]
        trainer._edges_to_update_batch(slots, metrics)
        buf = GearTreeReplayBuffer(
            target_edges_per_iteration=64,
            max_edge_age_iterations=8,
            max_edges_per_question_per_iteration=32,
            use_prob_mask=False,
            probability_mask_threshold=0.9,
        )
        buf.add(
            [_edge(9, 0.5)],
            generation_rollout_iteration=1,
            policy_snapshot_id="snap",
        )
        reservation = buf.reserve_for_update(current_rollout_iteration=1)
        trainer._finalize_successful_actor_update(
            buf,
            reservation,
            _ActorOutput({"metrics": {"actor/num_optimizer_steps": [1]}}),
            slots,
            {},
            metrics,
        )
        assert trainer.run_manifest.actor_update_skipped is False
