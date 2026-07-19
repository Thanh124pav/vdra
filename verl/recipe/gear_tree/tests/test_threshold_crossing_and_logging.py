"""PLAN.md P0.E: save/eval fire on crossed thresholds, logging is unambiguous.

The preserved host contract uses outer-update ``global_step`` units. The
threshold helper remains defensive for resume/migration or any non-unit jump,
and production code uses ``initial_next_threshold`` +
``advance_past_thresholds`` from ``trainer_state.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recipe.gear_tree.trainer_state import (
    advance_past_thresholds,
    initial_next_threshold,
)


class TestInitialNextThreshold:
    def test_fresh_run(self):
        assert initial_next_threshold(0, 10) == 10

    def test_mid_interval(self):
        assert initial_next_threshold(8, 10) == 10

    def test_resume_exactly_on_threshold_does_not_refire(self):
        # A checkpoint saved at step 10 already fired the step-10 save.
        assert initial_next_threshold(10, 10) == 20

    def test_resume_after_jump(self):
        assert initial_next_threshold(412, 10) == 420

    def test_disabled_freq(self):
        assert initial_next_threshold(8, 0) is None
        assert initial_next_threshold(8, -1) is None


class TestAdvancePastThresholds:
    def test_jump_8_to_12_crosses_10(self):
        crossed, nxt = advance_past_thresholds(
            previous_step=8, current_step=12, next_threshold=10, freq=10
        )
        assert crossed == 1
        assert nxt == 20

    def test_jump_8_to_28_crosses_two_thresholds(self):
        crossed, nxt = advance_past_thresholds(
            previous_step=8, current_step=28, next_threshold=10, freq=10
        )
        # Fire once (same trainer state), but advance past 10 AND 20.
        assert crossed == 2
        assert nxt == 30

    def test_no_crossing(self):
        crossed, nxt = advance_past_thresholds(
            previous_step=8, current_step=9, next_threshold=10, freq=10
        )
        assert crossed == 0
        assert nxt == 10

    def test_exact_landing_fires(self):
        crossed, nxt = advance_past_thresholds(
            previous_step=9, current_step=10, next_threshold=10, freq=10
        )
        assert crossed == 1
        assert nxt == 20

    def test_no_progress_does_not_fire(self):
        crossed, nxt = advance_past_thresholds(
            previous_step=10, current_step=10, next_threshold=10, freq=10
        )
        assert crossed == 0
        assert nxt == 10

    def test_disabled_freq_never_fires(self):
        crossed, nxt = advance_past_thresholds(
            previous_step=0, current_step=100, next_threshold=None, freq=0
        )
        assert crossed == 0
        assert nxt is None

    def test_consecutive_iterations_fire_each_threshold_once(self):
        # 0 -> 4 -> 8 -> 12 -> 16 -> 20 with freq 10: fires at the 8->12
        # and 16->20 iterations only.
        nxt = initial_next_threshold(0, 10)
        fired_at = []
        step = 0
        for _ in range(5):
            prev, step = step, step + 4
            crossed, nxt = advance_past_thresholds(
                previous_step=prev, current_step=step,
                next_threshold=nxt, freq=10,
            )
            if crossed:
                fired_at.append(step)
        assert fired_at == [12, 20]
        assert nxt == 30


class TestTrainerWiring:
    def _source(self) -> str:
        return (
            Path(__file__).resolve().parents[1] / "gear_ray_trainer.py"
        ).read_text()

    def test_no_modulo_triggers_remain(self):
        source = self._source()
        assert "% test_freq" not in source
        assert "% save_freq" not in source
        assert "advance_past_thresholds(" in source
        assert "initial_next_threshold(" in source

    def test_log_keys_are_unambiguous(self):
        source = self._source()
        # The ambiguous key is gone (the *_this_iteration key is distinct).
        assert '"training/optimizer_step"' not in source
        assert "'training/optimizer_step'" not in source
        assert '"training/global_step_before_update"' in source
        assert '"training/global_step_after_update"' in source
        assert '"training/global_step"' in source

    def test_no_stale_host_contract_claims_in_production_docs(self):
        root = Path(__file__).resolve().parents[1]
        targets = [
            root / "gear_ray_trainer.py",
            root / "policy_loss.py",
            root / "run_manifest.py",
            root / "config" / "gear_tree_trainer.yaml",
            root.parents[2] / "scripts" / "pre_gpu_check.sh",
        ]
        forbidden = [
            "global_step += 4",
            "global_step is the internal optimizer-step count",
            "scheduler must step once per internal PPO mini-batch",
            "total_training_steps must use optimizer-step units",
            "segment_objective_weights are required by vdra_segment_mean_ppo",
            "segment -> tree (divide by pre-filter",
            "segment-denominator",
            "training/all_zero_batch_skipped",
        ]
        for target in targets:
            source = target.read_text(encoding="utf-8")
            for phrase in forbidden:
                assert phrase not in source, f"{phrase!r} found in {target}"
