"""``RayGearTreeTrainer`` - VERL loop driven by native segment-tree rollout.

Generation builds tree edges with precomputed advantages and generation-time
behavior log-probabilities. A trainer-owned replay buffer applies the same edge
sampling protocol across SPO-tree and VDRA tree-family methods before forwarding
sampled edges to the actor update.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

_LOGGER = logging.getLogger(__name__)

from verl import DataProto
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

from recipe.gear_tree.config_validation import validate_policy_loss_consistency
from recipe.gear_tree.context_contract import (
    normalize_tree_shape,
    resolve_max_edge_prompt_length,
    resolve_max_original_prompt_length,
    validate_context_contract,
)
from recipe.gear_tree.replay_buffer import (
    GearTreeReplayBuffer,
    LOGICAL_SLOT_SCHEMA_VERSION,
    expected_optimizer_steps,
    reserve_replay_edges,
    should_postpone_sampled_update,
)
from recipe.gear_tree.manifest_lifecycle import (
    build_run_manifest,
    update_manifest_from_generated_edges,
    update_manifest_from_replay_batch,
)
from recipe.gear_tree.run_manifest import (
    ITERATION_STATUS_ACTOR_FAILED,
    ITERATION_STATUS_ALL_ZERO_SKIPPED,
    ITERATION_STATUS_FAILED_BEFORE_ACTOR,
    ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED,
    ITERATION_STATUS_NO_SAMPLE,
    ITERATION_STATUS_NOT_STARTED,
    ITERATION_STATUS_POSTPONED,
    ITERATION_STATUS_RUNNING,
    ITERATION_STATUS_UPDATED,
    ITERATION_STATUS_ZERO_ACTIVE_SKIPPED,
    VALID_ITERATION_STATUSES,
    ZERO_SIGNAL_SKIP_STATUSES,
    POLICY_AGGREGATION_LEGACY,
    POLICY_AGGREGATION_VDRA,
    RunManifest,
    validate_main_run,
)
from recipe.gear_tree.trainer_state import (
    GearTreeLiveState,
    GearTreeTrainerState,
    advance_past_thresholds,
    initial_next_threshold,
    load_live_state,
    load_trainer_state,
    save_live_state,
    save_trainer_state,
    trainer_state_path,
)


class RayGearTreeTrainer(RayPPOTrainer):
    VDRA_CHECKPOINT_COMPLETE = "VDRA_CHECKPOINT_COMPLETE"

    def _gear_tree_config(self) -> dict:
        """Resolve the top-level ``gear_tree`` block to a plain dict + demos_dir."""
        from omegaconf import OmegaConf

        raw = self.config.get("gear_tree", {})
        gt = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else dict(raw or {})
        if not gt.get("demos_dir"):
            gt["demos_dir"] = os.path.join(self.config.trainer.default_local_dir, "gear_demos")
        return gt

    def _replay_config(self) -> Dict[str, Any]:
        """PLAN.md P0.2: resolve the replay block.

        Accepts both the canonical field names
        (``target_edges_per_iteration``, ``max_edge_age_iterations``,
        ``max_edges_per_question_per_iteration``, ``replay_sampling_unit``)
        and the deprecated aliases (``target_edges_per_update``,
        ``max_edge_age``, ``max_edges_per_question``). The resolved dict uses
        the canonical names.
        """
        gt = self._gear_tree_config()
        raw = dict(gt.get("replay_buffer") or {})

        def _pick(new_name: str, old_name: str, default):
            if new_name in raw and raw[new_name] is not None:
                return raw[new_name]
            if old_name in raw and raw[old_name] is not None:
                return raw[old_name]
            return default

        return {
            "enabled": True,
            "target_edges_per_iteration": _pick(
                "target_edges_per_iteration", "target_edges_per_update", 512
            ),
            "max_edge_age_iterations": _pick(
                "max_edge_age_iterations", "max_edge_age", 8
            ),
            "max_edges_per_question_per_iteration": _pick(
                "max_edges_per_question_per_iteration",
                "max_edges_per_question",
                "auto",
            ),
            "replay_sampling_unit": raw.get("replay_sampling_unit", "edge"),
            "underfill_policy": raw.get("underfill_policy", "use_available"),
            "sampling_seed": raw.get("sampling_seed", 0),
            # PLAN.md §13: false (default) = fail fast on a replay
            # objective-mask mismatch; true = DISCARD the incompatible rows
            # and restart replay collection under the new objective.
            "reset_replay_on_objective_mismatch": bool(
                raw.get("reset_replay_on_objective_mismatch", False)
            ),
            "checkpoint": raw.get("checkpoint", True),
            "underfilled_update_policy": raw.get(
                "underfilled_update_policy", "postpone_until_divisible"
            ),
        }

    def _resolve_trees_per_question(self) -> int:
        """PLAN.md P0.2: R = number of stochastic trees per question per
        rollout iteration. Read from ``actor_rollout_ref.rollout.n`` when set;
        otherwise assume ``1``.
        """
        try:
            n = int(
                self.config.actor_rollout_ref.rollout.get("n", 1) or 1
            )
        except Exception:
            n = 1
        return max(n, 1)

    def _new_replay_buffer(self) -> GearTreeReplayBuffer:
        replay_cfg = self._replay_config()
        gt = self._gear_tree_config()
        tree_shape = normalize_tree_shape(gt.get("tree_shape") or [])
        return GearTreeReplayBuffer(
            target_edges_per_iteration=int(replay_cfg["target_edges_per_iteration"]),
            max_edge_age_iterations=int(replay_cfg["max_edge_age_iterations"]),
            max_edges_per_question_per_iteration=replay_cfg[
                "max_edges_per_question_per_iteration"
            ],
            replay_sampling_unit=str(replay_cfg.get("replay_sampling_unit", "edge")),
            tree_shape=tree_shape,
            trees_per_question=self._resolve_trees_per_question(),
            underfill_policy=replay_cfg.get("underfill_policy", "use_available"),
            sampling_seed=replay_cfg.get("sampling_seed", 0),
            # PLAN.md §13: persist the objective-mask identity the stored
            # zero-slot active counts are computed under.
            use_prob_mask=bool(
                self.config.actor_rollout_ref.actor.policy_loss.get(
                    "use_prob_mask", True
                )
            ),
            probability_mask_threshold=float(
                self.config.actor_rollout_ref.actor.policy_loss.get(
                    "probability_mask_threshold", 0.9
                )
            ),
            # PLAN.md §4: canonical runs never recompute missing metadata.
            require_logical_denominator_metadata=self._is_canonical_aggregation(),
        )

    def _is_canonical_aggregation(self) -> bool:
        """True for the canonical paper objectives (segment_mean/token_mean)."""
        pl = self.config.actor_rollout_ref.actor.policy_loss
        return str(pl.get("loss_mode", "")) == "vdra_segment_mean_ppo" and str(
            pl.get("policy_aggregation", "segment_mean")
        ).strip().lower() in ("segment_mean", "token_mean")

    def _ensure_replay_buffer(self) -> GearTreeReplayBuffer:
        if not hasattr(self, "replay_buffer"):
            self.replay_buffer = self._new_replay_buffer()
        return self.replay_buffer

    def _checkpoint_dir_for_step(self, step: int) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{int(step)}")

    def _resolve_resume_checkpoint_dir(self) -> Optional[str]:
        """Return the checkpoint folder VERL will restore, if any."""
        trainer_cfg = self.config.trainer
        if trainer_cfg.resume_mode == "disable":
            return None
        if trainer_cfg.default_hdfs_dir is not None:
            return None
        if trainer_cfg.resume_mode == "resume_path":
            path = trainer_cfg.resume_from_path
            if not isinstance(path, str) or "global_step_" not in path:
                return None
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            return path
        checkpoint_folder = trainer_cfg.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        return find_latest_ckpt_path(checkpoint_folder)

    def _is_resuming_from_checkpoint(self) -> bool:
        return bool(getattr(self, "_restored_checkpoint_dir", None))

    # --- PLAN.md P0.E: counter state must survive checkpoint/resume ------- #

    def _checkpoint_complete_marker_path(self, checkpoint_dir: str | Path) -> Path:
        return Path(checkpoint_dir) / self.VDRA_CHECKPOINT_COMPLETE

    def _write_checkpoint_complete_marker(self, checkpoint_dir: str | Path) -> None:
        marker = self._checkpoint_complete_marker_path(checkpoint_dir)
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(".tmp")
        tmp.write_text("complete\n", encoding="utf-8")
        tmp.replace(marker)

    def _write_replay_buffer_checkpoint(self, checkpoint_dir: str | Path) -> None:
        if not hasattr(self, "replay_buffer") or self.replay_buffer is None:
            raise FileNotFoundError("VDRA checkpoint bundle requires replay buffer")
        self.replay_buffer.save(checkpoint_dir)
        meta_path = Path(checkpoint_dir) / "gear_tree_replay_buffer_meta.json"
        if meta_path.exists():
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            data["segment_length"] = int(
                self._gear_tree_config().get("segment_length", 0) or 0
            )
            tmp = meta_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tmp.replace(meta_path)

    def _save_vdra_checkpoint_bundle(self) -> None:
        """Save model state plus all VDRA files, then mark complete last."""
        ckpt_dir = Path(self._checkpoint_dir_for_step(self.global_steps))
        marker = self._checkpoint_complete_marker_path(ckpt_dir)
        if marker.exists():
            marker.unlink()
        super()._save_checkpoint()
        save_trainer_state(
            ckpt_dir,
            GearTreeTrainerState(
                global_step=int(self.global_steps),
                rollout_iteration=int(getattr(self, "rollout_iteration", 0)),
                num_optimizer_steps_total=int(
                    getattr(self, "num_optimizer_steps_total", 0)
                ),
                successful_actor_updates=int(
                    getattr(self, "successful_actor_updates", 0)
                ),
                postponed_updates=int(getattr(self, "postponed_updates", 0)),
                failed_updates=int(getattr(self, "failed_updates", 0)),
                skipped_zero_gradient_updates=int(
                    getattr(self, "skipped_zero_gradient_updates", 0)
                ),
                consecutive_nonprogress_iterations=int(
                    getattr(self, "consecutive_nonprogress_iterations", 0)
                ),
            ),
        )
        if getattr(self, "run_manifest", None) is None:
            raise FileNotFoundError("VDRA checkpoint bundle requires run manifest")
        self._save_manifest(
            self.run_manifest, path=self._checkpoint_manifest_path(ckpt_dir)
        )
        self._write_replay_buffer_checkpoint(ckpt_dir)
        self._write_checkpoint_complete_marker(ckpt_dir)

    def _save_checkpoint(self):
        """Compatibility wrapper: save a complete VDRA checkpoint bundle."""
        self._save_vdra_checkpoint_bundle()

    def _validate_vdra_checkpoint_bundle_for_resume(self, checkpoint_dir: str | Path) -> None:
        ckpt_dir = Path(checkpoint_dir)
        marker = self._checkpoint_complete_marker_path(ckpt_dir)
        if not marker.exists():
            if self._is_canonical_aggregation():
                raise FileNotFoundError(
                    f"canonical VDRA resume requires {marker}; checkpoint "
                    "bundle is incomplete"
                )
            return
        required = (
            trainer_state_path(ckpt_dir),
            Path(self._checkpoint_manifest_path(ckpt_dir)),
            ckpt_dir / "gear_tree_replay_buffer.jsonl",
            ckpt_dir / "gear_tree_replay_buffer_meta.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "VDRA checkpoint bundle marker exists but required files are "
                f"missing: {missing}"
            )

    def _load_checkpoint(self):
        """Restore base checkpoint plus VDRA counter state.

        Resume detection is path-based, not ``global_steps > 0``, so a
        canonical ``global_step_0`` checkpoint can restore trainer state,
        replay, and its checkpoint-scoped manifest.
        """
        self._restored_checkpoint_dir = self._resolve_resume_checkpoint_dir()
        self._pending_live_state = None
        self._live_state_resume_metrics: Dict[str, float] = {}
        self._legacy_checkpoint_without_state = False
        if self._is_resuming_from_checkpoint():
            self._validate_vdra_checkpoint_bundle_for_resume(
                self._restored_checkpoint_dir
            )
        ret = super()._load_checkpoint()
        if not self._is_resuming_from_checkpoint():
            return ret

        ckpt_dir = Path(self._restored_checkpoint_dir)
        state = load_trainer_state(ckpt_dir)
        if state is None:
            self._legacy_checkpoint_without_state = True
            print(
                "WARNING (PLAN.md P0.E): checkpoint "
                f"{ckpt_dir.name} has no gear_tree_trainer_state.json "
                "(legacy checkpoint). The replay buffer will be RESET and "
                "rollout_iteration restarts at 0 so replay ages can never "
                "go negative."
            )
            return ret

        if int(state.global_step) != int(self.global_steps):
            raise ValueError(
                "gear_tree_trainer_state.json global_step="
                f"{state.global_step} does not match checkpoint folder "
                f"global_step_{self.global_steps}"
            )
        self.rollout_iteration = int(state.rollout_iteration)
        self.num_optimizer_steps_total = int(state.num_optimizer_steps_total)
        self.successful_actor_updates = int(state.successful_actor_updates)
        self.postponed_updates = int(state.postponed_updates)
        self.failed_updates = int(state.failed_updates)
        self.skipped_zero_gradient_updates = int(
            state.skipped_zero_gradient_updates
        )
        self._set_nonprogress_counter(state.consecutive_nonprogress_iterations)

        live = load_live_state(self.config.trainer.default_local_dir)
        if live is not None:
            if int(live.global_step) == int(state.global_step):
                if int(live.rollout_iteration) >= int(state.rollout_iteration):
                    self._pending_live_state = live
                    self._live_state_resume_metrics["vdra/live_state_merged"] = 1.0
                else:
                    self._live_state_resume_metrics[
                        "vdra/live_state_stale_rollout"
                    ] = 1.0
            elif int(live.global_step) > int(state.global_step):
                self._live_state_resume_metrics[
                    "vdra/live_state_ahead_of_checkpoint"
                ] = 1.0
            else:
                self._live_state_resume_metrics["vdra/live_state_stale"] = 1.0
        return ret

    def _restore_or_init_replay_buffer(self) -> Dict[str, Any]:
        replay_cfg = self._replay_config()
        metrics = {
            "buffer/checkpoint_restored": 0.0,
            "buffer/reset_on_resume": 0.0,
            "buffer/legacy_checkpoint_reset": 0.0,
        }
        ckpt_dir = Path(
            getattr(self, "_restored_checkpoint_dir", None)
            or self._checkpoint_dir_for_step(self.global_steps)
        )
        should_restore = (
            self._is_resuming_from_checkpoint()
            or int(getattr(self, "global_steps", 0) or 0) > 0
            or trainer_state_path(ckpt_dir).exists()
        )
        if not should_restore:
            self.replay_buffer = self._new_replay_buffer()
            return metrics

        # PLAN.md P0.E option A: a legacy checkpoint (no trainer-state file)
        # must NOT restore replay — its edges carry generation iterations far
        # above the reset rollout_iteration and would get negative ages.
        if getattr(self, "_legacy_checkpoint_without_state", False):
            self.replay_buffer = self._new_replay_buffer()
            metrics["buffer/reset_on_resume"] = 1.0
            metrics["buffer/legacy_checkpoint_reset"] = 1.0
            metrics.update(getattr(self, "_live_state_resume_metrics", {}))
            self.replay_buffer_resume_metrics = metrics
            return metrics

        meta_path = ckpt_dir / "gear_tree_replay_buffer_meta.json"
        if replay_cfg.get("checkpoint", True) and meta_path.exists():
            # PLAN.md §13: a restored zero slot's active-token count was
            # computed under a specific objective-mask configuration and can
            # never be recomputed — verify compatibility or fail fast.
            _pl = self.config.actor_rollout_ref.actor.policy_loss
            self.replay_buffer = GearTreeReplayBuffer.load(
                ckpt_dir,
                expected_use_prob_mask=bool(_pl.get("use_prob_mask", True)),
                expected_probability_mask_threshold=float(
                    _pl.get("probability_mask_threshold", 0.9)
                ),
                reset_replay_on_objective_mismatch=bool(
                    replay_cfg.get("reset_replay_on_objective_mismatch", False)
                ),
                require_logical_denominator_metadata=(
                    self._is_canonical_aggregation()
                ),
            )
            metrics["buffer/checkpoint_restored"] = 1.0
            metrics["buffer/restored_edges"] = float(len(self.replay_buffer))
        else:
            self.replay_buffer = self._new_replay_buffer()
            metrics["buffer/reset_on_resume"] = 1.0
        metrics.update(getattr(self, "_live_state_resume_metrics", {}))
        self.replay_buffer_resume_metrics = metrics
        return metrics

    def _current_policy_snapshot_id(self) -> str:
        return f"global_step:{int(self.global_steps)}"

    def _resolved_max_edge_prompt_length(self) -> int:
        """P0.2: delegate to :func:`resolve_max_edge_prompt_length` so startup
        validation, edges_to_dataproto, and future tensorization sites share
        one resolver.
        """

        return resolve_max_edge_prompt_length(self.config.data)

    def _resolved_max_original_prompt_length(self) -> int:
        return resolve_max_original_prompt_length(self.config.data)

    def _validate_replay_startup(self) -> None:
        replay_cfg = self._replay_config()
        target = int(replay_cfg["target_edges_per_iteration"])
        ppo_mini = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        if target % ppo_mini != 0:
            raise ValueError(
                "gear_tree.replay_buffer.target_edges_per_iteration must be "
                "divisible by actor_rollout_ref.actor.ppo_mini_batch_size "
                "(PLAN.md P0.2)."
            )
        # P1.4: `replay_buffer.enabled` is not a real ablation switch — the
        # trainer always routes through the buffer. Reject the field so a
        # config that sets it to False cannot silently do nothing.
        raw_replay = self._gear_tree_config().get("replay_buffer") or {}
        if "enabled" in raw_replay and not bool(raw_replay["enabled"]):
            raise ValueError(
                "gear_tree.replay_buffer.enabled=false is not supported "
                "(PLAN.md P1.4). Remove the field or set it to true."
            )
        # P0.5: strict no-truncation forbids silent context-length overflow at
        # training time. edges_to_dataproto REJECTS any accumulated edge query
        # longer than data.max_prompt_length (there is no truncation escape
        # hatch), so the correct precondition is:
        #     L_original_max + (d - 1) * M  <=  L_edge_max
        # where L_edge_max is the actor's edge-input limit (data.max_edge_prompt_length
        # if set, otherwise data.max_prompt_length — the *same* limit
        # edges_to_dataproto uses). The previous "max_prompt * 8" heuristic
        # accepted configs that then failed at training time when the deepest
        # edge query overflowed the actor limit.
        gt = self._gear_tree_config()
        try:
            model_context = int(
                self.config.actor_rollout_ref.rollout.get("prompt_length", 0)
                or self.config.actor_rollout_ref.rollout.get("max_model_len", 0)
                or 0
            )
        except Exception:
            model_context = 0
        # P0.2: single-source validation of the entire context contract.
        validate_context_contract(
            data_cfg=self.config.data,
            tree_shape=normalize_tree_shape(gt.get("tree_shape") or []),
            segment_length=int(gt.get("segment_length", 0) or 0),
            model_context_length=model_context,
        )
        # PLAN.md M5: the cross-level tree_policy <-> actor.policy_loss
        # validation lives in config_validation.validate_policy_loss_consistency
        # so the pre-GPU Hydra gate runs the exact same code.
        validate_policy_loss_consistency(self.config, gear_tree_cfg=gt)

    def _should_postpone_sampled_update(self, sampled_edges: List[Dict[str, Any]]) -> bool:
        """PLAN.md P0.D: enforce exact optimizer-batch cardinality.

        The sampler must never return more than
        ``target_edges_per_iteration`` — exceeding it is a sampler bug and
        raises. In canonical mode (``postpone_until_divisible``) any selected
        count not divisible by ``ppo_mini_batch_size`` is postponed, whether
        under- or over-filled, so no tail optimizer batch can ever form.
        """
        replay_cfg = self._replay_config()
        return should_postpone_sampled_update(
            selected_count=len(sampled_edges),
            target_edges_per_iteration=int(
                replay_cfg["target_edges_per_iteration"]
            ),
            ppo_mini_batch_size=int(
                self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ),
            underfilled_update_policy=str(
                replay_cfg.get(
                    "underfilled_update_policy", "postpone_until_divisible"
                )
            ),
        )

    def _fetch_rollout_server_weight_version(self, gear_cfg: Dict[str, Any]) -> str | None:
        """PLAN.md P0.5: delegate to :func:`scorer_verification.fetch_rollout_weight_version`
        so the two-mode contract can be unit-tested on CPU.
        """
        from recipe.gear_tree.scorer_verification import fetch_rollout_weight_version

        try:
            from recipe.gear_tree.gear_core.gear.vllm_scorer import (
                fetch_server_weight_version,
            )
        except Exception:
            if bool(gear_cfg.get("strict_vdra", True)):
                raise
            return None
        return fetch_rollout_weight_version(
            gear_cfg, fetch_fn=fetch_server_weight_version
        )

    def _generate_tree_edges(self, gen_batch: DataProto) -> List[Dict[str, Any]]:
        """Run tree rollout and return raw replayable edge records."""
        gt = self._gear_tree_config()
        snapshot_id = self._current_policy_snapshot_id()
        gt["policy_snapshot_id"] = snapshot_id
        gt["current_rollout_snapshot_id"] = snapshot_id
        gear_cfg = gt.setdefault("gear", {})
        if isinstance(gear_cfg, dict):
            gear_cfg["policy_snapshot_id"] = snapshot_id
        # PLAN.md P0.5: fetch the rollout server's own weight version once
        # per generation. TreeAgentLoop.run reads
        # non_tensor_batch['rollout_server_weight_version'] and passes it to
        # gate.bind_snapshot, which lets the strict-mode gate refuse to
        # continue when the scorer's server fingerprint diverges. Record the
        # observed verification result on the manifest.
        rollout_server_version = self._fetch_rollout_server_weight_version(gear_cfg)
        if getattr(self, "run_manifest", None) is not None:
            verified = bool(rollout_server_version)
            self.run_manifest.rollout_scorer_weights_verified = verified
            self.run_manifest.extras["rollout_server_weight_version"] = (
                rollout_server_version if rollout_server_version else None
            )
            self.run_manifest.extras["scorer_uses_rollout_server"] = bool(
                gear_cfg.get("scorer_uses_rollout_server", False)
            )
        gen_batch.meta_info["gear_tree_config"] = gt
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["rollout_iteration"] = getattr(self, "rollout_iteration", 0)
        gen_batch.meta_info["policy_snapshot_id"] = snapshot_id
        gen_batch.meta_info["current_rollout_snapshot_id"] = snapshot_id
        gen_batch.meta_info["rollout_server_weight_version"] = rollout_server_version
        # P0.1: propagate the snapshot into per-row non_tensor_batch so
        # AgentLoopWorker forwards it to TreeAgentLoop.run() as kwargs.
        # meta_info is retained above only for logging; the agent loop reads
        # per-sample fields, not meta_info.
        if not hasattr(gen_batch, "non_tensor_batch") or gen_batch.non_tensor_batch is None:
            gen_batch.non_tensor_batch = {}
        row_count = len(gen_batch)
        snapshot_col = np.array([snapshot_id] * row_count, dtype=object)
        gen_batch.non_tensor_batch["policy_snapshot_id"] = snapshot_col
        gen_batch.non_tensor_batch["current_rollout_snapshot_id"] = snapshot_col
        gen_batch.non_tensor_batch["rollout_server_weight_version"] = np.array(
            [rollout_server_version] * row_count, dtype=object
        )
        # PLAN.md P0.2: send the rollout_iteration and a per-row uuid to the
        # agent loop so each stochastic tree gets a globally-unique
        # tree_instance_id. The uuid is redundant with the counter fallback in
        # tree_rollout.make_tree_instance_id but is preferred because it makes
        # replay/tensorization idempotent regardless of worker scheduling.
        import uuid as _uuid

        gen_batch.non_tensor_batch["rollout_iteration"] = np.array(
            [self.rollout_iteration] * row_count, dtype=object
        )
        gen_batch.non_tensor_batch["tree_instance_uuid"] = np.array(
            [_uuid.uuid4().hex for _ in range(row_count)], dtype=object
        )
        # PLAN.md §2: the objective-mask configuration lives ONLY under
        # actor.policy_loss. Propagate the RESOLVED values per request so the
        # rollout worker never relies on a stale constructor-time copy after
        # resume or config composition, and so extraction-time active-token
        # counting uses exactly the threshold the actor loss will use.
        _pl = self.config.actor_rollout_ref.actor.policy_loss
        gen_batch.non_tensor_batch["policy_use_prob_mask"] = np.array(
            [bool(_pl.get("use_prob_mask", True))] * row_count, dtype=object
        )
        gen_batch.non_tensor_batch["policy_probability_mask_threshold"] = np.array(
            [float(_pl.get("probability_mask_threshold", 0.9))] * row_count,
            dtype=object,
        )
        backend = gt.get("rollout_backend", "async")
        if backend != "async":
            raise NotImplementedError(
                "Replay-buffered RayGearTreeTrainer currently requires rollout_backend='async' "
                "so raw generation-time log-probability edges are available."
            )

        from recipe.gear_tree.async_tree_rollout import (
            collect_tree_construction_summaries,
            collect_tree_edges,
        )

        rollout_out = self.actor_rollout_wg.generate_sequences(gen_batch)
        self._last_rollout_reward_parse_metrics = self._collect_rollout_reward_parse_metrics(rollout_out)
        edges = collect_tree_edges(rollout_out)
        # Zero-filter contract: per-tree construction summaries preserve the
        # realized/allocated facts of parents (or whole trees) whose retained
        # edge set is empty because every child had exactly zero advantage.
        self._last_construction_summaries = collect_tree_construction_summaries(
            rollout_out
        )
        return self._normalize_generated_edges(edges, snapshot_id=snapshot_id)

    def _collect_rollout_reward_parse_metrics(self, rollout_out: DataProto) -> Dict[str, float]:
        stats_items = []
        if getattr(rollout_out, "non_tensor_batch", None) is not None:
            stats_items = list(rollout_out.non_tensor_batch.get("gear_tree_reward_parse_stats", []) or [])
        attempts = 0.0
        failures = 0.0
        boxed = 0.0
        answer = 0.0
        for item in stats_items:
            if not isinstance(item, dict):
                continue
            attempts += float(item.get("reward/answer_parse_attempts", 0.0) or 0.0)
            failures += float(item.get("reward/answer_parse_failures", 0.0) or 0.0)
            boxed += float(item.get("reward/answer_parse_mode_boxed", 0.0) or 0.0)
            answer += float(item.get("reward/answer_parse_mode_answer", 0.0) or 0.0)
        return {
            "reward/answer_parse_attempts": attempts,
            "reward/answer_parse_failures": failures,
            "reward/answer_parse_failure_rate": float(failures / attempts) if attempts else 0.0,
            "reward/answer_parse_mode_boxed": boxed,
            "reward/answer_parse_mode_answer": answer,
        }

    def _normalize_generated_edges(
        self, edges: List[Dict[str, Any]], *, snapshot_id: str
    ) -> List[Dict[str, Any]]:
        """PLAN.md P0.H: delegate to the production normalizer. Strict main
        runs require make_tree_instance_id-derived identities and refuse the
        legacy fallback chains."""
        from recipe.gear_tree.tree_data import normalize_generated_edges

        strict = bool(
            (self.config.get("tree_policy") or {}).get(
                "strict_group_integrity", False
            )
        )
        return normalize_generated_edges(
            edges, snapshot_id=snapshot_id, strict=strict
        )

    def _edges_to_update_batch(
        self, sampled_edges: List[Dict[str, Any]], metrics: Dict[str, Any]
    ) -> Optional[DataProto]:
        from recipe.gear_tree.tree_data import (
            build_logical_update_batch,
            edges_to_dataproto,
        )

        # P0.2: use the same L_edge_max the startup validator resolved so a
        # config that clears validation cannot then fail here on a deep edge.
        # PLAN.md P0.C: the configured loss mode decides whether the float
        # objective-weight tensors are attached (node-balanced ablation only).
        loss_mode = str(
            self.config.actor_rollout_ref.actor.policy_loss.get(
                "loss_mode", "vdra_segment_mean_ppo"
            )
        )
        aggregation = str(
            self.config.actor_rollout_ref.actor.policy_loss.get(
                "policy_aggregation", "segment_mean"
            )
        ).strip().lower()
        canonical_aggregation = loss_mode == "vdra_segment_mean_ppo" and aggregation in (
            "segment_mean",
            "token_mean",
        )
        if canonical_aggregation:
            # PLAN.md §1.2: the reservation is a list of LOGICAL slots in
            # reservation order; logical optimizer batches and their
            # pre-filter M_B/T_B denominators are fixed HERE, before tensor
            # filtering. Returns None for a fully-zero reservation (explicit
            # skipped update, PLAN.md §1.3).
            if self.config.trainer.get("balance_batch", False):
                raise ValueError(
                    "trainer.balance_batch reorders rows and would break the "
                    "rank-major logical-batch layout required by the "
                    "canonical VDRA aggregations (PLAN.md §1.2); disable it."
                )
            sp = int(
                self.config.actor_rollout_ref.actor.get(
                    "ulysses_sequence_parallel_size", 1
                )
                or 1
            )
            if sp > 1:
                raise NotImplementedError(
                    "canonical VDRA aggregations are only specified for "
                    "ulysses_sequence_parallel_size == 1 (PLAN.md §1.3)."
                )
            dp_size = max(
                int(self.config.trainer.nnodes)
                * int(self.config.trainer.n_gpus_per_node),
                1,
            )
            # PLAN.md §5: the denominators must be computed under the SAME
            # mask identity the actor loss will use.
            policy_loss_cfg = self.config.actor_rollout_ref.actor.policy_loss
            edge_batch, logical_stats = build_logical_update_batch(
                sampled_edges,
                self.tokenizer,
                max_prompt_length=self._resolved_max_edge_prompt_length(),
                max_response_length=self.config.data.max_response_length,
                ppo_mini_batch_size=int(
                    self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                ),
                dp_size=dp_size,
                loss_mode=loss_mode,
                include_old_log_probs=True,
                use_prob_mask=bool(policy_loss_cfg.get("use_prob_mask", True)),
                probability_mask_threshold=float(
                    policy_loss_cfg.get("probability_mask_threshold", 0.9)
                ),
                require_logical_denominator_metadata=True,
            )
            # PLAN.md §8: expected optimizer steps count only TRAINABLE
            # logical batches — a skipped batch must not mark the accounting
            # invalid.
            self._expected_optimizer_steps = int(
                logical_stats.get("vdra/trainable_logical_batches", 0)
            ) * int(self.config.actor_rollout_ref.actor.get("ppo_epochs", 1))
            # PLAN.md §14: record which denominator the run actually used.
            if aggregation == "segment_mean":
                observed_denominator = "segment_slots"
            elif bool(policy_loss_cfg.get("use_prob_mask", True)):
                observed_denominator = "prob_mask_tokens"
            else:
                observed_denominator = "response_tokens"
            self.run_manifest.observed_logical_denominator = observed_denominator
            metrics.update(logical_stats)
            if edge_batch is None:
                return None
        else:
            edge_batch = edges_to_dataproto(
                sampled_edges,
                self.tokenizer,
                max_prompt_length=self._resolved_max_edge_prompt_length(),
                max_response_length=self.config.data.max_response_length,
                include_old_log_probs=True,
                loss_mode=loss_mode,
            )
            if self.config.trainer.get("balance_batch", False):
                self._balance_batch(edge_batch, metrics=metrics)
        edge_batch.meta_info["global_token_num"] = edge_batch.batch["attention_mask"].sum(dim=-1).tolist()
        edge_batch.meta_info["multi_turn"] = False
        # P0.4: replay tree edges carry stored generation-time behavior
        # log-probs; the actor must always use them as the PPO denominator,
        # even in the single-minibatch/one-epoch shape that otherwise treats
        # the update as on-policy and overwrites old_log_prob with the
        # current policy's log_prob.
        edge_batch.meta_info["force_stored_old_log_probs"] = True
        # PLAN.md P0.5: forward the selected segment_token_reduction alongside
        # the edge batch so the actor's loss reads the same value the trainer
        # / manifest recorded.
        tree_policy = self.config.get("tree_policy") or {}
        edge_batch.meta_info["segment_token_reduction"] = str(
            tree_policy.get("segment_token_reduction", "mean")
        )
        if "old_log_probs" not in edge_batch.batch:
            raise AssertionError("edge_batch is missing stored old_log_probs")
        if edge_batch.batch["old_log_probs"].shape != edge_batch.batch["responses"].shape:
            raise AssertionError("old_log_probs shape must match responses shape")
        if (
            "edge_weights" in edge_batch.batch
            and edge_batch.batch["edge_weights"].shape != edge_batch.batch["responses"].shape
        ):
            raise AssertionError("edge_weights shape must match responses shape")
        return edge_batch

    def _generate_edge_batch(self, gen_batch: DataProto) -> DataProto:
        """Compatibility helper for callers that still expect a DataProto."""
        metrics: Dict[str, Any] = {}
        return self._edges_to_update_batch(self._generate_tree_edges(gen_batch), metrics)

    def _maybe_save_replay_buffer(self) -> Dict[str, Any]:
        replay_cfg = self._replay_config()
        if not replay_cfg.get("checkpoint", True) or not hasattr(self, "replay_buffer"):
            return {"buffer/checkpoint_saved": 0.0}
        ckpt_dir = self._checkpoint_dir_for_step(self.global_steps)
        self._write_replay_buffer_checkpoint(ckpt_dir)
        return {"buffer/checkpoint_saved": 1.0}

    # --- PLAN.md P0.N8: run-manifest lifecycle --------------------------- #

    def _build_run_manifest(self) -> RunManifest:
        """PLAN.md P0.N8 / P0.7: delegate to :func:`build_run_manifest` and
        stamp the config-derived replay/optimizer knobs so ``validate_main_run``
        can compare against observed runtime values later.
        """
        gt = self._gear_tree_config()
        manifest = build_run_manifest(
            tree_policy=(self.config.get("tree_policy") or {}),
            gear_tree_cfg=gt,
            actor_loss_mode=str(
                self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
            ),
        )
        replay_cfg = self._replay_config()
        manifest.replay_sampling_unit = str(
            replay_cfg.get("replay_sampling_unit", "edge")
        )
        manifest.target_edges_per_iteration = int(
            replay_cfg.get("target_edges_per_iteration", 0)
        )
        manifest.max_edge_age_iterations = int(
            replay_cfg.get("max_edge_age_iterations", 0)
        )
        manifest.ppo_mini_batch_size = int(
            self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size", 0)
        )
        manifest.ppo_epochs = int(
            self.config.actor_rollout_ref.actor.get("ppo_epochs", 1)
        )
        # PLAN.md §14: objective-mask snapshot + logical-slot schema version.
        _pl = self.config.actor_rollout_ref.actor.policy_loss
        manifest.use_prob_mask = bool(_pl.get("use_prob_mask", True))
        manifest.probability_mask_threshold = float(
            _pl.get("probability_mask_threshold", 0.9)
        )
        manifest.logical_slot_schema_version = LOGICAL_SLOT_SCHEMA_VERSION
        manifest.tree_shape = normalize_tree_shape(gt.get("tree_shape") or [])
        manifest.trees_per_question = int(self._resolve_trees_per_question())
        manifest.segment_length = int(gt.get("segment_length", 0) or 0)
        # The resolved auto cap is populated once the replay buffer is built.
        if hasattr(self, "replay_buffer") and self.replay_buffer is not None:
            manifest.resolved_max_edges_per_question_per_iteration = int(
                self.replay_buffer.resolved_max_edges_per_question_per_iteration
            )
        return manifest

    def _current_manifest_config_for_validation(self) -> RunManifest:
        current = self._build_run_manifest()
        # Resolve the cap from the current config, never from the restored
        # replay buffer metadata.
        fresh_buffer = self._new_replay_buffer()
        current.resolved_max_edges_per_question_per_iteration = int(
            fresh_buffer.resolved_max_edges_per_question_per_iteration
        )
        return current

    def _validate_resumed_manifest_config(
        self, manifest: RunManifest, *, checkpoint_dir: str | Path | None = None
    ) -> None:
        """Fail fast if a checkpoint manifest belongs to another resolved config."""
        current = self._current_manifest_config_for_validation()
        fields = (
            "policy_aggregation",
            "segment_token_reduction",
            "advantage_mode",
            "use_prob_mask",
            "probability_mask_threshold",
            "logical_slot_schema_version",
            "replay_sampling_unit",
            "target_edges_per_iteration",
            "resolved_max_edges_per_question_per_iteration",
            "max_edge_age_iterations",
            "ppo_mini_batch_size",
            "ppo_epochs",
            "tree_shape",
            "trees_per_question",
            "segment_length",
        )
        for field in fields:
            old = getattr(manifest, field)
            new = getattr(current, field)
            if isinstance(old, float) or isinstance(new, float):
                if abs(float(old) - float(new)) > 1e-12:
                    raise ValueError(
                        "resumed VDRA manifest config mismatch for "
                        f"{field}: checkpoint has {old!r}, current config "
                        f"resolves to {new!r}. Refusing to reset historical "
                        "manifest counters/invariants (PLAN.md §1)."
                    )
            elif old != new:
                raise ValueError(
                    "resumed VDRA manifest config mismatch for "
                    f"{field}: checkpoint has {old!r}, current config "
                    f"resolves to {new!r}. Refusing to reset historical "
                    "manifest counters/invariants (PLAN.md §1)."
                )
        if checkpoint_dir is not None:
            self._validate_replay_metadata_matches_current_config(
                checkpoint_dir, current
            )

    def _validate_replay_metadata_matches_current_config(
        self, checkpoint_dir: str | Path, current: RunManifest
    ) -> None:
        meta_path = Path(checkpoint_dir) / "gear_tree_replay_buffer_meta.json"
        if not meta_path.exists():
            if self._is_canonical_aggregation():
                raise FileNotFoundError(
                    f"canonical VDRA resume requires replay metadata {meta_path}"
                )
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checks = {
            "tree_shape": list(current.tree_shape),
            "trees_per_question": int(current.trees_per_question),
            "resolved_max_edges_per_question_per_iteration": int(
                current.resolved_max_edges_per_question_per_iteration
            ),
            "target_edges_per_iteration": int(current.target_edges_per_iteration),
            "max_edge_age_iterations": int(current.max_edge_age_iterations),
            "replay_sampling_unit": str(current.replay_sampling_unit),
            "segment_length": int(current.segment_length),
        }
        for field, expected in checks.items():
            if field not in meta:
                raise ValueError(
                    f"replay metadata is missing {field}; refusing resume"
                )
            old = meta[field]
            if isinstance(expected, list):
                old_norm = [int(x) for x in old]
            elif isinstance(expected, int):
                old_norm = int(old)
            else:
                old_norm = str(old)
            if old_norm != expected:
                raise ValueError(
                    "replay metadata config mismatch for "
                    f"{field}: checkpoint has {old!r}, current config "
                    f"resolves to {expected!r}"
                )

    def _checkpoint_manifest_path(self, checkpoint_dir: str | Path) -> str:
        return os.path.join(str(checkpoint_dir), "vdra_run_manifest.json")

    def _assert_resumed_manifest_counters(self, manifest: RunManifest) -> None:
        expected = {
            "global_step": int(self.global_steps),
            "rollout_iteration": int(self.rollout_iteration),
            "num_optimizer_steps_total": int(self.num_optimizer_steps_total),
        }
        actual = {
            "global_step": int(manifest.global_step),
            "rollout_iteration": int(manifest.rollout_iteration),
            "num_optimizer_steps_total": int(manifest.num_optimizer_steps_total),
        }
        for key, value in expected.items():
            if int(actual[key]) != int(value):
                raise ValueError(
                    "resumed VDRA manifest counter mismatch for "
                    f"{key}: checkpoint manifest has {actual[key]!r}, "
                    f"trainer state has {value!r}"
                )

    def _load_or_build_run_manifest(self) -> RunManifest:
        if self._is_resuming_from_checkpoint():
            ckpt_dir = Path(self._restored_checkpoint_dir)
            path = self._checkpoint_manifest_path(ckpt_dir)
            if os.path.exists(path):
                manifest = RunManifest.load(path)
                self._validate_resumed_manifest_config(
                    manifest, checkpoint_dir=ckpt_dir
                )
                self._assert_resumed_manifest_counters(manifest)
                return manifest
            if self._is_canonical_aggregation():
                raise FileNotFoundError(
                    "canonical VDRA resume requires checkpoint-scoped "
                    f"manifest {path}; refusing to rebuild from the live root "
                    "manifest (PLAN.md final resume fixes)."
                )
            manifest = self._build_run_manifest()
            manifest.manifest_resume_provenance_missing = True
            return manifest
        return self._build_run_manifest()

    def _current_live_state(self) -> GearTreeLiveState:
        return GearTreeLiveState(
            global_step=int(getattr(self, "global_steps", 0)),
            rollout_iteration=int(getattr(self, "rollout_iteration", 0)),
            num_optimizer_steps_total=int(
                getattr(self, "num_optimizer_steps_total", 0)
            ),
            successful_actor_updates=int(
                getattr(self, "successful_actor_updates", 0)
            ),
            postponed_updates=int(getattr(self, "postponed_updates", 0)),
            failed_updates=int(getattr(self, "failed_updates", 0)),
            skipped_zero_gradient_updates=int(
                getattr(self, "skipped_zero_gradient_updates", 0)
            ),
            consecutive_nonprogress_iterations=int(
                getattr(self, "consecutive_nonprogress_iterations", 0)
            ),
            last_iteration_status=str(
                getattr(
                    getattr(self, "run_manifest", None),
                    "last_iteration_status",
                    ITERATION_STATUS_NOT_STARTED,
                )
            ),
            group_integrity_failures=int(
                getattr(getattr(self, "run_manifest", None), "group_integrity_failures", 0)
            ),
            segment_count_failures=int(
                getattr(getattr(self, "run_manifest", None), "segment_count_failures", 0)
            ),
            replay_batch_failures=int(
                getattr(getattr(self, "run_manifest", None), "replay_batch_failures", 0)
            ),
            parent_split_count=int(
                getattr(getattr(self, "run_manifest", None), "parent_split_count", 0)
            ),
            tree_split_count=int(
                getattr(getattr(self, "run_manifest", None), "tree_split_count", 0)
            ),
            optimizer_step_accounting_observations=int(
                getattr(getattr(self, "run_manifest", None), "optimizer_step_accounting_observations", 0)
            ),
            optimizer_step_accounting_failures=int(
                getattr(getattr(self, "run_manifest", None), "optimizer_step_accounting_failures", 0)
            ),
            optimizer_step_accounting_unverifiable=int(
                getattr(getattr(self, "run_manifest", None), "optimizer_step_accounting_unverifiable", 0)
            ),
            segment_count_invariants_passed=bool(
                getattr(getattr(self, "run_manifest", None), "segment_count_invariants_passed", False)
            ),
            stored_old_log_probs_used=bool(
                getattr(getattr(self, "run_manifest", None), "stored_old_log_probs_used", False)
            ),
            rollout_scorer_weights_verified=bool(
                getattr(getattr(self, "run_manifest", None), "rollout_scorer_weights_verified", False)
            ),
            no_truncation=bool(
                getattr(getattr(self, "run_manifest", None), "no_truncation", False)
            ),
            replay_age_uses_rollout_iteration=bool(
                getattr(getattr(self, "run_manifest", None), "replay_age_uses_rollout_iteration", False)
            ),
            unique_tree_ids_verified=bool(
                getattr(getattr(self, "run_manifest", None), "unique_tree_ids_verified", False)
            ),
        )

    def _save_live_state_best_effort(self, metrics: Dict[str, Any]) -> None:
        try:
            save_live_state(
                self.config.trainer.default_local_dir, self._current_live_state()
            )
        except Exception as exc:
            metrics["vdra/live_state_save_failed"] = 1.0
            _LOGGER.warning("Failed to persist the VDRA live state: %s", exc)

    def _merge_live_manifest_provenance(self, live: GearTreeLiveState) -> None:
        m = self.run_manifest
        for field in (
            "group_integrity_failures",
            "segment_count_failures",
            "replay_batch_failures",
            "parent_split_count",
            "tree_split_count",
            "optimizer_step_accounting_observations",
            "optimizer_step_accounting_failures",
            "optimizer_step_accounting_unverifiable",
        ):
            setattr(m, field, max(int(getattr(m, field)), int(getattr(live, field))))
        if bool(getattr(live, "_provenance_booleans_present", True)):
            for field in (
                "segment_count_invariants_passed",
                "stored_old_log_probs_used",
                "rollout_scorer_weights_verified",
                "no_truncation",
                "replay_age_uses_rollout_iteration",
                "unique_tree_ids_verified",
            ):
                setattr(
                    m, field, bool(getattr(m, field)) and bool(getattr(live, field))
                )
        m.optimizer_step_accounting_valid = (
            int(m.optimizer_step_accounting_observations) > 0
            and int(m.optimizer_step_accounting_failures) == 0
            and int(m.optimizer_step_accounting_unverifiable) == 0
        )

    def _merge_pending_live_state(self) -> None:
        live = getattr(self, "_pending_live_state", None)
        if live is None:
            return
        self.rollout_iteration = int(live.rollout_iteration)
        self.num_optimizer_steps_total = int(live.num_optimizer_steps_total)
        self.successful_actor_updates = int(live.successful_actor_updates)
        self.postponed_updates = int(live.postponed_updates)
        self.failed_updates = int(live.failed_updates)
        self.skipped_zero_gradient_updates = int(
            live.skipped_zero_gradient_updates
        )
        self._set_nonprogress_counter(live.consecutive_nonprogress_iterations)
        if str(live.last_iteration_status) in VALID_ITERATION_STATUSES:
            self._set_iteration_status(str(live.last_iteration_status))
        self._merge_live_manifest_provenance(live)
        self._stamp_manifest_iteration_state()
        self._pending_live_state = None

    def _save_manifest_best_effort(self, metrics: Dict[str, Any]) -> None:
        try:
            self._save_manifest(self.run_manifest)
        except Exception as exc:
            metrics["vdra/manifest_save_failed"] = 1.0
            _LOGGER.warning("Failed to persist the VDRA manifest: %s", exc)

    def _append_timing_row(
        self,
        timing_path: str,
        row: Mapping[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        try:
            with open(timing_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(row)) + "\n")
        except Exception as exc:
            metrics["vdra/timing_write_failed"] = 1.0
            _LOGGER.warning("Failed to write the VDRA timing row: %s", exc)

    def _record_iteration_on_manifest(
        self,
        *,
        selected_edges: int,
        sample_stats: Dict[str, Any],
        actual_optimizer_steps: int,
        record_optimizer_observation: bool = True,
    ) -> None:
        """Stamp per-iteration observed facts on the manifest."""
        m = self.run_manifest
        self._stamp_manifest_iteration_state()
        m.optimizer_steps_last_iteration = int(actual_optimizer_steps)
        m.selected_edges_last_iteration = int(selected_edges)
        m.unique_questions_last_iteration = int(
            sample_stats.get("buffer/unique_questions", 0) or 0
        )
        m.mean_edge_age_last_iteration = float(
            sample_stats.get("buffer/mean_edge_age", 0.0) or 0.0
        )
        m.max_edge_age_last_iteration = int(
            sample_stats.get("buffer/max_edge_age", 0) or 0
        )
        m.per_question_selected_count_max_last_iteration = int(
            sample_stats.get("buffer/edges_per_question_max", 0) or 0
        )
        hist = sample_stats.get("buffer/edge_age_histogram") or {}
        if isinstance(hist, Mapping):
            m.edge_age_histogram_last_iteration = {
                int(k): int(v) for k, v in hist.items()
            }
        expected = self._resolve_expected_optimizer_steps(int(selected_edges))
        if expected is None:
            return
        expected = int(expected)
        m.expected_optimizer_steps_last_iteration = expected
        if record_optimizer_observation:
            m.optimizer_step_accounting_observations += 1
            if int(actual_optimizer_steps) != expected:
                m.optimizer_step_accounting_failures += 1
            m.optimizer_step_accounting_valid = (
                m.optimizer_step_accounting_observations > 0
                and m.optimizer_step_accounting_failures == 0
                and m.optimizer_step_accounting_unverifiable == 0
            )
        elif m.optimizer_step_accounting_observations == 0:
            m.optimizer_step_accounting_valid = False

    def _manifest_path(self) -> str:
        return os.path.join(
            self.config.trainer.default_local_dir, "vdra_run_manifest.json"
        )

    def _stamp_manifest_iteration_state(self) -> None:
        if getattr(self, "run_manifest", None) is None:
            return
        self.run_manifest.rollout_iteration = int(getattr(self, "rollout_iteration", 0))
        self.run_manifest.global_step = int(getattr(self, "global_steps", 0))
        self.run_manifest.num_optimizer_steps_total = int(
            getattr(self, "num_optimizer_steps_total", 0)
        )
        if hasattr(self, "replay_buffer") and self.replay_buffer is not None:
            self.run_manifest.resolved_max_edges_per_question_per_iteration = int(
                self.replay_buffer.resolved_max_edges_per_question_per_iteration
            )

    def _save_manifest(
        self, manifest: RunManifest, *, path: str | Path | None = None
    ) -> None:
        if manifest is self.run_manifest:
            self._stamp_manifest_iteration_state()
        target = Path(path) if path is not None else Path(self._manifest_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest.save(target)

    def _update_manifest_from_generated_edges(
        self,
        manifest: RunManifest,
        generated_edges: List[Dict[str, Any]],
        *,
        strict: bool,
        construction_summaries: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """PLAN.md P0.B: construction-stage validation of the COMPLETE
        generated batch, before replay insertion. ``construction_summaries``
        carry the pre-filter facts of parents/trees whose every child had
        exactly zero advantage. Under sparse execution those children still
        appear in ``generated_edges`` as metadata-only logical slots (no
        TRAINABLE rows); the summaries remain authoritative either way."""
        return update_manifest_from_generated_edges(
            manifest,
            generated_edges,
            strict=strict,
            construction_summaries=construction_summaries,
        )

    def _update_manifest_from_replay_batch(
        self,
        manifest: RunManifest,
        sampled_edges: List[Dict[str, Any]],
        *,
        strict: bool,
    ) -> Dict[str, Any]:
        """PLAN.md P0.B: row-local validation of the sampled replay batch.
        Partial trees/parent groups are legal here by design."""
        replay_buffer = getattr(self, "replay_buffer", None)
        kwargs: Dict[str, Any] = {}
        if replay_buffer is not None:
            kwargs = {
                "target_edges_per_iteration": int(
                    replay_buffer.target_edges_per_iteration
                ),
                "max_edge_age_iterations": int(
                    replay_buffer.max_edge_age_iterations
                ),
                "current_rollout_iteration": int(self.rollout_iteration),
            }
            # PLAN.md P0.A/P0.B: the per-question cap is an EDGE-sampler
            # contract. The complete_tree ablation packs whole trees and may
            # legitimately exceed it for a single-tree question, so the cap
            # is only asserted on the canonical edge path.
            if str(replay_buffer.replay_sampling_unit) == "edge":
                kwargs["max_edges_per_question_per_iteration"] = int(
                    replay_buffer.resolved_max_edges_per_question_per_iteration
                )
        return update_manifest_from_replay_batch(
            manifest, sampled_edges, strict=strict, **kwargs
        )

    def _execute_reserved_actor_update(
        self,
        replay_buffer,
        reservation,
        sampled_edges: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        *,
        manifest_strict: bool,
    ):
        """PLAN.md M2: transaction stages for a reserved replay batch.

        validation failure     -> rollback reservation, no counter mutation
        tensorization failure  -> rollback reservation, no counter mutation
        actor RPC exception    -> rollback reservation, failed_updates += 1
        actor RPC success      -> return; commit + outer counters stay in fit()

        An actor RPC may mutate model parameters before raising; replay
        rollback only restores buffer state and never claims to undo a model
        update.
        """
        try:
            # PLAN.md P0.B: sampled replay batches get ROW-LOCAL checks only
            # (edge-level replay legally splits trees and parent groups). Run
            # BEFORE tensorization and actor update so a broken reservation
            # cannot be converted into model inputs.
            replay_batch_metrics = self._update_manifest_from_replay_batch(
                self.run_manifest, sampled_edges, strict=manifest_strict
            )
            metrics.update(replay_batch_metrics)
            edge_batch = self._edges_to_update_batch(sampled_edges, metrics)
        except Exception:
            replay_buffer.rollback(reservation)
            # PLAN.md §5: validation/tensorization failed BEFORE the actor RPC.
            self._set_iteration_status(ITERATION_STATUS_FAILED_BEFORE_ACTOR)
            raise
        if edge_batch is None:
            # PLAN.md §1.3: every logical batch of this reservation is
            # all-zero — an EXPLICIT skipped update, not an RPC failure. The
            # model is never touched and neither global_step nor the
            # scheduler advances (update_actor is never called). Per the M2
            # contract this helper does NOT commit or mutate outer counters;
            # the fit loop consumes the reservation and records the skip.
            return None, None, 0.0
        # PLAN.md P0.J: strict tensorization refuses truncation with a
        # ValueError, so REACHING this line is the observed no-truncation
        # event for this batch.
        self.run_manifest.no_truncation = True
        self.run_manifest.extras["no_truncation"] = True

        t0 = time.time()
        # P0.9: RayGearTreeTrainer never trains a critic, so there is no
        # critic warmup to gate the actor update on. Always update.
        try:
            actor_output = self.actor_rollout_wg.update_actor(edge_batch)
        except Exception:
            replay_buffer.rollback(reservation)
            self.failed_updates += 1
            # PLAN.md §5: the actor RPC itself failed — model parameters may
            # already have changed, which replay rollback cannot undo.
            self._set_iteration_status(ITERATION_STATUS_ACTOR_FAILED)
            raise
        return edge_batch, actor_output, time.time() - t0

    @staticmethod
    def _flatten_int_list(x) -> List[int]:
        out: List[int] = []
        if x is None:
            return out
        if isinstance(x, (list, tuple)):
            for item in x:
                out.extend(RayGearTreeTrainer._flatten_int_list(item))
        else:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out

    def _finalize_successful_actor_update(
        self,
        replay_buffer,
        reservation,
        actor_output,
        sampled_edges: List[Dict[str, Any]],
        sample_stats: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """PLAN.md M2 (item 3): commit + outer counters, then diagnostics.

        Once ``update_actor`` returns successfully the model may already have
        changed, so the reservation commit and the OUTER-update counters
        (``successful_actor_updates``, ``global_step``) are secured
        UNCONDITIONALLY first. Only afterwards are the DIAGNOSTIC actor
        metrics parsed. A missing or malformed ``meta_info["metrics"]`` must
        NOT rollback replay and must NOT undo the outer update — it only marks
        ``optimizer_step_accounting_valid`` invalid and logs a diagnostic
        failure. ``num_optimizer_steps_total`` is a diagnostic and advances
        only when parsing succeeds.
        """
        removed = replay_buffer.commit(reservation)
        metrics["buffer/removed_edges"] = float(len(removed))
        metrics["buffer/size_after"] = float(len(replay_buffer))
        # PLAN.md §5/§6: this iteration DID run the actor. The status write
        # also clears the derived actor_update_skipped compatibility flag.
        self._set_iteration_status(ITERATION_STATUS_UPDATED)
        # PLAN.md §2: only a successful actor update breaks a skip streak.
        self._set_nonprogress_counter(0)
        self.successful_actor_updates += 1
        # PLAN.md M1 three-counter contract: global_step advances by 1 per
        # successful outer update_actor call (the host VERL
        # loop/checkpoint/save/eval unit). The internal PPO optimizer-batch
        # count accumulates separately below as an observational diagnostic.
        self.global_steps += 1
        metrics["training/successful_actor_updates"] = float(
            self.successful_actor_updates
        )

        metrics_parse_ok = True
        n_optim_steps = 0
        actor_used_stored_old_log_probs = False
        try:
            meta_info = getattr(actor_output, "meta_info", None)
            if not isinstance(meta_info, Mapping) or "metrics" not in meta_info:
                raise KeyError("actor_output.meta_info has no 'metrics' mapping")
            actor_metrics_meta = meta_info["metrics"]
            if not isinstance(actor_metrics_meta, Mapping):
                raise TypeError(
                    "actor meta_info['metrics'] is "
                    f"{type(actor_metrics_meta).__name__}, expected a mapping"
                )
            step_ints = self._flatten_int_list(
                actor_metrics_meta.get("actor/num_optimizer_steps", None)
            )
            if not step_ints:
                # PLAN.md §3: NEVER invent a step count. The outer update is
                # already committed and must stay committed, but the internal
                # optimizer-step accounting becomes UNKNOWN — defaulting to 1
                # would fabricate a diagnostic and could make a broken run
                # look correctly accounted.
                raise KeyError(
                    "actor meta_info['metrics'] has no "
                    "'actor/num_optimizer_steps'; optimizer-step accounting "
                    "is unknown for this update"
                )
            n_optim_steps = max(step_ints)
            # PLAN.md P0.J: OBSERVED stored-old-log-prob use, reported by the
            # actor itself. The manifest bit flips only from this metric —
            # never from tensor presence alone.
            used_stored_ints = self._flatten_int_list(
                actor_metrics_meta.get("actor/used_stored_old_log_probs")
            )
            actor_used_stored_old_log_probs = bool(
                used_stored_ints and min(used_stored_ints) >= 1
            )
            metrics.update(reduce_metrics(actor_metrics_meta))
        except Exception as exc:  # diagnostic-only: never fatal here
            metrics_parse_ok = False
            metrics["vdra/actor_metrics_parse_failed"] = 1.0
            _LOGGER.warning(
                "actor update succeeded but its diagnostic metrics could not "
                "be parsed (%s); committing the outer update and marking "
                "optimizer-step accounting invalid.",
                exc,
            )

        if metrics_parse_ok:
            self.optimizer_steps_this_iteration = int(n_optim_steps)
            self.num_optimizer_steps_total += int(n_optim_steps)
        else:
            # The outer update stands; only the internal step count is unknown.
            self.optimizer_steps_this_iteration = 0

        # PLAN.md P0.J: manifest bit only from the actor-observed metric —
        # never inferred from tensor presence.
        if actor_used_stored_old_log_probs:
            self.run_manifest.stored_old_log_probs_used = True
            self.run_manifest.extras["stored_old_log_probs_used"] = True
        # PLAN.md P0.J: at least one successful update with no integrity
        # failures flips ONLY the invariant bit matching the configured loss
        # mode — segment-mean and node-balanced are different claims.
        if self.run_manifest.group_integrity_failures == 0:
            loss_mode = str(
                self.config.actor_rollout_ref.actor.policy_loss.get(
                    "loss_mode", "vdra_segment_mean_ppo"
                )
            )
            if loss_mode == "vdra_node_balanced_ppo":
                self.run_manifest.record_node_balanced_invariant_pass()
            else:
                self.run_manifest.record_segment_invariant_pass()
        # PLAN.md P0.7: stamp observed counters + replay diagnostics.
        self._record_iteration_on_manifest(
            selected_edges=len(sampled_edges),
            sample_stats=sample_stats,
            actual_optimizer_steps=int(n_optim_steps),
            record_optimizer_observation=metrics_parse_ok,
        )
        # PLAN.md M2/M4: the internal optimizer-step count is a DIAGNOSTIC,
        # not the outer training unit. A parse failure or a count/expected
        # mismatch marks ``optimizer_step_accounting_valid`` invalid but never
        # crashes a run whose actor already updated the model.
        if not metrics_parse_ok:
            self.run_manifest.optimizer_step_accounting_unverifiable += 1
            self.run_manifest.optimizer_step_accounting_valid = False
            return
        # PLAN.md §8: ONE authoritative expectation. On the canonical sparse
        # path it counts only TRAINABLE logical batches (a batch skipped for
        # all_zero_advantage or zero_active_tokens legitimately performs no
        # optimizer step); the selected-slot formula would over-count and mark
        # a valid mixed update as an accounting mismatch.
        expected_steps = self._resolve_expected_optimizer_steps(len(sampled_edges))
        if expected_steps is None:
            return
        metrics["training/expected_optimizer_steps"] = float(expected_steps)
        if int(n_optim_steps) != expected_steps:
            self.run_manifest.optimizer_step_accounting_valid = False
            metrics["vdra/optimizer_step_accounting_mismatch"] = 1.0
            _LOGGER.warning(
                "actor performed %d optimizer steps but %d were expected "
                "(trainable logical batches x ppo_epochs); marking "
                "optimizer-step accounting invalid (diagnostic).",
                int(n_optim_steps),
                expected_steps,
            )

    def _set_iteration_status(self, status: str) -> str:
        """PLAN.md §5: record what THIS iteration did.

        ``actor_update_skipped`` is DERIVED here so the boolean can never
        disagree with the authoritative status.
        """
        if status not in VALID_ITERATION_STATUSES:
            raise ValueError(
                f"unknown iteration status {status!r}; expected one of "
                f"{VALID_ITERATION_STATUSES}"
            )
        self.run_manifest.last_iteration_status = status
        self.run_manifest.actor_update_skipped = status in ZERO_SIGNAL_SKIP_STATUSES
        return status

    @staticmethod
    def _zero_signal_skip_status(metrics: Mapping[str, Any]) -> str:
        """Classify a fully skipped reservation by WHY every batch skipped."""
        n_all_zero = float(metrics.get("vdra/all_zero_advantage_logical_batches", 0.0))
        n_zero_active = float(
            metrics.get("vdra/zero_active_token_logical_batches", 0.0)
        )
        if n_all_zero > 0 and n_zero_active > 0:
            return ITERATION_STATUS_MIXED_ZERO_SIGNAL_SKIPPED
        if n_zero_active > 0:
            return ITERATION_STATUS_ZERO_ACTIVE_SKIPPED
        return ITERATION_STATUS_ALL_ZERO_SKIPPED

    def _resolve_max_nonprogress_iterations(self) -> int:
        gt = self._gear_tree_config()
        raw_new = gt.get("max_consecutive_nonprogress_iterations", None)
        raw_old = gt.get("max_consecutive_skipped_updates", None)
        if raw_new is not None and raw_old is not None:
            if int(raw_new) != int(raw_old):
                raise ValueError(
                    "gear_tree.max_consecutive_skipped_updates is a deprecated "
                    "alias for gear_tree.max_consecutive_nonprogress_iterations; "
                    "setting both to different values is invalid (PLAN.md §3)."
                )
        raw = raw_new if raw_new is not None else raw_old
        return int(raw) if raw is not None else 0

    def _set_nonprogress_counter(self, value: int) -> None:
        value = int(value)
        self.consecutive_nonprogress_iterations = value
        # Deprecated one-release compatibility alias.
        self.consecutive_skipped_updates = value

    def _increment_nonprogress_counter(self) -> None:
        self._set_nonprogress_counter(
            int(getattr(self, "consecutive_nonprogress_iterations", 0)) + 1
        )

    def _enforce_livelock_guards(self, metrics: Dict[str, Any]) -> None:
        """PLAN.md §2/§3: abort a run stuck in nonprogress iterations.

        A nonprogress iteration is any path that does not advance
        ``global_step``: zero-signal skip, postponed reservation, or empty
        reservation. Saves whatever state it can, then raises with full
        diagnostic context.
        """
        limit = self._resolve_max_nonprogress_iterations()
        current = int(getattr(self, "consecutive_nonprogress_iterations", 0))
        if limit <= 0 or current < limit:
            return
        self._save_manifest_best_effort(metrics)
        self._save_live_state_best_effort(metrics)
        try:
            self._save_vdra_checkpoint_bundle()
        except Exception as exc:
            _LOGGER.warning("could not persist a checkpoint before abort: %s", exc)
        replay_buffer = getattr(self, "replay_buffer", None)
        buffer_size = len(replay_buffer) if replay_buffer is not None else 0
        raise RuntimeError(
            "VDRA aborting: "
            f"{current} consecutive nonprogress iterations reached the "
            "configured limit "
            f"gear_tree.max_consecutive_nonprogress_iterations={limit}. "
            "A nonprogress iteration performs no optimizer step, so "
            "global_step cannot advance and the run would live-lock. "
            f"global_step={self.global_steps}, "
            f"rollout_iteration={self.rollout_iteration}, "
            f"last_iteration_status={self.run_manifest.last_iteration_status!r}, "
            f"consecutive_nonprogress_iterations={current}, "
            f"buffer_size={buffer_size}, "
            f"postponed_updates={self.postponed_updates}, "
            "skipped_zero_gradient_updates="
            f"{self.skipped_zero_gradient_updates}, "
            "all_zero_advantage_logical_batches="
            f"{metrics.get('vdra/all_zero_advantage_logical_batches', 0.0)}, "
            "zero_active_token_logical_batches="
            f"{metrics.get('vdra/zero_active_token_logical_batches', 0.0)}. "
            "Check the advantage/reward signal, replay divisibility, or raise "
            "the limit (<= 0 or null disables the guard)."
        )

    def _log_livelock_counters(self, metrics: Dict[str, Any]) -> None:
        """PLAN.md §2/§3: make the guard state visible on every iteration."""
        limit = self._resolve_max_nonprogress_iterations()
        raw_iters = self._gear_tree_config().get("max_rollout_iterations", None)
        current = float(getattr(self, "consecutive_nonprogress_iterations", 0))
        metrics["training/consecutive_nonprogress_iterations"] = current
        metrics["training/max_consecutive_nonprogress_iterations"] = float(limit)
        # Deprecated aliases retained for one release.
        metrics["training/consecutive_skipped_updates"] = current
        metrics["training/max_consecutive_skipped_updates"] = float(limit)
        metrics["training/max_rollout_iterations"] = float(
            int(raw_iters) if raw_iters is not None else 0
        )

    def _record_nonprogress_iteration(
        self,
        *,
        status: str,
        timing_path: str,
        t_gen: float,
        sample_stats: Mapping[str, Any],
        metrics: Dict[str, Any],
        loop_start: float,
        cumulative_train_seconds: float,
    ) -> float:
        """Record a no-progress iteration with one shared code path."""
        self._set_iteration_status(status)
        self._increment_nonprogress_counter()
        updated_cum_train = float(cumulative_train_seconds) + float(t_gen)
        selected_edges = int(
            sample_stats.get(
                "buffer/selected_edges",
                sample_stats.get("selected_edges", sample_stats.get("selected_count", 0)),
            )
            or 0
        )
        self.optimizer_steps_this_iteration = 0
        self._expected_optimizer_steps = 0
        self._record_iteration_on_manifest(
            selected_edges=selected_edges,
            sample_stats=dict(sample_stats),
            actual_optimizer_steps=0,
            record_optimizer_observation=False,
        )
        replay_buffer = getattr(self, "replay_buffer", None)
        buffer_size = len(replay_buffer) if replay_buffer is not None else 0
        actor_update_skipped = status in ZERO_SIGNAL_SKIP_STATUSES
        row = {
            "step": int(self.global_steps),
            "rollout_iteration": int(self.rollout_iteration),
            "last_iteration_status": status,
            "actor_update_skipped": actor_update_skipped,
            "timing/generation_seconds": float(t_gen),
            "timing/update_seconds": 0.0,
            "timing/train_total_seconds": float(t_gen),
            "timing/cumulative_train_seconds": updated_cum_train,
            "timing/wall_seconds": time.time() - loop_start,
            "training/global_step": float(self.global_steps),
            "training/expected_optimizer_steps": 0.0,
            "training/optimizer_steps_this_iteration": 0.0,
            "training/num_optimizer_steps_total": float(
                self.num_optimizer_steps_total
            ),
            "training/successful_actor_updates": float(
                self.successful_actor_updates
            ),
            "training/postponed_updates": float(self.postponed_updates),
            "training/failed_updates": float(self.failed_updates),
            "training/skipped_zero_gradient_updates": float(
                self.skipped_zero_gradient_updates
            ),
            "training/consecutive_nonprogress_iterations": float(
                self.consecutive_nonprogress_iterations
            ),
            "training/consecutive_skipped_updates": float(
                self.consecutive_nonprogress_iterations
            ),
            "buffer/size_after": float(buffer_size),
            "buffer/selected_edges": float(selected_edges),
        }
        if "buffer/mean_edge_age" in sample_stats:
            row["train/mean_edge_age"] = float(
                sample_stats.get("buffer/mean_edge_age", 0.0) or 0.0
            )
        if "buffer/max_edge_age" in sample_stats:
            row["train/max_edge_age"] = float(
                sample_stats.get("buffer/max_edge_age", 0) or 0
            )
        metrics.update(row)
        metrics["training/last_iteration_status"] = status
        metrics["vdra/skipped_zero_gradient_updates"] = float(
            self.skipped_zero_gradient_updates
        )
        self._append_timing_row(timing_path, row, metrics)
        self._save_manifest_best_effort(metrics)
        self._save_live_state_best_effort(metrics)
        self._log_livelock_counters(metrics)
        self._enforce_livelock_guards(metrics)
        return updated_cum_train

    def _rollout_iteration_budget_exhausted(self) -> bool:
        """PLAN.md §2: stop after ``max_rollout_iterations`` even when
        ``global_step`` never reached ``total_training_steps`` (which is what
        happens when iterations keep getting skipped or postponed)."""
        raw = self._gear_tree_config().get("max_rollout_iterations", None)
        limit = int(raw) if raw is not None else 0
        if limit <= 0:
            return False
        return int(self.rollout_iteration) >= limit

    def _resolve_expected_optimizer_steps(self, selected_count: int):
        """Authoritative expected internal optimizer-step count (PLAN.md §8).

        Canonical sparse path: ``trainable_logical_batches * ppo_epochs``,
        stamped by ``_edges_to_update_batch`` from the logical-batch statuses.
        Non-canonical paths keep the legacy selected-slot formula, which can
        never override the canonical value. Returns ``None`` when no
        expectation is well defined (legacy path with an indivisible count).
        """
        canonical = getattr(self, "_expected_optimizer_steps", None)
        if canonical is not None:
            return int(canonical)
        ppo_mini = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        ppo_epochs = int(self.config.actor_rollout_ref.actor.get("ppo_epochs", 1))
        if int(selected_count) % max(ppo_mini, 1) != 0:
            return None
        return expected_optimizer_steps(
            selected_count=int(selected_count),
            ppo_mini_batch_size=ppo_mini,
            ppo_epochs=ppo_epochs,
        )

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.rollout_iteration = 0
        self.successful_actor_updates = 0
        self.postponed_updates = 0
        self.failed_updates = 0
        # PLAN.md §1.3: observational counter for explicit all-zero skipped
        # updates (session-local diagnostic; never drives the outer loop).
        self.skipped_zero_gradient_updates = 0
        # PLAN.md §3: consecutive nonprogress iterations for the anti-livelock
        # guard. Reset only by a SUCCESSFUL actor update.
        self._set_nonprogress_counter(0)
        # PLAN.md M1: keep rollout iteration, outer global_step, and
        # internal optimizer-step diagnostics as distinct counters.
        self.optimizer_steps_this_iteration = 0
        self.num_optimizer_steps_total = 0
        self._load_checkpoint()
        self._validate_replay_startup()
        replay_resume_metrics = self._restore_or_init_replay_buffer()
        replay_buffer = self._ensure_replay_buffer()

        # PLAN.md P0.N8: preserve a checkpoint manifest on resume so failure
        # counters and invariant verdicts cannot be reset by rebuilding it.
        self.run_manifest = self._load_or_build_run_manifest()
        self._merge_pending_live_state()
        manifest_strict = bool(
            (self.config.get("tree_policy") or {}).get(
                "strict_group_integrity", False
            )
        )
        # PLAN.md P0.A: the reservation path is selected by the configured
        # sampling unit ONLY. Strictness controls validation, never sampling.
        replay_sampling_unit = str(
            self._replay_config().get("replay_sampling_unit", "edge")
        )

        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        timing_path = os.path.join(self.config.trainer.default_local_dir, "training_timing.jsonl")
        loop_start = time.time()
        cum_train = 0.0

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
        if replay_resume_metrics:
            logger.log(data=replay_resume_metrics, step=self.global_steps)

        test_freq = self.config.trainer.get("test_freq", -1)
        save_freq = self.config.trainer.get("save_freq", -1)
        # PLAN.md P0.E: save/eval fire on CROSSED thresholds, never on
        # modulo. Derived from the (possibly restored) global_steps so resume
        # never re-fires an already-passed threshold.
        self.next_eval_step = initial_next_threshold(self.global_steps, test_freq)
        self.next_save_step = initial_next_threshold(self.global_steps, save_freq)

        while self.global_steps < self.total_training_steps:
            # PLAN.md §2: stop on the rollout-iteration budget even though
            # global_step has not reached total_training_steps — that is
            # exactly the state a skipped/postponed run gets stuck in.
            if self._rollout_iteration_budget_exhausted():
                _LOGGER.warning(
                    "stopping: rollout_iteration=%d reached "
                    "gear_tree.max_rollout_iterations while global_step=%d < "
                    "total_training_steps=%d.",
                    self.rollout_iteration,
                    self.global_steps,
                    self.total_training_steps,
                )
                break
            for batch_dict in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    break
                if self._rollout_iteration_budget_exhausted():
                    break
                self.rollout_iteration += 1
                # PLAN.md P0.3: reset per-iteration optimizer-step counter so
                # postponed / failed iterations do not carry the previous
                # iteration's count forward.
                self.optimizer_steps_this_iteration = 0
                # PLAN.md §8: the canonical expectation is stamped freshly by
                # _edges_to_update_batch each iteration; clear it so a stale
                # value can never be attributed to a later update.
                self._expected_optimizer_steps = None
                # PLAN.md P0.E: pre-update value for threshold crossing and
                # unambiguous before/after logging.
                global_step_before_update = int(self.global_steps)
                metrics: Dict[str, Any] = {}
                self._set_iteration_status(ITERATION_STATUS_RUNNING)
                try:
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    gen_batch = self._get_gen_batch(batch)

                    t0 = time.time()
                    new_edges = self._generate_tree_edges(gen_batch)
                    t_gen = time.time() - t0
                    metrics.update(getattr(self, "_last_rollout_reward_parse_metrics", {}) or {})
                    # PLAN.md P0.B: full-tree CONSTRUCTION validation runs on the
                    # complete generated batch, before replay insertion. This is
                    # the only stage that may require complete parent groups and
                    # full-tree queue identities.
                    construction_summaries = list(
                        getattr(self, "_last_construction_summaries", []) or []
                    )
                    if new_edges or construction_summaries:
                        construction_metrics = (
                            self._update_manifest_from_generated_edges(
                                self.run_manifest,
                                new_edges,
                                strict=manifest_strict,
                                construction_summaries=construction_summaries,
                            )
                        )
                        metrics.update(construction_metrics)
                    # PLAN.md P0.2: replay age is stamped in rollout iterations,
                    # not optimizer steps. `self.rollout_iteration` has already been
                    # incremented for this iteration above.
                    replay_buffer.add(
                        new_edges,
                        generation_rollout_iteration=self.rollout_iteration,
                        policy_snapshot_id=self._current_policy_snapshot_id(),
                    )
                    # PLAN.md P0.A: canonical strict main uses EDGE-level
                    # reservation. Complete-tree replay is reachable only via the
                    # explicit `replay_sampling_unit: complete_tree` ablation and
                    # is never selected by strict_group_integrity.
                    reservation = reserve_replay_edges(
                        replay_buffer,
                        replay_sampling_unit=replay_sampling_unit,
                        current_rollout_iteration=self.rollout_iteration,
                    )
                    sampled_edges = [dict(edge) for edge in reservation.edges]
                    sample_stats = reservation.stats
                    metrics.update({k: v for k, v in sample_stats.items() if k != "removed_edge_ids"})
                    metrics["buffer/new_edges"] = len(new_edges)
                    metrics["buffer/postponed_update"] = 0.0
                    metrics["training/rollout_iteration"] = float(self.rollout_iteration)
                    # PLAN.md P0.E: never log a pre-update value under an
                    # ambiguous name; the final training/global_step is always the
                    # post-update value set in the timing block below.
                    metrics["training/global_step_before_update"] = float(
                        global_step_before_update
                    )
                    metrics["training/successful_actor_updates"] = float(self.successful_actor_updates)
                    if not sampled_edges:
                        stats = dict(sample_stats)
                        stats["buffer/selected_edges"] = 0
                        cum_train = self._record_nonprogress_iteration(
                            status=ITERATION_STATUS_NO_SAMPLE,
                            timing_path=timing_path,
                            t_gen=t_gen,
                            sample_stats=stats,
                            metrics=metrics,
                            loop_start=loop_start,
                            cumulative_train_seconds=cum_train,
                        )
                        logger.log(data=metrics, step=self.global_steps)
                        continue
                    if self._should_postpone_sampled_update(sampled_edges):
                        replay_buffer.rollback(reservation)
                        self.postponed_updates += 1
                        metrics["buffer/postponed_update"] = 1.0
                        metrics["training/postponed_updates"] = float(self.postponed_updates)
                        metrics["buffer/size_after"] = float(len(replay_buffer))
                        stats = dict(sample_stats)
                        stats["buffer/selected_edges"] = len(sampled_edges)
                        cum_train = self._record_nonprogress_iteration(
                            status=ITERATION_STATUS_POSTPONED,
                            timing_path=timing_path,
                            t_gen=t_gen,
                            sample_stats=stats,
                            metrics=metrics,
                            loop_start=loop_start,
                            cumulative_train_seconds=cum_train,
                        )
                        logger.log(data=metrics, step=self.global_steps)
                        continue

                    # PLAN.md M2: validation, tensorization, and the actor RPC run
                    # inside one transaction helper — any failure before a
                    # successful RPC rolls the reservation back.
                    edge_batch, actor_output, t_update = (
                        self._execute_reserved_actor_update(
                            replay_buffer,
                            reservation,
                            sampled_edges,
                            metrics,
                            manifest_strict=manifest_strict,
                        )
                    )
                    if edge_batch is None:
                        # PLAN.md §1.3: explicit skipped update (all-zero
                        # reservation). Consume the reservation before
                        # recording the nonprogress iteration.
                        removed = replay_buffer.commit(reservation)
                        metrics["buffer/removed_edges"] = float(len(removed))
                        metrics["buffer/size_after"] = float(len(replay_buffer))
                        self.skipped_zero_gradient_updates += 1
                        metrics["vdra/skipped_zero_gradient_updates"] = float(
                            self.skipped_zero_gradient_updates
                        )
                        skip_status = self._zero_signal_skip_status(metrics)
                        stats = dict(sample_stats)
                        stats["buffer/selected_edges"] = len(sampled_edges)
                        cum_train = self._record_nonprogress_iteration(
                            status=skip_status,
                            timing_path=timing_path,
                            t_gen=t_gen,
                            sample_stats=stats,
                            metrics=metrics,
                            loop_start=loop_start,
                            cumulative_train_seconds=cum_train,
                        )
                        logger.log(data=metrics, step=self.global_steps)
                        continue
                    actor_updated = True
                    # PLAN.md M2 (item 3): commit + outer counters are secured
                    # unconditionally on actor success; the diagnostic actor
                    # metrics are parsed afterwards and can never revert them.
                    self._finalize_successful_actor_update(
                        replay_buffer,
                        reservation,
                        actor_output,
                        sampled_edges,
                        sample_stats,
                        metrics,
                    )
                    # PLAN.md §5/§9: dummy rows are collective-safety padding and
                    # count in NO objective, replay, or diagnostic metric. Report
                    # the REAL (non-dummy) row and token counts explicitly.
                    _is_dummy = edge_batch.batch.get("is_dummy")
                    if _is_dummy is None:
                        real_row_mask = torch.ones(
                            len(edge_batch), dtype=torch.bool
                        )
                        dummy_rows = 0
                    else:
                        real_row_mask = ~_is_dummy.bool()
                        dummy_rows = int(_is_dummy.sum())
                    trainable_tensor_rows = int(real_row_mask.sum())
                    real_response_tokens = int(
                        edge_batch.batch["response_mask"][real_row_mask].sum()
                    )
                    # PLAN.md P0.2: age uses rollout_iteration (never global_step).
                    ages = [
                        self.rollout_iteration
                        - int(
                            edge.get(
                                "generation_rollout_iteration",
                                edge.get(
                                    "generation_step", self.rollout_iteration
                                ),
                            )
                        )
                        for edge in sampled_edges
                    ]
                    cum_train += t_gen + t_update
                    # PLAN.md §8: the SAME authoritative expectation the manifest
                    # uses — never the selected-slot formula on the sparse path.
                    expected_optim_steps = self._resolve_expected_optimizer_steps(
                        len(sampled_edges)
                    )
                    self._log_livelock_counters(metrics)
                    timing = {
                        "step": self.global_steps,
                        "rollout_iteration": self.rollout_iteration,
                        # PLAN.md §5: authoritative record of what this iteration
                        # did, plus the derived compatibility flag.
                        "last_iteration_status": self.run_manifest.last_iteration_status,
                        "actor_update_skipped": self.run_manifest.actor_update_skipped,
                        "training/consecutive_nonprogress_iterations": float(
                            self.consecutive_nonprogress_iterations
                        ),
                        "training/consecutive_skipped_updates": float(
                            self.consecutive_nonprogress_iterations
                        ),
                        "timing/generation_seconds": t_gen,
                        "timing/update_seconds": t_update,
                        "timing/train_total_seconds": t_gen + t_update,
                        "timing/cumulative_train_seconds": cum_train,
                        "timing/wall_seconds": time.time() - loop_start,
                        # PLAN.md §5: explicit, dummy-free accounting. The tensor
                        # batch includes dummy padding and EXCLUDES zero logical
                        # slots, so its raw length is not an edge count.
                        "train/logical_slots": float(len(sampled_edges)),
                        "train/trainable_tensor_rows": float(trainable_tensor_rows),
                        "train/dummy_rows": float(dummy_rows),
                        "train/real_response_tokens": float(real_response_tokens),
                        # Legacy key, explicitly labeled as the raw tensor-row
                        # count (dummy rows included) for dashboard continuity.
                        "train/tensor_rows_including_dummy": float(len(edge_batch)),
                        "train/unique_questions": float(len({edge.get("question_id") for edge in sampled_edges})),
                        "train/mean_edge_age": float(sum(ages) / len(ages) if ages else 0.0),
                        "train/max_edge_age": float(max(ages) if ages else 0),
                        "training/rollout_iteration": float(self.rollout_iteration),
                        # PLAN.md P0.E: one unambiguous final global_step (the
                        # post-update value) plus explicit before/after keys.
                        "training/global_step_before_update": float(
                            global_step_before_update
                        ),
                        "training/global_step_after_update": float(self.global_steps),
                        "training/global_step": float(self.global_steps),
                        "training/optimizer_steps_this_iteration": float(
                            self.optimizer_steps_this_iteration
                        ),
                        "training/num_optimizer_steps_total": float(
                            self.num_optimizer_steps_total
                        ),
                        "training/selected_edges_this_iteration": float(len(sampled_edges)),
                        # PLAN.md §8: expected steps no longer depend only on the
                        # selected slot count — skipped logical batches perform no
                        # optimizer step.
                        "training/expected_optimizer_steps": float(
                            expected_optim_steps
                            if expected_optim_steps is not None
                            else -1
                        ),
                        "vdra/all_zero_advantage_logical_batches": float(
                            metrics.get("vdra/all_zero_advantage_logical_batches", 0.0)
                        ),
                        "vdra/zero_active_token_logical_batches": float(
                            metrics.get("vdra/zero_active_token_logical_batches", 0.0)
                        ),
                        "training/successful_actor_updates": float(self.successful_actor_updates),
                        "training/postponed_updates": float(self.postponed_updates),
                        "training/failed_updates": float(self.failed_updates),
                    }
                    self._append_timing_row(timing_path, timing, metrics)
                    metrics.update(timing)

                    # PLAN.md P0.N8: persist the manifest every optimizer step so
                    # a killed run still leaves a snapshot on disk. Overwrites in
                    # place — the manifest is small and monotonic per step.
                    self._save_manifest_best_effort(metrics)
                    self._save_live_state_best_effort(metrics)
                    # Report the manifest verdict as a boolean metric so runs can
                    # be filtered by validity at analysis time.
                    metrics["vdra/manifest_valid_main_run"] = float(
                        validate_main_run(self.run_manifest) is None
                    )
                    logger.log(data=metrics, step=self.global_steps)

                    is_last_step = self.global_steps >= self.total_training_steps
                    # PLAN.md P0.E: a jump like 8 -> 12 must still fire the
                    # step-10 save/eval. Fire once when any threshold was crossed
                    # and advance the counter past every crossed threshold.
                    eval_crossed, self.next_eval_step = advance_past_thresholds(
                        previous_step=global_step_before_update,
                        current_step=self.global_steps,
                        next_threshold=self.next_eval_step,
                        freq=test_freq,
                    )
                    if (
                        test_freq > 0
                        and (eval_crossed > 0 or is_last_step)
                        and self.val_reward_fn is not None
                    ):
                        val_metrics = self._validate()
                        if val_metrics:
                            logger.log(data=val_metrics, step=self.global_steps)

                    save_crossed, self.next_save_step = advance_past_thresholds(
                        previous_step=global_step_before_update,
                        current_step=self.global_steps,
                        next_threshold=self.next_save_step,
                        freq=save_freq,
                    )
                    if save_freq > 0 and (save_crossed > 0 or is_last_step):
                        self._save_vdra_checkpoint_bundle()
                        logger.log(
                            data={
                                "buffer/checkpoint_saved": 1.0,
                                "vdra/checkpoint_complete": 1.0,
                            },
                            step=self.global_steps,
                        )

                    if is_last_step:
                        return

                except Exception:
                    if (
                        self.run_manifest.last_iteration_status
                        == ITERATION_STATUS_RUNNING
                    ):
                        self._set_iteration_status(
                            ITERATION_STATUS_FAILED_BEFORE_ACTOR
                        )
                    self._save_manifest_best_effort(metrics)
                    self._save_live_state_best_effort(metrics)
                    raise
        self._save_vdra_checkpoint_bundle()
        final_metrics: Dict[str, Any] = {}
        self._save_live_state_best_effort(final_metrics)
