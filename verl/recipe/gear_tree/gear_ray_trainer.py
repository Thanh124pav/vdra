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

_LOGGER = logging.getLogger(__name__)

from verl import DataProto
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from recipe.gear_tree.config_validation import validate_policy_loss_consistency
from recipe.gear_tree.context_contract import (
    resolve_max_edge_prompt_length,
    resolve_max_original_prompt_length,
    validate_context_contract,
)
from recipe.gear_tree.replay_buffer import (
    GearTreeReplayBuffer,
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
    POLICY_AGGREGATION_LEGACY,
    POLICY_AGGREGATION_VDRA,
    RunManifest,
    validate_main_run,
)
from recipe.gear_tree.trainer_state import (
    GearTreeTrainerState,
    advance_past_thresholds,
    initial_next_threshold,
    load_trainer_state,
    save_trainer_state,
)


class RayGearTreeTrainer(RayPPOTrainer):
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
        tree_shape = list(gt.get("tree_shape") or [])
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
        )

    def _ensure_replay_buffer(self) -> GearTreeReplayBuffer:
        if not hasattr(self, "replay_buffer"):
            self.replay_buffer = self._new_replay_buffer()
        return self.replay_buffer

    def _checkpoint_dir_for_step(self, step: int) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{int(step)}")

    # --- PLAN.md P0.E: counter state must survive checkpoint/resume ------- #

    def _save_checkpoint(self):
        """Base checkpoint plus the VDRA counter state file.

        The base trainer restores only ``global_steps`` from the folder
        name; ``gear_tree_trainer_state.json`` carries the remaining
        counters (most importantly ``rollout_iteration``, which drives
        replay ages).
        """
        super()._save_checkpoint()
        save_trainer_state(
            self._checkpoint_dir_for_step(self.global_steps),
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
            ),
        )

    def _load_checkpoint(self):
        """Restore ``global_steps`` (base) plus the VDRA counter state.

        A checkpoint without the state file is a LEGACY checkpoint: replay
        restore is disabled for it (PLAN.md P0.E option A) because restored
        edges would carry generation iterations far above the reset
        ``rollout_iteration`` and get negative, never-expiring ages.
        """
        ret = super()._load_checkpoint()
        self._legacy_checkpoint_without_state = False
        if int(getattr(self, "global_steps", 0) or 0) > 0:
            state = load_trainer_state(
                self._checkpoint_dir_for_step(self.global_steps)
            )
            if state is None:
                self._legacy_checkpoint_without_state = True
                print(
                    "WARNING (PLAN.md P0.E): checkpoint "
                    f"global_step_{self.global_steps} has no "
                    "gear_tree_trainer_state.json (legacy checkpoint). "
                    "The replay buffer will be RESET and rollout_iteration "
                    "restarts at 0 so replay ages can never go negative."
                )
            else:
                if int(state.global_step) != int(self.global_steps):
                    raise ValueError(
                        "gear_tree_trainer_state.json global_step="
                        f"{state.global_step} does not match checkpoint "
                        f"folder global_step_{self.global_steps}"
                    )
                self.rollout_iteration = int(state.rollout_iteration)
                self.num_optimizer_steps_total = int(
                    state.num_optimizer_steps_total
                )
                self.successful_actor_updates = int(
                    state.successful_actor_updates
                )
                self.postponed_updates = int(state.postponed_updates)
                self.failed_updates = int(state.failed_updates)
        return ret

    def _restore_or_init_replay_buffer(self) -> Dict[str, Any]:
        replay_cfg = self._replay_config()
        metrics = {
            "buffer/checkpoint_restored": 0.0,
            "buffer/reset_on_resume": 0.0,
            "buffer/legacy_checkpoint_reset": 0.0,
        }
        if int(getattr(self, "global_steps", 0) or 0) <= 0:
            self.replay_buffer = self._new_replay_buffer()
            return metrics

        # PLAN.md P0.E option A: a legacy checkpoint (no trainer-state file)
        # must NOT restore replay — its edges carry generation iterations far
        # above the reset rollout_iteration and would get negative ages.
        if getattr(self, "_legacy_checkpoint_without_state", False):
            self.replay_buffer = self._new_replay_buffer()
            metrics["buffer/reset_on_resume"] = 1.0
            metrics["buffer/legacy_checkpoint_reset"] = 1.0
            self.replay_buffer_resume_metrics = metrics
            return metrics

        ckpt_dir = Path(self._checkpoint_dir_for_step(self.global_steps))
        meta_path = ckpt_dir / "gear_tree_replay_buffer_meta.json"
        if replay_cfg.get("checkpoint", True) and meta_path.exists():
            self.replay_buffer = GearTreeReplayBuffer.load(ckpt_dir)
            metrics["buffer/checkpoint_restored"] = 1.0
            metrics["buffer/restored_edges"] = float(len(self.replay_buffer))
        else:
            self.replay_buffer = self._new_replay_buffer()
            metrics["buffer/reset_on_resume"] = 1.0
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
            tree_shape=list(gt.get("tree_shape") or []),
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
        edges = collect_tree_edges(rollout_out)
        # Zero-filter contract: per-tree construction summaries preserve the
        # realized/allocated facts of parents (or whole trees) whose retained
        # edge set is empty because every child had exactly zero advantage.
        self._last_construction_summaries = collect_tree_construction_summaries(
            rollout_out
        )
        return self._normalize_generated_edges(edges, snapshot_id=snapshot_id)

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
        self.replay_buffer.save(ckpt_dir)
        return {"buffer/checkpoint_saved": 1.0}

    # --- PLAN.md P0.N8: run-manifest lifecycle --------------------------- #

    def _build_run_manifest(self) -> RunManifest:
        """PLAN.md P0.N8 / P0.7: delegate to :func:`build_run_manifest` and
        stamp the config-derived replay/optimizer knobs so ``validate_main_run``
        can compare against observed runtime values later.
        """
        manifest = build_run_manifest(
            tree_policy=(self.config.get("tree_policy") or {}),
            gear_tree_cfg=self._gear_tree_config(),
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
        # The resolved auto cap is populated once the replay buffer is built.
        if hasattr(self, "replay_buffer") and self.replay_buffer is not None:
            manifest.resolved_max_edges_per_question_per_iteration = int(
                self.replay_buffer.resolved_max_edges_per_question_per_iteration
            )
        return manifest

    def _record_iteration_on_manifest(
        self,
        *,
        selected_edges: int,
        sample_stats: Dict[str, Any],
        actual_optimizer_steps: int,
    ) -> None:
        """PLAN.md P0.7: stamp per-iteration observed facts on the manifest.

        The trainer calls this after a successful actor update so the
        manifest tracks live counters (``rollout_iteration``, ``global_step``,
        ``optimizer_steps_last_iteration``) and replay diagnostics
        (observed edge-age histogram, per-question cap max, etc.).
        """
        m = self.run_manifest
        m.rollout_iteration = int(self.rollout_iteration)
        m.global_step = int(self.global_steps)
        m.optimizer_steps_last_iteration = int(actual_optimizer_steps)
        m.num_optimizer_steps_total = int(self.num_optimizer_steps_total)
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
        if hasattr(self, "replay_buffer") and self.replay_buffer is not None:
            m.resolved_max_edges_per_question_per_iteration = int(
                self.replay_buffer.resolved_max_edges_per_question_per_iteration
            )
        # PLAN.md P0.7 / §8: `optimizer_step_accounting_valid` iff the observed
        # step count equals the expected count for this iteration. On the
        # canonical sparse path the expectation counts only TRAINABLE logical
        # batches — a batch skipped for all_zero_advantage or zero_active_tokens
        # must NOT mark a valid mixed update as accounting-invalid.
        ppo_mini = int(
            self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size", 1)
        )
        ppo_epochs = int(
            self.config.actor_rollout_ref.actor.get("ppo_epochs", 1)
        )
        expected = getattr(self, "_expected_optimizer_steps", None)
        if expected is None:
            expected = (int(selected_edges) // max(ppo_mini, 1)) * max(ppo_epochs, 1)
        expected = int(expected)
        m.expected_optimizer_steps_last_iteration = expected
        m.optimizer_step_accounting_valid = int(actual_optimizer_steps) == expected

    def _manifest_path(self) -> str:
        return os.path.join(
            self.config.trainer.default_local_dir, "vdra_run_manifest.json"
        )

    def _save_manifest(self, manifest: RunManifest) -> None:
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        manifest.save(self._manifest_path())

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
        carry the pre-filter facts of parents/trees whose every child was
        zero-filtered away (they have no rows in ``generated_edges``)."""
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
            n_optim_steps = max(step_ints) if step_ints else 1
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
        )
        # PLAN.md M2/M4: the internal optimizer-step count is a DIAGNOSTIC,
        # not the outer training unit. A parse failure or a count/expected
        # mismatch marks ``optimizer_step_accounting_valid`` invalid but never
        # crashes a run whose actor already updated the model.
        if not metrics_parse_ok:
            self.run_manifest.optimizer_step_accounting_valid = False
            return
        ppo_mini = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        ppo_epochs = int(self.config.actor_rollout_ref.actor.get("ppo_epochs", 1))
        if len(sampled_edges) % ppo_mini == 0:
            expected_steps = expected_optimizer_steps(
                selected_count=len(sampled_edges),
                ppo_mini_batch_size=ppo_mini,
                ppo_epochs=ppo_epochs,
            )
            metrics["training/optimizer_steps_expected"] = float(expected_steps)
            if int(n_optim_steps) != expected_steps:
                self.run_manifest.optimizer_step_accounting_valid = False
                metrics["vdra/optimizer_step_accounting_mismatch"] = 1.0
                _LOGGER.warning(
                    "actor performed %d optimizer steps but %d were expected "
                    "for %d selected edges / ppo_mini_batch_size=%d, "
                    "ppo_epochs=%d; marking optimizer-step accounting invalid "
                    "(diagnostic).",
                    int(n_optim_steps),
                    expected_steps,
                    len(sampled_edges),
                    ppo_mini,
                    ppo_epochs,
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
        # PLAN.md M1: keep rollout iteration, outer global_step, and
        # internal optimizer-step diagnostics as distinct counters.
        self.optimizer_steps_this_iteration = 0
        self.num_optimizer_steps_total = 0
        self._load_checkpoint()
        self._validate_replay_startup()
        replay_resume_metrics = self._restore_or_init_replay_buffer()
        replay_buffer = self._ensure_replay_buffer()

        # PLAN.md P0.N8: build the run manifest from config; the trainer
        # updates it as invariants pass/fail and persists it at every step.
        self.run_manifest = self._build_run_manifest()
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
            for batch_dict in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    break
                self.rollout_iteration += 1
                # PLAN.md P0.3: reset per-iteration optimizer-step counter so
                # postponed / failed iterations do not carry the previous
                # iteration's count forward.
                self.optimizer_steps_this_iteration = 0
                # PLAN.md P0.E: pre-update value for threshold crossing and
                # unambiguous before/after logging.
                global_step_before_update = int(self.global_steps)
                metrics: Dict[str, Any] = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = self._get_gen_batch(batch)

                t0 = time.time()
                new_edges = self._generate_tree_edges(gen_batch)
                t_gen = time.time() - t0
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
                    logger.log(data=metrics, step=self.global_steps)
                    continue
                if self._should_postpone_sampled_update(sampled_edges):
                    replay_buffer.rollback(reservation)
                    self.postponed_updates += 1
                    metrics["buffer/postponed_update"] = 1.0
                    metrics["training/postponed_updates"] = float(self.postponed_updates)
                    metrics["buffer/size_after"] = float(len(replay_buffer))
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
                    # reservation). Consume the reservation (the zero signal
                    # was processed), record the skip, and move on — no
                    # global_step, no scheduler.step, no actor RPC.
                    removed = replay_buffer.commit(reservation)
                    metrics["buffer/removed_edges"] = float(len(removed))
                    metrics["buffer/size_after"] = float(len(replay_buffer))
                    self.skipped_zero_gradient_updates += 1
                    metrics["vdra/skipped_zero_gradient_updates"] = float(
                        self.skipped_zero_gradient_updates
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
                response_tokens = edge_batch.batch["response_mask"].sum().item()
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
                expected_optim_steps = int(
                    len(sampled_edges)
                    // int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
                    * int(self.config.actor_rollout_ref.actor.get("ppo_epochs", 1))
                )
                timing = {
                    "step": self.global_steps,
                    "rollout_iteration": self.rollout_iteration,
                    "timing/generation_seconds": t_gen,
                    "timing/update_seconds": t_update,
                    "timing/train_total_seconds": t_gen + t_update,
                    "timing/cumulative_train_seconds": cum_train,
                    "timing/wall_seconds": time.time() - loop_start,
                    "train/num_edges": float(len(edge_batch)),
                    "train/num_response_tokens": float(response_tokens),
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
                    "training/expected_optimizer_steps_from_selected_edges": float(
                        expected_optim_steps
                    ),
                    "training/successful_actor_updates": float(self.successful_actor_updates),
                    "training/postponed_updates": float(self.postponed_updates),
                    "training/failed_updates": float(self.failed_updates),
                }
                with open(timing_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(timing) + "\n")
                metrics.update(timing)

                # PLAN.md P0.N8: persist the manifest every optimizer step so
                # a killed run still leaves a snapshot on disk. Overwrites in
                # place — the manifest is small and monotonic per step.
                try:
                    self._save_manifest(self.run_manifest)
                except Exception:
                    # Manifest IO must never break training; log via metrics.
                    metrics["vdra/manifest_save_failed"] = 1.0
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
                    self._save_checkpoint()
                    replay_metrics = self._maybe_save_replay_buffer()
                    logger.log(data=replay_metrics, step=self.global_steps)

                if is_last_step:
                    return

        self._save_checkpoint()
        self._maybe_save_replay_buffer()
