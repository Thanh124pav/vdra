"""``RayGearTreeTrainer`` - VERL loop driven by native segment-tree rollout.

Generation builds tree edges with precomputed advantages and generation-time
behavior log-probabilities. A trainer-owned replay buffer applies the same edge
sampling protocol across SPO-tree and VDRA tree-family methods before forwarding
sampled edges to the actor update.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from verl import DataProto
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from recipe.gear_tree.context_contract import (
    resolve_max_edge_prompt_length,
    resolve_max_original_prompt_length,
    validate_context_contract,
)
from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer
from recipe.gear_tree.manifest_lifecycle import (
    build_run_manifest,
    update_manifest_from_edges,
)
from recipe.gear_tree.run_manifest import (
    POLICY_AGGREGATION_LEGACY,
    POLICY_AGGREGATION_VDRA,
    RunManifest,
    validate_main_run,
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
        gt = self._gear_tree_config()
        return {
            "enabled": True,
            "target_edges_per_update": 512,
            "max_edges_per_question": 32,
            "max_edge_age": 8,
            "underfill_policy": "use_available",
            "sampling_seed": 0,
            "checkpoint": True,
            "underfilled_update_policy": "postpone_until_divisible",
            **dict(gt.get("replay_buffer") or {}),
        }

    def _new_replay_buffer(self) -> GearTreeReplayBuffer:
        replay_cfg = self._replay_config()
        return GearTreeReplayBuffer(
            target_edges_per_update=replay_cfg["target_edges_per_update"],
            max_edges_per_question=replay_cfg["max_edges_per_question"],
            max_edge_age=replay_cfg["max_edge_age"],
            underfill_policy=replay_cfg.get("underfill_policy", "use_available"),
            sampling_seed=replay_cfg.get("sampling_seed", 0),
        )

    def _ensure_replay_buffer(self) -> GearTreeReplayBuffer:
        if not hasattr(self, "replay_buffer"):
            self.replay_buffer = self._new_replay_buffer()
        return self.replay_buffer

    def _checkpoint_dir_for_step(self, step: int) -> str:
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{int(step)}")

    def _restore_or_init_replay_buffer(self) -> Dict[str, Any]:
        replay_cfg = self._replay_config()
        metrics = {
            "buffer/checkpoint_restored": 0.0,
            "buffer/reset_on_resume": 0.0,
        }
        if int(getattr(self, "global_steps", 0) or 0) <= 0:
            self.replay_buffer = self._new_replay_buffer()
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
        target = int(replay_cfg["target_edges_per_update"])
        ppo_mini = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        if target % ppo_mini != 0:
            raise ValueError(
                "gear_tree.replay_buffer.target_edges_per_update must be divisible by "
                "actor_rollout_ref.actor.ppo_mini_batch_size"
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
        # PLAN.md P1.R7: refuse the deprecated ablation `_original` names in
        # strict main runs, and refuse to combine the *_style_ablation modes
        # with the canonical vdra_node_balanced policy aggregation (they are
        # ablations, not main-paper losses). The gate's own strict checks
        # cover pilot_execution_mode and allocation_runtime.
        gear_cfg = gt.get("gear") or {}
        strict = bool(gear_cfg.get("strict_vdra", True))
        tree_update_mode = str(gt.get("tree_update_mode", "spo"))
        tree_policy = self.config.get("tree_policy") or {}
        policy_agg = str(tree_policy.get("policy_aggregation", "legacy_token_mean"))
        if strict:
            if tree_update_mode in {"treepo_original", "treerl_original"}:
                raise ValueError(
                    "strict VDRA main runs must not use the deprecated "
                    "tree_update_mode aliases (PLAN.md P1.R7); rename to "
                    "'*_style_ablation' or set strict_vdra=false."
                )
            if (
                policy_agg == "vdra_node_balanced"
                and tree_update_mode
                in {"treepo_style_ablation", "treerl_style_ablation"}
            ):
                raise ValueError(
                    "The style-ablation tree_update_modes are not main-paper "
                    "advantage estimators (PLAN.md P1.R7). Use "
                    "tree_update_mode='spo' with policy_aggregation="
                    "'vdra_node_balanced' for the main run."
                )

    def _should_postpone_sampled_update(self, sampled_edges: List[Dict[str, Any]]) -> bool:
        replay_cfg = self._replay_config()
        policy = replay_cfg.get("underfilled_update_policy", "postpone_until_divisible")
        if policy == "use_available":
            return False
        if policy != "postpone_until_divisible":
            raise ValueError(f"Unknown underfilled_update_policy: {policy!r}")
        if not sampled_edges:
            return False
        ppo_mini = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        target = int(replay_cfg["target_edges_per_update"])
        return len(sampled_edges) < target and len(sampled_edges) % ppo_mini != 0

    def _fetch_rollout_server_weight_version(self, gear_cfg: Dict[str, Any]) -> str | None:
        """P0.1: server-reported weight version for the rollout replica.

        The trainer probes the SAME endpoint the scorer probes, so a mismatch
        between the two fingerprints proves the replicas have diverged. We do
        not treat a missing server fingerprint as a passing weight-version
        verification — the return value is either a non-empty string or None.
        """

        api_base = gear_cfg.get("rollout_api_base") or gear_cfg.get("scorer_api_base")
        if not api_base:
            return None
        try:
            from recipe.gear_tree.gear_core.gear.vllm_scorer import (
                fetch_server_weight_version,
            )
        except Exception:
            return None
        try:
            return fetch_server_weight_version(
                str(api_base),
                api_key=str(gear_cfg.get("scorer_api_key", "EMPTY")),
                timeout=float(gear_cfg.get("scorer_version_timeout", 5.0)),
            )
        except Exception:
            return None

    def _generate_tree_edges(self, gen_batch: DataProto) -> List[Dict[str, Any]]:
        """Run tree rollout and return raw replayable edge records."""
        gt = self._gear_tree_config()
        snapshot_id = self._current_policy_snapshot_id()
        gt["policy_snapshot_id"] = snapshot_id
        gt["current_rollout_snapshot_id"] = snapshot_id
        gear_cfg = gt.setdefault("gear", {})
        if isinstance(gear_cfg, dict):
            gear_cfg["policy_snapshot_id"] = snapshot_id
        # P0.1: fetch the rollout server's own weight version once per
        # generation. TreeAgentLoop.run reads
        # non_tensor_batch['rollout_server_weight_version'] and passes it to
        # gate.bind_snapshot, which lets the strict-mode gate refuse to
        # continue when the scorer's server fingerprint diverges.
        rollout_server_version = self._fetch_rollout_server_weight_version(gear_cfg)
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
        backend = gt.get("rollout_backend", "async")
        if backend != "async":
            raise NotImplementedError(
                "Replay-buffered RayGearTreeTrainer currently requires rollout_backend='async' "
                "so raw generation-time log-probability edges are available."
            )

        from recipe.gear_tree.async_tree_rollout import collect_tree_edges

        rollout_out = self.actor_rollout_wg.generate_sequences(gen_batch)
        edges = collect_tree_edges(rollout_out)
        return self._normalize_generated_edges(edges, snapshot_id=snapshot_id)

    def _normalize_generated_edges(
        self, edges: List[Dict[str, Any]], *, snapshot_id: str
    ) -> List[Dict[str, Any]]:
        import hashlib

        normalized: List[Dict[str, Any]] = []
        for idx, edge in enumerate(edges):
            record = dict(edge)
            # P1.5: deterministic edge id = stable hash of the identifying
            # tuple, so replay sampling is reproducible across restarts and
            # across worker processes. Falls back to (snapshot,idx) when
            # some of the source fields are missing.
            parent_path = record.get("parent_path") or record.get("gear_parent_segment_id", "")
            tree_id = record.get("tree_id", record.get("gear_segment_id", ""))
            qid = record.get("question_id", "")
            child_index = record.get("child_index", idx)
            key = f"{snapshot_id}|{qid}|{tree_id}|{parent_path}|{child_index}"
            digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
            record.setdefault("edge_id", f"{snapshot_id}:{digest}")
            record.setdefault("policy_snapshot_id", snapshot_id)
            if record["policy_snapshot_id"] != snapshot_id:
                raise ValueError("Generated edge policy_snapshot_id mismatches rollout snapshot")
            response = list(record.get("response_token_ids") or [])
            log_probs = record.get("actor_shifted_log_probs")
            if log_probs is None:
                raise ValueError("Generated edge is missing generation-time actor_shifted_log_probs")
            if len(log_probs) != len(response):
                raise ValueError("Generated edge log-probs do not align with response tokens")
            record.setdefault("depth", int(record.get("depth", 0) or 0))
            record.setdefault("leaf", bool(record.get("leaf", False)))
            record.setdefault("pruned", bool(record.get("pruned", False)))
            record.setdefault("tree_update_mode", record.get("tree_update_mode", "spo"))
            normalized.append(record)
        return normalized

    def _edges_to_update_batch(self, sampled_edges: List[Dict[str, Any]], metrics: Dict[str, Any]) -> DataProto:
        from recipe.gear_tree.tree_data import edges_to_dataproto

        # P0.2: use the same L_edge_max the startup validator resolved so a
        # config that clears validation cannot then fail here on a deep edge.
        edge_batch = edges_to_dataproto(
            sampled_edges,
            self.tokenizer,
            max_prompt_length=self._resolved_max_edge_prompt_length(),
            max_response_length=self.config.data.max_response_length,
            include_old_log_probs=True,
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
        """PLAN.md P0.N8: delegate to :func:`build_run_manifest` so the
        manifest construction can be unit-tested without importing the full
        RayPPOTrainer / torchdata / verl-worker stack.
        """
        return build_run_manifest(
            tree_policy=(self.config.get("tree_policy") or {}),
            gear_tree_cfg=self._gear_tree_config(),
            actor_loss_mode=str(
                self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
            ),
        )

    def _manifest_path(self) -> str:
        return os.path.join(
            self.config.trainer.default_local_dir, "vdra_run_manifest.json"
        )

    def _save_manifest(self, manifest: RunManifest) -> None:
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        manifest.save(self._manifest_path())

    def _update_manifest_from_edges(
        self,
        manifest: RunManifest,
        sampled_edges: List[Dict[str, Any]],
        *,
        strict: bool,
    ) -> Dict[str, Any]:
        return update_manifest_from_edges(manifest, sampled_edges, strict=strict)

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

        while self.global_steps < self.total_training_steps:
            for batch_dict in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    break
                self.rollout_iteration += 1
                metrics: Dict[str, Any] = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = self._get_gen_batch(batch)

                t0 = time.time()
                new_edges = self._generate_tree_edges(gen_batch)
                t_gen = time.time() - t0
                replay_buffer.add(
                    new_edges,
                    generation_step=self.global_steps,
                    policy_snapshot_id=self._current_policy_snapshot_id(),
                )
                # PLAN.md P0.N6: complete-tree replay when
                # tree_policy.strict_group_integrity=true; otherwise fall back
                # to the legacy per-edge reserve so SPO baselines keep working.
                if manifest_strict:
                    reservation = replay_buffer.reserve_complete_trees_for_update(
                        current_step=self.global_steps
                    )
                else:
                    reservation = replay_buffer.reserve_for_update(
                        current_step=self.global_steps
                    )
                sampled_edges = [dict(edge) for edge in reservation.edges]
                sample_stats = reservation.stats
                metrics.update({k: v for k, v in sample_stats.items() if k != "removed_edge_ids"})
                metrics["buffer/new_edges"] = len(new_edges)
                metrics["buffer/postponed_update"] = 0.0
                metrics["training/rollout_iteration"] = float(self.rollout_iteration)
                metrics["training/optimizer_step"] = float(self.global_steps)
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

                edge_batch = self._edges_to_update_batch(sampled_edges, metrics)

                # PLAN.md P0.N7/N8: run group-integrity checks + metrics BEFORE
                # the actor step so a broken reservation cannot corrupt state.
                integrity_metrics = self._update_manifest_from_edges(
                    self.run_manifest, sampled_edges, strict=manifest_strict
                )
                metrics.update(integrity_metrics)

                t0 = time.time()
                # P0.9: RayGearTreeTrainer never trains a critic, so there is
                # no critic warmup to gate the actor update on. Always update.
                try:
                    actor_output = self.actor_rollout_wg.update_actor(edge_batch)
                except Exception:
                    replay_buffer.rollback(reservation)
                    self.failed_updates += 1
                    raise
                actor_updated = True
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                if actor_updated:
                    removed = replay_buffer.commit(reservation)
                    metrics["buffer/removed_edges"] = float(len(removed))
                    metrics["buffer/size_after"] = float(len(replay_buffer))
                    self.successful_actor_updates += 1
                    self.global_steps += 1
                    # PLAN.md P0.N8: at least one successful update with no
                    # integrity failures flips the invariants-passed bit on.
                    if self.run_manifest.group_integrity_failures == 0:
                        self.run_manifest.record_invariant_pass()
                t_update = time.time() - t0

                response_tokens = edge_batch.batch["response_mask"].sum().item()
                ages = [self.global_steps - int(edge.get("generation_step", self.global_steps)) for edge in sampled_edges]
                cum_train += t_gen + t_update
                timing = {
                    "step": self.global_steps,
                    "rollout_iteration": self.rollout_iteration,
                    "optimizer_step": self.global_steps,
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
                if (
                    test_freq > 0
                    and (self.global_steps % test_freq == 0 or is_last_step)
                    and self.val_reward_fn is not None
                ):
                    val_metrics = self._validate()
                    if val_metrics:
                        logger.log(data=val_metrics, step=self.global_steps)

                if save_freq > 0 and (self.global_steps % save_freq == 0 or is_last_step):
                    self._save_checkpoint()
                    replay_metrics = self._maybe_save_replay_buffer()
                    logger.log(data=replay_metrics, step=self.global_steps)

                if is_last_step:
                    return

        self._save_checkpoint()
        self._maybe_save_replay_buffer()
