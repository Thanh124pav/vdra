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

from verl import DataProto
from verl.trainer.ppo.metric_utils import reduce_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from recipe.gear_tree.replay_buffer import GearTreeReplayBuffer


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

    def _validate_replay_startup(self) -> None:
        replay_cfg = self._replay_config()
        target = int(replay_cfg["target_edges_per_update"])
        ppo_mini = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
        if target % ppo_mini != 0:
            raise ValueError(
                "gear_tree.replay_buffer.target_edges_per_update must be divisible by "
                "actor_rollout_ref.actor.ppo_mini_batch_size"
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

    def _generate_tree_edges(self, gen_batch: DataProto) -> List[Dict[str, Any]]:
        """Run tree rollout and return raw replayable edge records."""
        gt = self._gear_tree_config()
        snapshot_id = self._current_policy_snapshot_id()
        gt["policy_snapshot_id"] = snapshot_id
        gt["current_rollout_snapshot_id"] = snapshot_id
        gear_cfg = gt.setdefault("gear", {})
        if isinstance(gear_cfg, dict):
            gear_cfg["policy_snapshot_id"] = snapshot_id
        gen_batch.meta_info["gear_tree_config"] = gt
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["rollout_iteration"] = getattr(self, "rollout_iteration", 0)
        gen_batch.meta_info["policy_snapshot_id"] = snapshot_id
        gen_batch.meta_info["current_rollout_snapshot_id"] = snapshot_id
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
        normalized: List[Dict[str, Any]] = []
        for idx, edge in enumerate(edges):
            record = dict(edge)
            record.setdefault("edge_id", f"{snapshot_id}:{idx}:{uuid.uuid4().hex}")
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

        edge_batch = edges_to_dataproto(
            sampled_edges,
            self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            include_old_log_probs=True,
        )
        if self.config.trainer.get("balance_batch", False):
            self._balance_batch(edge_batch, metrics=metrics)
        edge_batch.meta_info["global_token_num"] = edge_batch.batch["attention_mask"].sum(dim=-1).tolist()
        edge_batch.meta_info["multi_turn"] = False
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
                reservation = replay_buffer.reserve_for_update(current_step=self.global_steps)
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

                t0 = time.time()
                actor_updated = False
                if self.config.trainer.critic_warmup <= self.rollout_iteration:
                    try:
                        actor_output = self.actor_rollout_wg.update_actor(edge_batch)
                    except Exception:
                        replay_buffer.rollback(reservation)
                        self.failed_updates += 1
                        raise
                    actor_updated = True
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                else:
                    replay_buffer.rollback(reservation)
                    self.postponed_updates += 1
                    metrics["buffer/postponed_update"] = 1.0
                    metrics["training/postponed_updates"] = float(self.postponed_updates)
                    logger.log(data=metrics, step=self.global_steps)
                    continue
                if actor_updated:
                    removed = replay_buffer.commit(reservation)
                    metrics["buffer/removed_edges"] = float(len(removed))
                    metrics["buffer/size_after"] = float(len(replay_buffer))
                    self.successful_actor_updates += 1
                    self.global_steps += 1
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
