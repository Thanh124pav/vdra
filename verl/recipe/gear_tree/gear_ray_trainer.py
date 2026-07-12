"""``RayGearTreeTrainer`` — verl RL loop driven by native segment-tree rollout.

Subclasses ``RayPPOTrainer`` and overrides ``fit()`` so generation is the GEAR/
SPO tree rollout (variable number of edge rows with **precomputed advantages**)
instead of verl's fixed-size ``generate_sequences`` + ``compute_advantage`` flow.
Everything else (worker init, actor update, checkpoint, validation) reuses the
base class unchanged.

Per step:
  prompts -> ``actor_rollout_wg.build_trees`` (tree rollout + SPO/GEAR advantages)
          -> ``actor_rollout_wg.compute_log_prob`` (old log-probs)
          -> ``actor_rollout_wg.update_actor`` (uses the ``treetune_ppo`` loss).

Advantage is produced in generation (``tree_advantage``), so verl's
``compute_advantage`` is intentionally bypassed. Per-step timing is written to
``training_timing.jsonl`` (treetune-style); tree stats + full-tree examples are
logged by the rollout worker to ``gear_demos/``.

Requires a GPU + Ray cluster; the tree/advantage/loss core is CPU-tested.
"""

from __future__ import annotations

import json
import os
import time

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.metric_utils import reduce_metrics


class RayGearTreeTrainer(RayPPOTrainer):
    def _gear_tree_config(self) -> dict:
        """Resolve the top-level ``gear_tree`` block to a plain dict + demos_dir."""
        from omegaconf import OmegaConf

        gt = OmegaConf.to_container(self.config.get("gear_tree", {}), resolve=True) or {}
        if not gt.get("demos_dir"):
            gt["demos_dir"] = os.path.join(self.config.trainer.default_local_dir, "gear_demos")
        return gt

    def _generate_edge_batch(self, gen_batch: DataProto) -> DataProto:
        """Run the tree rollout and return a flat edge DataProto.

        Two backends:
          * ``async`` (verl >= 0.7 + vLLM >= 0.20): standard ``generate_sequences``
            routes to the ``gear_tree_agent`` agent loop, which returns per-prompt
            edges in ``non_tensor_batch['gear_tree_edges']``; we flatten them.
          * ``spmd`` (verl 0.6): the custom ``build_trees`` worker method.
        Advantages are precomputed in either case (verl ``compute_advantage`` bypassed).
        """
        gt = self._gear_tree_config()
        gen_batch.meta_info["gear_tree_config"] = gt
        backend = gt.get("rollout_backend", "async")
        if backend == "async":
            from recipe.gear_tree.async_tree_rollout import collect_tree_edges
            from recipe.gear_tree.tree_data import edges_to_dataproto

            rollout_out = self.actor_rollout_wg.generate_sequences(gen_batch)
            edges = collect_tree_edges(rollout_out)
            return edges_to_dataproto(
                edges,
                self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
            )
        return self.actor_rollout_wg.build_trees(gen_batch)

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
        self._load_checkpoint()

        # Offline per-iteration timing log (matches treetune training_timing.jsonl).
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        timing_path = os.path.join(self.config.trainer.default_local_dir, "training_timing.jsonl")
        loop_start = time.time()
        cum_train = 0.0

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)

        self.global_steps += 1
        total_epochs = self.config.trainer.total_epochs
        test_freq = self.config.trainer.get("test_freq", -1)
        save_freq = self.config.trainer.get("save_freq", -1)

        for _epoch in range(total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps

                # --- native segment-tree rollout (advantages precomputed) ----
                t0 = time.time()
                edge_batch = self._generate_edge_batch(gen_batch)
                t_gen = time.time() - t0

                # --- old log-probs recomputed with the actor -----------------
                old_log_prob = self.actor_rollout_wg.compute_log_prob(edge_batch)
                edge_batch = edge_batch.union(old_log_prob)

                # --- actor PPO update (treetune_ppo loss) --------------------
                t0 = time.time()
                if self.config.trainer.critic_warmup <= self.global_steps:
                    actor_output = self.actor_rollout_wg.update_actor(edge_batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                t_update = time.time() - t0

                cum_train += t_gen + t_update
                timing = {
                    "step": self.global_steps,
                    "timing/generation_seconds": t_gen,
                    "timing/update_seconds": t_update,
                    "timing/train_total_seconds": t_gen + t_update,
                    "timing/cumulative_train_seconds": cum_train,
                    "timing/wall_seconds": time.time() - loop_start,
                    "train/num_edges": float(len(edge_batch)),
                }
                with open(timing_path, "a") as fh:
                    fh.write(json.dumps(timing) + "\n")
                metrics.update(timing)
                logger.log(data=metrics, step=self.global_steps)

                if test_freq > 0 and self.global_steps % test_freq == 0 and self.val_reward_fn is not None:
                    val_metrics = self._validate()
                    if val_metrics:
                        logger.log(data=val_metrics, step=self.global_steps)

                if save_freq > 0 and self.global_steps % save_freq == 0:
                    self._save_checkpoint()

                self.global_steps += 1

        self._save_checkpoint()
