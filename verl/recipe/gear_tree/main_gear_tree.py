"""Entry point for the GEAR/Tree recipe.

Subclasses verl's ``TaskRunnerBase`` to (1) use ``GearTreeActorRolloutWorker`` (adds
the dispatched ``build_trees`` method) and (2) run ``RayGearTreeTrainer`` (tree
rollout loop). Importing the recipe modules registers ``treetune_ppo`` (policy
loss) and ``gear_math`` (reward manager). Config lives in ``./config``.

GPU/Ray required to actually train.
"""

from __future__ import annotations

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunnerBase, create_rl_dataset, create_rl_sampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

# Import for decorator side effects (registers treetune_ppo + gear_math + the
# gear_tree_agent agent loop for the async rollout backend).
import recipe.gear_tree.policy_loss  # noqa: F401
import recipe.gear_tree.reward  # noqa: F401
import recipe.gear_tree.async_tree_rollout  # noqa: F401


@ray.remote(num_cpus=1)
class GearTreeTaskRunner(TaskRunnerBase):
    def add_actor_rollout_worker(self, config):
        """Pick the worker by rollout backend.

        * async (verl>=0.7): use verl's stock async worker; the tree is built by
          the registered gear_tree_agent agent loop.
        * spmd (verl 0.6): force the custom SPMD tree worker (build_trees).
        """
        backend = config.get("gear_tree", {}).get("rollout_backend", "async")
        if backend != "spmd":
            return super().add_actor_rollout_worker(config)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from recipe.gear_tree.gear_tree_worker import GearTreeActorRolloutWorker

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(GearTreeActorRolloutWorker)
        return GearTreeActorRolloutWorker, RayWorkerGroup

    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.config import validate_config
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.trainer.ppo.utils import need_critic, need_reference_policy

        from recipe.gear_tree.gear_ray_trainer import RayGearTreeTrainer

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        # P0.9: gear-tree trainer never trains a critic. Only add a critic
        # worker when the config explicitly asks for one; otherwise the Ray
        # placement group and FSDP shards would create a critic replica that
        # never receives a training call.
        if need_critic(config):
            self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}) or {})
        gear_tree_cfg = config.get("gear_tree", {}) or {}
        reward_kwargs.setdefault("answer_prefix", gear_tree_cfg.get("answer_prefix", "# Answer\n"))
        reward_kwargs.setdefault(
            "use_minerva_few_shot_prompt",
            gear_tree_cfg.get("use_minerva_few_shot_prompt", False),
        )
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **reward_kwargs
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **reward_kwargs
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayGearTreeTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def run_gear_tree(config) -> None:
    if not ray.is_initialized():
        runtime_env = OmegaConf.merge(
            get_ppo_ray_runtime_env(), config.ray_kwargs.get("ray_init", {}).get("runtime_env", {})
        )
        ray.init(runtime_env=OmegaConf.to_container(runtime_env, resolve=True))
    runner = GearTreeTaskRunner.remote()
    ray.get(runner.run.remote(config))


@hydra.main(config_path="config", config_name="gear_tree_trainer", version_base=None)
def main(config):
    run_gear_tree(config)


if __name__ == "__main__":
    main()
