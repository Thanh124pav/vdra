local num_episodes_per_iteration = 512;
local num_rollouts_per_sample = 128;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;

(import 'polIter_deepseekR1Qwen_spo_chain_MATH.jsonnet')
+ {
  episode_generator+: {
    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,

    type: 'math_episode_generator_w_group_advantages',
    adv_method: 'grpo',
    inference_strategy+: {
      // GRPO computes one group advantage from these sibling rollouts.
      samples: num_rollouts_per_sample,
    },
  },
  num_episodes_per_iteration: num_episodes_per_iteration,
  trainer+: {
    general_training_args+: {
      target_train_batch_size: 128,
      per_device_train_batch_size: 2,
      per_device_eval_batch_size: 2,
    },
    params+: {
      use_prob_mask: false,
    },
    // cache_deepspeed_engines: false,
    // move_reference_model_to_cpu: false,
  },
}
