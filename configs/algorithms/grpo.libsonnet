// GRPO — Group Relative Policy Optimization (DeepSeek).
//
// Same PPO trainer, but advantages are normalized across G sibling rollouts
// of the same prompt (group-wise mean/std).  Replaces vanilla PPO on tasks
// where a value head is unstable.
{
  episode_generator+: {
    type: 'math_episode_generator_w_group_advantages',
    adv_method: 'grpo',
    inference_strategy+: { type: 'cot' },
  },
  trainer+: {
    type: 'ppo',
    params+: { use_prob_mask: false },
  },
}
