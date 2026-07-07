// RLOO — REINFORCE Leave-One-Out.
//
// PPO trainer + group advantages computed by leave-one-out baseline across
// G sibling rollouts (instead of GRPO-style mean/std normalization).
{
  episode_generator+: {
    type: 'math_episode_generator_w_group_advantages',
    adv_method: 'rloo',
    inference_strategy+: { type: 'cot' },
  },
  trainer+: {
    type: 'ppo',
    params+: { use_prob_mask: false },
  },
}
