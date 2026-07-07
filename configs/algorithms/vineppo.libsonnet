// VinePPO — PPO with vine-style value estimation.
//
// Same PPO trainer; the episode generator samples short branched rollouts
// at every reasoning step and uses them as a non-parametric value baseline,
// removing the need for a learned value head.
{
  episode_generator+: {
    type: 'vineppo_episode_generator',
    // The upstream VinePPO experiments reconstruct contiguous response slices
    // with delimiter.join(steps), and configure the delimiter as an empty string.
    reasoning_step_delimiter: '',
    inference_strategy+: { type: 'cot' },
  },
  trainer+: {
    type: 'ppo',
    params+: { use_prob_mask: false },
  },
}
