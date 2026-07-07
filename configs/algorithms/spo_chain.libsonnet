// SPO-chain — Segment Policy Optimization on linear chains.
//
// PPO trainer; the chain episode generator splits each rollout into
// reasoning segments and assigns per-segment advantages using MC rollouts
// from intermediate states.
{
  episode_generator+: {
    type: 'math_episode_generator',
    inference_strategy+: { type: 'cot' },
  },
  trainer+: { type: 'ppo' },
}
