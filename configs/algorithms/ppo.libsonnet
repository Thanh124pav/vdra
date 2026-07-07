// PPO — Proximal Policy Optimization.
//
// Pure on-policy single-trajectory policy gradient with PPO clip.  This is
// the default trainer/episode-generator stack that every other algorithm
// in this repo extends (GRPO, RLOO, VinePPO, SPO-*, GEAR).
//
// Compose with a model/task overlay, e.g.:
//   --configs configs/polIter_deepseekR1Qwen_ppo_MATH.jsonnet,configs/algorithms/ppo.libsonnet
{
  episode_generator+: {
    type: 'math_episode_generator',
    inference_strategy+: { type: 'cot' },
  },
  trainer+: { type: 'ppo' },
}
