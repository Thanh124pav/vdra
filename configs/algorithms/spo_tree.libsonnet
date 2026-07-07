// SPO-tree — Segment Policy Optimization on branching trees.
//
// PPO trainer; the hybrid episode generator builds a depth-D, width-W tree
// of reasoning rollouts and assigns advantages from subtree MC averages.
// Pair with a `branch_factor_*.jsonnet` overlay to set tree shape, e.g.:
//   --configs configs/algorithms/spo_tree.libsonnet,configs/episode_generators/branch_factor_666.jsonnet
{
  episode_generator+: {
    type: 'hybrid_episode_generator',
    inference_strategy+: { type: 'hybrid' },
  },
  trainer+: { type: 'ppo' },
}
