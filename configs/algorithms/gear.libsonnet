// GEAR — Information-Gated Policy Optimization.
//
// PPO trainer on top of SPO-tree, augmented with two online triggers:
//   * ValueShare: collapse segments whose log-prob distribution matches an
//                 already-evaluated sibling/parent.
//   * Prune:      drop segments whose value cannot exceed the parent's by
//                 more than epsilon.
// Both triggers are gated by Total Variation bounds.
//
// Compose with model/task base + gear defaults + overlay:
//   (import 'algorithms/gear.libsonnet')
//   + (import 'gear_defaults.libsonnet')
//   + (import 'gear_overlay.libsonnet')
{
  episode_generator+: {
    type: 'gear_episode_generator',
    inference_strategy+: { type: 'gear' },
  },
  trainer+: { type: 'ppo' },
}
