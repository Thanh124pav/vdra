(import '../gear_defaults.libsonnet') + {
  // Evaluation nodes must provide vdra_empirical_reward_variance.
  gear+: { allocation_proxy: 'empirical_variance' },
}
