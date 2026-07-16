// Ablation #10: stochastic (probability-proportional) rounding, seeded.
(import '../gear_defaults.libsonnet') + {
  gear+: { rounding_strategy: 'stochastic', rounding_seed: 0 },
}
