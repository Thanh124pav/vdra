// Ablation #10: nearest-integer rounding with budget repair.
(import '../gear_defaults.libsonnet') + {
  gear+: { rounding_strategy: 'nearest_repair' },
}
