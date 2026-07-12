(import '../gear_defaults.libsonnet') + {
  // Evaluation nodes must provide vdra_external_dispersion_C.
  gear+: { allocation_proxy: 'external_score' },
}
