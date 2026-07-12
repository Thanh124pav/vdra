(import '../gear_defaults.libsonnet') + {
  // Calibration/evaluation only; nodes must provide vdra_oracle_value_dispersion.
  gear+: { allocation_proxy: 'oracle' },
}
