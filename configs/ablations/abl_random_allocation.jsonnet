// RQ1 baseline: random non-uniform allocation (seeded U(0,1] per node id).
(import '../gear_defaults.libsonnet') + {
  gear+: { allocation_proxy: 'random' },
}
