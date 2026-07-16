// Ablation #11: long queue timeout (larger batches, staler allocation).
(import '../gear_defaults.libsonnet') + {
  gear+: { budget_queue_timeout_seconds: 2.0 },
}
