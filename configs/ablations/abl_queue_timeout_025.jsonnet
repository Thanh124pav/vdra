// Ablation #11: short queue timeout (smaller batches, fresher allocation).
(import '../gear_defaults.libsonnet') + {
  gear+: { budget_queue_timeout_seconds: 0.25 },
}
