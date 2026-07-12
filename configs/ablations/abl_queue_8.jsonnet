// Ablation #11 (Summary.md): larger allocation queues + slower timeout.
{
  gear+: {
    budget_queue_count: 8,
    budget_queue_timeout_seconds: 2.0,
  },
}
