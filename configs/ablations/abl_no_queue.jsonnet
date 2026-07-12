// Ablation #5 (Summary.md): no queue batching — one queue, immediate flush.
// Each allocation batch degenerates to whatever is enqueued at flush time.
{
  gear+: {
    budget_queue_count: 1,
    budget_queue_timeout_seconds: 0.0,
  },
}
