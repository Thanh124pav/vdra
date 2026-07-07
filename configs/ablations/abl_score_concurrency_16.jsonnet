// Lower vLLM concurrency (16) — useful when running on a single small GPU.
{ gear+: { score_concurrency: 16 } }
+ {
  episode_generator+: {
    inference_strategy+: { gear_score_concurrency: 16 },
  },
}
