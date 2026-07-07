// Shallow tree: D=2, W=6 (sanity-check that GEAR does *not* hurt at
// shallow depth; expected to be roughly compute-neutral).
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 2,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 6 },
          { depth: 1, branch_factor: 6 },
        ],
      },
    },
  },
}
