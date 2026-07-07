// Tree depth 3, branch factor 5 at every level (gap-fill between 4 and 6).
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 3,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 5 },
          { depth: 1, branch_factor: 5 },
          { depth: 2, branch_factor: 5 },
        ],
      },
    },
  },
}
