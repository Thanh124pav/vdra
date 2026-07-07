// Tree depth 3, branch factor 7 at every level.
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 3,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 7 },
          { depth: 1, branch_factor: 7 },
          { depth: 2, branch_factor: 7 },
        ],
      },
    },
  },
}
