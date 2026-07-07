// Very deep tree: D=5, W=6 at every level (M tightened to 400).
{
  episode_generator+: {
    inference_strategy+: {
      M: 400,
      max_depth: 5,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 6 },
          { depth: 1, branch_factor: 6 },
          { depth: 2, branch_factor: 6 },
          { depth: 3, branch_factor: 6 },
          { depth: 4, branch_factor: 6 },
        ],
      },
    },
  },
}
