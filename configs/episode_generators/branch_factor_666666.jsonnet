// Extra-deep tree: D=6, W=6 at every level (M tightened to 300).
{
  episode_generator+: {
    inference_strategy+: {
      M: 300,
      max_depth: 6,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 6 },
          { depth: 1, branch_factor: 6 },
          { depth: 2, branch_factor: 6 },
          { depth: 3, branch_factor: 6 },
          { depth: 4, branch_factor: 6 },
          { depth: 5, branch_factor: 6 },
        ],
      },
    },
  },
}
