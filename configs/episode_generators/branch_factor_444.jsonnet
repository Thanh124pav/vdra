// Tree depth 3, branch factor 4 at every level (Exp 1 in PLAN.md).
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 3,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 4 },
          { depth: 1, branch_factor: 4 },
          { depth: 2, branch_factor: 4 },
        ],
      },
    },
  },
}
