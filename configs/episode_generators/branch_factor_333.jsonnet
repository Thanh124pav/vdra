// Cheap dev tree: D=3, W=3 (27 leaves).
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 3,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 3 },
          { depth: 1, branch_factor: 3 },
          { depth: 2, branch_factor: 3 },
        ],
      },
    },
  },
}
