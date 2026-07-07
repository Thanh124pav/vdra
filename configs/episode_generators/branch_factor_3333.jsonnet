// Deep tree: D=4, W=3 at every level (cheap deep variant).
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 4,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 3 },
          { depth: 1, branch_factor: 3 },
          { depth: 2, branch_factor: 3 },
          { depth: 3, branch_factor: 3 },
        ],
      },
    },
  },
}
