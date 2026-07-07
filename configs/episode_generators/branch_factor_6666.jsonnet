// Deep tree: D=4, W=6 at every level (M tightened to 500 to keep the
// per-segment context within budget at the extra depth).
{
  episode_generator+: {
    inference_strategy+: {
      M: 500,
      max_depth: 4,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 6 },
          { depth: 1, branch_factor: 6 },
          { depth: 2, branch_factor: 6 },
          { depth: 3, branch_factor: 6 },
        ],
      },
    },
  },
}
