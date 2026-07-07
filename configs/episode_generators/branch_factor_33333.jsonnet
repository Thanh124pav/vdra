// Very deep tree: D=5, W=3 at every level. The GEAR compute advantage
// is most pronounced here (rho^4 savings at the deepest level).
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 5,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 3 },
          { depth: 1, branch_factor: 3 },
          { depth: 2, branch_factor: 3 },
          { depth: 3, branch_factor: 3 },
          { depth: 4, branch_factor: 3 },
        ],
      },
    },
  },
}
