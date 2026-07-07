// Deep tree: D=4, W=4 at every level. Used by run_exp_deep_tree.sh to test
// the break-even argument (GEAR compute beats SPO at deep D).
{
  episode_generator+: {
    inference_strategy+: {
      max_depth: 4,
      branch_factor_strategy+: {
        branch_factors: [
          { depth: 0, branch_factor: 4 },
          { depth: 1, branch_factor: 4 },
          { depth: 2, branch_factor: 4 },
          { depth: 3, branch_factor: 4 },
        ],
      },
    },
  },
}
