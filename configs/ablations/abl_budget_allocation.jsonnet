// Switch GEAR from legacy TV SHARE/PRUNE to simulation-lemma budget allocation.
// TV is used only for reward variance; branch-factor under-allocation is kept.
{
  gear+: {
    budget_overhead_mode: 'flexible',
    n_tv_estimates: 8,
    tv_subnode_max_tokens: 120,
    tv_second_phase_tokens: 60,
    budget_lambda: 0.02,
  },
}
