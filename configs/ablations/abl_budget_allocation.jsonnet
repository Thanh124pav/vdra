// Switch GEAR from legacy TV SHARE/PRUNE to simulation-lemma budget allocation.
// TV is used only for reward variance; branch-factor under-allocation is kept.
{
  gear+: {
    budget_overhead_mode: 'flexible',
    pilot_branch_factor: 8,
    tv_subnode_max_tokens: 120,
    tv_second_phase_tokens: 60,
  },
}
