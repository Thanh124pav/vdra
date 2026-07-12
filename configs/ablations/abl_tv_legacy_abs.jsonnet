// Estimator ablation: legacy unnormalized |exp(LP_i)-exp(LP_j)| TV estimator.
// Numerically degenerate for sequence-level log-probs (TV ~ 0 for all pairs):
// kept only to quantify the damage vs the §9 tanh estimator.
{
  gear+: {
    tv_estimator: 'legacy_abs',
  },
}
