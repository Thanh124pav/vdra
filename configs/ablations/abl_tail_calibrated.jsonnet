// Ablation #3 (Summary.md): VDRA WITH global tail correction.
// 0.2 is a placeholder — replace with the Q_{1-alpha} estimate from
//   python scripts/calibrate_tail_divergence.py ... (RQ3 output
//   summary.per_horizon[m].eps_tail_quantiles).
{
  gear+: {
    eps_tail: 0.2,
  },
}
