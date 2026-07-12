// Ablation #4 (Summary.md): no allocation floor (n_min = 0). Nodes whose
// dispersion proxy is ~0 can be pruned to zero branches.
{
  gear+: {
    n_min: 0,
  },
}
