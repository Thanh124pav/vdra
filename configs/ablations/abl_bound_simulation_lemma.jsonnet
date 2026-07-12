// Ablation #8 (Summary.md): gamma-discounted simulation-lemma bound instead of
// the direct linear TV bound (default). Compares the two f(.) transformations
// in B_ij = f(D_m + (1-D_m)*eps_tail); both are clamped to r_max.
{
  gear+: {
    bound_form: 'simulation_lemma',
  },
}
