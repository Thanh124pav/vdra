// Larger budget-allocation TV estimate for accuracy-oriented runs.
(import 'abl_budget_allocation.jsonnet') + {
  gear+: { n_tv_estimates: 16 },
}
