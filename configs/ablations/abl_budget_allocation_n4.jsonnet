// Cheaper budget-allocation TV estimate for overhead-sensitive runs.
(import 'abl_budget_allocation.jsonnet') + {
  gear+: { n_tv_estimates: 4 },
}
