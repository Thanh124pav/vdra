// Larger budget-allocation TV estimate for accuracy-oriented runs.
(import 'abl_budget_allocation.jsonnet') + {
  gear+: { pilot_branch_factor: 16 },
}
