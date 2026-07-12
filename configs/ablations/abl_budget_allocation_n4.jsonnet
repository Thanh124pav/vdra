// Cheaper budget-allocation TV estimate for overhead-sensitive runs.
(import 'abl_budget_allocation.jsonnet') + {
  gear+: { pilot_branch_factor: 4 },
}
