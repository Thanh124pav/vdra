# VDRA Logging and Runtime Accounting Fix Request

Repository: `Thanh124pav/vdra`  
Reviewed branch: `main`  
Reviewed commit: `a11a9cdc662f4756fa25dca9d9439c3874e01c85`

## Objective

Update the repository so that both the `verl` and `treetune` implementations expose a consistent, auditable logging schema for the complete VDRA pipeline:

```text
predicted_k
    -> pruning
    -> residual-budget contribution
    -> online queue
    -> dispersion-guided residual allocation
    -> final allocated_k
    -> pilot reuse
    -> main expansion
    -> total compute accounting
```

The logging must be sufficient to verify the mathematical formulation, debug the implementation, and support the paper claim:

> VDRA performs adaptive rollout allocation under a fixed main-expansion budget. Pilot generation and likelihood scoring introduce additional computation, so this is not a fixed-total-compute comparison unless those costs are separately matched.

This task includes correcting existing logging bugs, unifying field names, adding strict runtime validation, and adding tests.

---

# 1. Blocking bug: unify allocation field names

The current `verl` path writes the final allocation to:

```python
node["gear_branch_allocation"]
```

while the shared logging helpers read:

```python
node["gear_allocated_branch_factor"]
```

This can make the logs report zero or missing allocated budget even when allocation actually ran.

## Required change

Use one canonical field across `verl`, `treetune`, tree serialization, demo logging, CSV logging, tests, and downstream consumers.

Preferred canonical VDRA field:

```python
node["vdra_allocated_k"]
```

For backward compatibility, temporary aliases may be written:

```python
node["gear_branch_allocation"] = node["vdra_allocated_k"]
node["gear_allocated_branch_factor"] = node["vdra_allocated_k"]
```

However, all new logging and tests must use `vdra_allocated_k`.

Also introduce canonical fields:

```python
node["vdra_default_k"]
node["vdra_predicted_k"]
node["vdra_cap_k"]
node["vdra_base_k"]
node["vdra_saved_k"]
node["vdra_unmet_demand"]
node["vdra_dispersion_C"]
node["vdra_allocation_weight"]
node["vdra_additional_k"]
node["vdra_allocated_k"]
```

---

# 2. Log the complete pruning-allocation formulation

VDRA uses two distinct signals:

- `predicted_k`: predicted useful branch demand, used for pruning and as the demand cap;
- `dispersion_C`: value-dispersion upper bound, used to prioritize residual-budget allocation.

They must not be merged.

For each node:

\[
k_s^{\mathrm{cap}}
=
\max(k_{\min}, \widehat{k}_s^{\mathrm{need}})
\]

\[
k_s^{\mathrm{base}}
=
\min(k_s^{\mathrm{default}}, k_s^{\mathrm{cap}})
\]

\[
\mathrm{saved}_s
=
k_s^{\mathrm{default}} - k_s^{\mathrm{base}}
\]

\[
\mathrm{unmet}_s
=
\max(k_s^{\mathrm{cap}} - k_s^{\mathrm{base}}, 0)
\]

\[
k_s^{\mathrm{alloc}}
=
k_s^{\mathrm{base}} + k_s^{\mathrm{additional}}
\]

with

\[
0
\le
k_s^{\mathrm{additional}}
\le
\mathrm{unmet}_s
\]

and

\[
k_s^{\mathrm{base}}
\le
k_s^{\mathrm{alloc}}
\le
k_s^{\mathrm{cap}}.
\]

## Required per-node fields

Every node that participates in pruning or allocation must contain:

```python
{
    "vdra_default_k": int,
    "vdra_predicted_k": int,
    "vdra_cap_k": int,
    "vdra_base_k": int,
    "vdra_saved_k": int,
    "vdra_unmet_demand": int,

    "vdra_dispersion_C": float,
    "vdra_allocation_weight": float,

    "vdra_additional_k": int,
    "vdra_allocated_k": int,

    "vdra_reserve_contribution": int,
    "vdra_reserve_received": int,
}
```

The following identities must hold:

```python
assert cap_k == max(k_min, predicted_k)
assert base_k == min(default_k, cap_k)
assert saved_k == default_k - base_k
assert unmet_demand == max(cap_k - base_k, 0)
assert additional_k == allocated_k - base_k
assert reserve_contribution == saved_k
assert reserve_received == additional_k
assert base_k <= allocated_k <= cap_k
```

Use `k_min = 1` in the main VDRA config.

---

# 3. Rename misleading mathematical fields

The current code uses names such as:

```text
gear_reward_variance
gear_sigma2
gear_sigma4
```

for a quantity computed from pairwise TV-derived value-gap bounds.

This quantity is not an empirical reward variance. It is the VDRA dispersion upper bound:

\[
C_s.
\]

## Required change

Use:

```python
node["vdra_dispersion_C"]
```

as the canonical field.

Keep empirical child reward variance separate:

```python
node["vdra_empirical_child_reward_variance"]
```

For temporary compatibility:

```python
node["gear_reward_variance"] = node["vdra_dispersion_C"]
```

but do not display it as `reward_variance` or `sigma2` in new logs.

The allocation weight in the main method is:

\[
w_s = \sqrt{C_s}.
\]

Log it explicitly as:

```python
node["vdra_allocation_weight"]
```

---

# 4. Trace residual-budget source and destination

Current logs only expose aggregate reserve totals. That is insufficient to verify that pruning actually funds later allocation.

## Required per-node logging

For each node:

```python
vdra_reserve_contribution = vdra_saved_k
vdra_reserve_received = vdra_additional_k
```

## Required per-run invariants

At the end of a run or tree batch:

\[
\sum_s \mathrm{vdra\_saved\_k}_s
=
\mathrm{reserve\_contributed}
\]

and

\[
\sum_s \mathrm{vdra\_additional\_k}_s
=
\mathrm{reserve\_consumed}.
\]

The implementation must check these invariants in strict mode.

## Required aggregate fields

```python
{
    "vdra_total_saved_branches": int,
    "vdra_total_redistributed_branches": int,
    "vdra_total_unallocated_reserve": int,
    "vdra_reserve_contributed": int,
    "vdra_reserve_consumed": int,
    "vdra_reserve_remaining": int,
}
```

---

# 5. Replace ambiguous queue logging

The online queue must distinguish three flush reasons:

```text
capacity
timeout
final_drain
```

Do not use a single Boolean such as `timed_out`.

## Required queue state

Each queue must maintain:

```python
{
    "queue_id": int,
    "policy_snapshot_id": str,
    "first_enqueued_at": float,
    "capacity": int,
    "timeout_seconds": float,
    "items": list,
}
```

## Required queue flush record

Every queue flush must emit one machine-readable record:

```python
{
    "queue_id": int,
    "policy_snapshot_id": str,
    "flush_reason": "capacity" | "timeout" | "final_drain",

    "first_enqueued_at": float,
    "flushed_at": float,
    "queue_wait_seconds": float,
    "queue_size_at_flush": int,

    "default_queue_budget": int,
    "total_saved_budget": int,
    "total_unmet_demand": int,

    "reserve_before_flush": int,
    "reserve_drawn": int,
    "reserve_after_flush": int,

    "allocated_residual_budget": int,
    "unallocated_residual_budget": int,

    "allocation_seconds": float,
}
```

## Required behavior

Flush on capacity:

```python
len(queue.items) >= queue.capacity
```

Flush on timeout:

```python
time.monotonic() - queue.first_enqueued_at >= queue.timeout_seconds
```

Final drain is allowed only when:

- tree generation has ended;
- no more eligible nodes can be produced;
- the policy iteration is about to update.

A zero timeout must not be logged as a timeout event. It must either be prohibited in the main VDRA config or classified as an explicitly named immediate/debug mode.

---

# 6. Correct allocation timing

Current timing around queue allocation is effectively measured after allocation has already occurred, making `allocation_seconds` approximately zero.

## Required change

Measure directly around the actual allocation call:

```python
t0 = time.perf_counter()

summary = allocate_branch_factors(...)

allocation_seconds = time.perf_counter() - t0
```

Store `allocation_seconds` in the queue flush result and propagate it to:

```text
queue_flushes.jsonl
tree stats
per-depth stats
training metrics
```

Do not include child expansion time in `allocation_seconds`.

Use separate timers for:

```text
pilot_generation_seconds
likelihood_scoring_seconds
dispersion_estimation_seconds
allocation_seconds
main_expansion_seconds
tree_construction_seconds
```

---

# 7. Add full compute accounting

The experiment fixes the main-expansion budget, not total compute.

The logs must separately report pilot generation, scoring, main expansion, and total cost.

## Required token counters

```python
{
    "vdra_main_expansion_generated_tokens": int,
    "vdra_pilot_generated_tokens": int,
    "vdra_total_generated_tokens": int,

    "vdra_scoring_request_count": int,
    "vdra_scoring_prefill_tokens": int,
    "vdra_scoring_continuation_tokens": int,
    "vdra_total_scored_tokens": int,
}
```

with:

```python
vdra_total_generated_tokens = (
    vdra_main_expansion_generated_tokens
    + vdra_pilot_generated_tokens
)
```

## Required forward-pass counters

```python
{
    "vdra_generation_forward_calls": int,
    "vdra_scoring_forward_calls": int,
    "vdra_total_model_forward_calls": int,
}
```

If exact FLOPs are unavailable, introduce a documented compute proxy:

```python
vdra_total_forward_pass_cost
```

The proxy must distinguish at least:

- autoregressive decode tokens;
- scoring/prefill tokens.

Document the formula in code and in the run manifest.

## Required time counters

```python
{
    "vdra_pilot_generation_seconds": float,
    "vdra_likelihood_scoring_seconds": float,
    "vdra_dispersion_estimation_seconds": float,
    "vdra_allocation_seconds": float,
    "vdra_main_expansion_seconds": float,
    "vdra_total_training_wall_seconds": float,
}
```

---

# 8. Add pilot reuse accounting

Pilot children must be retained and reused.

## Required per-node fields

```python
{
    "vdra_pilot_children_generated": int,
    "vdra_pilot_children_reused": int,
    "vdra_pilot_children_discarded": int,
    "vdra_additional_children_generated": int,
}
```

## Required run metric

```python
vdra_pilot_reuse_rate = (
    vdra_pilot_children_reused
    / max(vdra_pilot_children_generated, 1)
)
```

## Required consistency check

For each expanded node:

```python
allocated_k == pilot_children_reused + additional_children_generated
```

unless a documented early-termination or invalid-generation condition occurs.

Such exceptions must be explicitly logged.

---

# 9. Merge all tree-level stats in `verl`

The `treetune` path already stores many metrics in:

```python
tree["gear_stats"]
```

The `verl` `TreeDemoLogger` currently recomputes only aggregate tree stats and may drop existing queue, reserve, generation, and timing metrics.

## Required change

Merge stored and recomputed stats:

```python
stored_stats = dict(tree.get("gear_stats", {}))
derived_stats = aggregate_tree_stats(tree)

gear_stats = {
    **stored_stats,
    **derived_stats,
}
```

Define a clear precedence rule. Prefer explicit stored runtime counters over derived values when the two represent the same metric.

Apply the same schema in both `verl` and `treetune`.

---

# 10. Make logging schema identical in `verl` and `treetune`

Both implementations must emit the same file names and JSON schemas.

## Required artifacts

Each run must produce:

```text
run_manifest.json
trees.jsonl
nodes.jsonl
queue_flushes.jsonl
compute_summary.json
training_timing.jsonl
```

Optional human-readable files:

```text
trees.md
full_trees/
```

## Required parity

For equivalent synthetic inputs, `verl` and `treetune` must produce identical values for:

```text
default_k
predicted_k
cap_k
base_k
saved_k
unmet_demand
dispersion_C
allocation_weight
additional_k
allocated_k
reserve_contribution
reserve_received
```

---

# 11. Expand the run manifest

The run manifest must explicitly report the executed algorithm, not only the requested configuration.

## Required fields

```python
{
    "algorithm_requested": "vdra",
    "algorithm_executed": "vdra",
    "run_valid_for_main_results": True,

    "vdra_enabled": True,
    "strict_vdra": True,

    "k_predictor": str,
    "allocation_objective": "sum_s C_s / k_s",
    "allocation_weight": "sqrt(C_s)",

    "k_min": 1,
    "budget_lambda": 0.0,

    "tv_estimator": "tanh",
    "value_bound": "linear",

    "eps_tail": float,
    "eps_tail_source": str,
    "eps_tail_calibration_path": str,

    "queue_mode": "online_timeout",
    "queue_capacity": int,
    "queue_timeout_seconds": float,

    "budget_claim": "fixed_main_expansion_budget",
    "compute_proxy_definition": str,
}
```

The banner must print at least:

```text
algorithm_requested
algorithm_executed
run_valid_for_main_results
strict_vdra
k_predictor
allocation_weight
k_min
eps_tail_source
queue_mode
budget_claim
```

---

# 12. Prohibit hidden fallbacks

Current code permits scoring failure to fall back to uniform branching.

This is not acceptable for main VDRA runs.

## Required strict behavior

Add:

```yaml
strict_vdra: true
```

In strict mode, raise an exception for:

- missing scorer;
- missing log-probabilities;
- scoring failure;
- insufficient pair-specific support;
- non-finite dispersion bound;
- missing tail-calibration artifact;
- inactive queue worker;
- root-allocation flag with no runtime effect;
- accidental uniform allocation fallback;
- inconsistent logging invariants.

## Non-strict debug behavior

A fallback is permitted only when explicitly configured:

```yaml
strict_vdra: false
allow_uniform_fallback: true
```

When fallback occurs, log:

```python
{
    "algorithm_requested": "vdra",
    "algorithm_executed": "uniform_fallback",
    "fallback_reason": str,
    "fallback_node_count": int,
    "run_valid_for_main_results": False,
}
```

Never silently switch algorithms.

Remove or rewrite tests that currently treat uniform fallback as the correct main behavior.

---

# 13. Per-node JSON schema

Each node record in `nodes.jsonl` must contain:

```python
{
    "run_id": str,
    "tree_id": str,
    "node_id": str,
    "parent_id": str | None,
    "depth": int,
    "policy_snapshot_id": str,

    "default_k": int,
    "predicted_k": int,
    "cap_k": int,
    "base_k": int,
    "saved_k": int,
    "unmet_demand": int,

    "dispersion_C": float | None,
    "allocation_weight": float | None,

    "additional_k": int,
    "allocated_k": int,

    "reserve_contribution": int,
    "reserve_received": int,

    "pilot_children_generated": int,
    "pilot_children_reused": int,
    "pilot_children_discarded": int,
    "additional_children_generated": int,

    "pilot_generated_tokens": int,
    "main_generated_tokens": int,
    "scored_tokens": int,

    "queue_id": int | None,
    "queue_wait_seconds": float | None,
    "flush_reason": str | None,

    "fallback_used": bool,
    "fallback_reason": str | None,
}
```

---

# 14. Per-run summary schema

`compute_summary.json` must contain:

```python
{
    "main_expansion_requested_branches": int,
    "main_expansion_allocated_branches": int,
    "main_expansion_built_branches": int,

    "total_saved_branches": int,
    "total_redistributed_branches": int,
    "total_unallocated_reserve": int,

    "pilot_children_generated": int,
    "pilot_children_reused": int,
    "pilot_children_discarded": int,
    "additional_children_generated": int,
    "pilot_reuse_rate": float,

    "main_expansion_generated_tokens": int,
    "pilot_generated_tokens": int,
    "total_generated_tokens": int,

    "likelihood_scoring_requests": int,
    "scoring_prefill_tokens": int,
    "scoring_continuation_tokens": int,
    "total_scored_tokens": int,

    "generation_forward_calls": int,
    "scoring_forward_calls": int,
    "total_model_forward_calls": int,
    "total_forward_pass_cost": float,

    "pilot_generation_seconds": float,
    "likelihood_scoring_seconds": float,
    "dispersion_estimation_seconds": float,
    "allocation_seconds": float,
    "main_expansion_seconds": float,
    "total_training_wall_seconds": float,

    "timeout_flush_count": int,
    "capacity_flush_count": int,
    "final_drain_count": int,

    "fallback_count": int,
    "run_valid_for_main_results": bool,
}
```

---

# 15. Required tests

## Test A: canonical field compatibility

Run a synthetic `verl` allocation and verify that:

```python
node["vdra_allocated_k"]
```

is visible to all logging helpers and produces a nonzero allocated budget.

## Test B: pruning trace

Use:

```text
default_k = 6
predicted_k = 3
```

Expected:

```text
cap_k = 3
base_k = 3
saved_k = 3
unmet_demand = 0
additional_k = 0
allocated_k = 3
reserve_contribution = 3
```

## Test C: redistribution trace

Use two nodes:

```text
A: default_k=6, predicted_k=2
B: default_k=6, predicted_k=10
```

Expected:

```text
A saved_k = 4
B unmet_demand = 4
```

The residual pool must transfer up to four branches from A to B.

Verify:

```python
sum(saved_k) == reserve_contributed
sum(additional_k) == reserve_consumed
allocated_k_B <= predicted_k_B
```

## Test D: capped allocation

Use a node with very large `dispersion_C` but `unmet_demand=1`.

Verify that it receives at most one extra branch.

## Test E: minimum branch factor

Use:

```text
predicted_k = 0
k_min = 1
```

Expected:

```text
cap_k = 1
base_k = 1
allocated_k >= 1
```

## Test F: queue timeout

Enqueue a node, advance the monotonic clock past the timeout, and verify:

```text
flush_reason = timeout
queue_wait_seconds >= timeout_seconds
```

## Test G: capacity flush

Fill the queue to capacity and verify:

```text
flush_reason = capacity
```

It must not increment `timeout_flush_count`.

## Test H: final drain

Drain a nonempty queue at the end and verify:

```text
flush_reason = final_drain
```

## Test I: allocation timing

Mock an allocator with a controlled delay and verify that `allocation_seconds` measures the allocator call, not the later expansion.

## Test J: pilot reuse accounting

Generate four pilot candidates and allocate six children.

Expected:

```text
pilot_children_reused = 4
additional_children_generated = 2
allocated_k = 6
```

## Test K: strict failure

Make the scorer raise an exception with:

```yaml
strict_vdra: true
```

Expected: training initialization or rollout must fail.

No uniform fallback is permitted.

## Test L: non-strict fallback labeling

With:

```yaml
strict_vdra: false
allow_uniform_fallback: true
```

Expected:

```text
algorithm_executed = uniform_fallback
run_valid_for_main_results = false
fallback_count > 0
```

## Test M: `verl`–`treetune` parity

For the same synthetic queue input, both implementations must produce identical node and queue records.

## Test N: compute accounting consistency

Verify:

```python
total_generated_tokens == (
    pilot_generated_tokens
    + main_expansion_generated_tokens
)

total_model_forward_calls == (
    generation_forward_calls
    + scoring_forward_calls
)
```

---

# 16. Suggested files to inspect and modify

At minimum, inspect and update:

```text
verl/recipe/gear_tree/gear_gate.py
verl/recipe/gear_tree/tree_logging.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/gear_core/gear/online_budget.py
verl/recipe/gear_tree/gear_core/gear/logging_helpers.py
verl/recipe/gear_tree/gear_core/gear/tree_policy_logging.py
verl/recipe/gear_tree/gear_core/gear/budget_allocation.py

treetune/inference_strategies/gear_inference_strategy.py
treetune/episode_generators/gear_episode_generator.py
treetune/gear/online_budget.py
treetune/gear/logging_helpers.py
treetune/gear/tree_policy_logging.py
treetune/gear/budget_allocation.py

tests/test_online_gear.py
tests/test_logging_helpers.py
tests/test_tree_stats.py
verl/recipe/gear_tree/tests/test_gear_gate.py
```

Prefer extracting the canonical VDRA logging schema and invariant checks into a shared module instead of maintaining divergent implementations.

Suggested shared modules:

```text
vdra_core/logging_schema.py
vdra_core/accounting.py
vdra_core/invariants.py
```

---

# 17. Acceptance criteria

The task is complete only when all conditions below hold.

1. `verl` and `treetune` use the same canonical node field names.
2. Every participating node logs the full pruning-allocation chain.
3. The logs distinguish `predicted_k`, `base_k`, and `allocated_k`.
4. The logs distinguish `dispersion_C` from empirical reward variance.
5. Every saved branch is traceable to a source node.
6. Every redistributed branch is traceable to a destination node.
7. Queue flushes have explicit reasons: capacity, timeout, or final drain.
8. Queue wait time and queue size at flush are logged.
9. Allocation timing measures the actual optimizer.
10. Pilot generation, scoring, and main expansion tokens are counted separately.
11. Total forward-pass cost is reported using a documented definition.
12. Pilot reuse is measured explicitly.
13. The `verl` logger preserves all runtime stats already present in the tree.
14. The default main VDRA run cannot silently fall back to uniform allocation.
15. Fallback runs are labeled invalid for main experimental results.
16. `verl` and `treetune` pass parity tests.
17. All accounting invariants pass in strict mode.
18. The produced logs are sufficient to substantiate the paper statement:

```text
fixed main-expansion budget,
with total generation and scoring compute reported separately.
```
