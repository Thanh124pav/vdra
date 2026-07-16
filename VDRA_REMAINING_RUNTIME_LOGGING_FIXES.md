# VDRA Remaining Runtime and Logging Fixes

Repository: `Thanh124pav/vdra`  
Target branch: `main`  
Reviewed commit: `e4d159381348fd846b655e316b64530d219bec3b`

## Goal

Finish the remaining fixes after the shared VDRA core and logging schema refactor.

The mathematical allocation core is mostly correct. This task must make the
default run executable, make `online_timeout` safe, correct compute accounting,
and persist the new logging schema.

Do not reintroduce silent fallbacks.

---

## 1. Fix the default config that currently fails strict validation

Current conflict:

```yaml
segment_length: 100
tv_first_phase_tokens: 120
strict_vdra: true
```

`validate_main_config()` correctly rejects a reusable pilot longer than the main
segment.

Change the main defaults to:

```yaml
segment_length: 100
tv_first_phase_tokens: 60
tv_second_phase_tokens: 60
```

Keep the strict check:

```python
if strict_vdra and tv_first_phase_tokens > segment_length:
    raise ValueError(...)
```

Retained pilots must be completed to the main segment boundary:

```text
remaining_tokens = segment_length - actual_pilot_tokens
```

Add a test that loads the default config and confirms:

```python
tv_first_phase_tokens <= segment_length
pilot_branch_factor > max(tree_shape)
queue_timeout_seconds > 0
```

---

## 2. Make calibration artifacts compatible with the strict loader

The loader requires top-level `metadata`, but the calibration script currently
writes only `args`, `summary`, and `records`.

Update `scripts/calibrate_tail_divergence.py` to emit:

```json
{
  "metadata": {
    "model": "...",
    "checkpoint": "...",
    "dataset": "...",
    "pilot_branch_factor": 8,
    "likelihood_samples_per_distribution": 2,
    "first_phase_tokens": 60,
    "short_horizon": 60,
    "full_horizon": 512,
    "quantile": 0.99,
    "seed": 0
  },
  "args": {},
  "summary": {},
  "records": []
}
```

Use:

```python
metadata = {
    "model": args.model,
    "checkpoint": args.checkpoint,
    "dataset": args.dataset,
    "pilot_branch_factor": args.k0,
    "likelihood_samples_per_distribution": args.r,
    "first_phase_tokens": args.first_phase_tokens,
    "short_horizon": selected_runtime_horizon,
    "full_horizon": args.full_tokens,
    "quantile": args.quantile,
    "seed": args.seed,
}
```

`short_horizon` must match runtime `tv_second_phase_tokens`.

Add a round-trip test:

```text
calibration script -> artifact -> strict loader
```

Also test mismatched `k0`, `r`, and `short_horizon`.

---

## 3. Fix premature return from queued nodes

Current `online_timeout` control flow may enqueue a node and return before that
node is expanded and rewarded. The parent can then execute:

```python
child_rewards = [child["reward"] for child in children]
```

while a queued child has no `reward`.

Add a completion future to each queue item:

```python
@dataclass
class OnlineQueueItem:
    node: MutableMapping[str, Any]
    default_branch_factor: int
    depth: int
    weight_key: Optional[str] = None
    policy_snapshot_id: Optional[str] = None
    completion_future: Optional[asyncio.Future] = None
```

When enqueueing:

```python
loop = asyncio.get_running_loop()
future = loop.create_future()

manager.enqueue(
    OnlineQueueItem(
        node=node,
        default_branch_factor=default_bf,
        depth=depth,
        policy_snapshot_id=policy_snapshot_id,
        completion_future=future,
    )
)

await future
```

The future must be resolved only after:

```text
allocation
-> pilot reuse/completion
-> child generation
-> recursive descendant completion
-> node reward computation
```

Flush handler:

```python
try:
    await expand_and_finalize(item.node, ...)
    if not item.completion_future.done():
        item.completion_future.set_result(item.node)
except Exception as exc:
    if not item.completion_future.done():
        item.completion_future.set_exception(exc)
    raise
```

Required invariant:

```python
when _process_expandable(node) returns:
    "reward" in node
    node is fully finalized
```

---

## 4. Add a real timeout worker

Calling `flush_ready()` only after enqueue does not implement a true timeout.
An isolated node may never flush until final drain.

Start a background worker for the lifetime of the runtime context:

```python
async def queue_timeout_worker(manager, handle_flush, stop_event, poll_interval):
    while not stop_event.is_set():
        await asyncio.sleep(poll_interval)
        for result in await manager.flush_ready():
            await handle_flush(result)
```

Recommended:

```python
poll_interval = max(
    min(queue_timeout_seconds / 4.0, 0.1),
    0.01,
)
```

Lifecycle:

```text
create runtime context
start timeout worker
process tree tasks
await queued-node futures
stop timeout worker
final drain
await final flush expansions
assert queues empty
backpropagate rewards
```

For strict `online_timeout` runs require:

```python
queue_timeout_seconds > 0
```

The worker must propagate exceptions and must not leave detached failed tasks.

---

## 5. Prevent premature final drain

Final drain must occur only after no further tree nodes can be generated.

Before backpropagation require:

```python
assert all(queue.items == [] for queue in manager.queues)
assert all(created_futures_are_done)
assert all(finalized_nodes_have_reward)
```

Do not backpropagate while a queue flush or queued expansion is still running.

---

## 6. Preserve all pilots for compute accounting

Current runtime stores reusable/unique pilots and later uses that list to
recompute pilot generation cost. This undercounts duplicate pilots.

Store both:

```python
node["vdra_all_pilot_children"] = list(result.candidates)
node["vdra_reusable_pilot_children"] = list(
    result.unique_candidates or result.candidates
)
```

Use all pilots for compute accounting:

```python
all_pilots = node["vdra_all_pilot_children"]
reusable = node["vdra_reusable_pilot_children"]

node["vdra_pilot_children_generated"] = len(all_pilots)
node["vdra_pilot_generated_tokens"] = sum(
    len(candidate.get("response_token_ids") or [])
    for candidate in all_pilots
)
```

Use only reusable pilots for expansion:

```python
selected = reusable[:allocated_k]
```

Then:

```python
node["vdra_pilot_children_reused"] = len(selected)
node["vdra_pilot_children_discarded"] = len(all_pilots) - len(selected)
```

Do not overwrite the generated count with the unique count.

Test:

```text
8 generated pilots
3 unique pilots
2 selected for reuse
```

Expected:

```text
generated = 8
reused = 2
discarded = 6
reuse_rate = 2 / 8
```

---

## 7. Correct compute metric names and units

Do not define:

```text
generation_forward_calls = generated tokens
```

and then add it to scoring request count.

Use separate counters:

```text
vdra_generation_request_count
vdra_generation_decode_tokens

vdra_scoring_request_count
vdra_scoring_prefill_tokens
vdra_scoring_continuation_tokens
vdra_total_scored_tokens
```

Use a clearly named proxy:

```text
vdra_token_equivalent_compute_proxy
```

with:

\[
\text{proxy}
=
\text{pilot decode tokens}
+
\text{main-expansion decode tokens}
+
\text{scored prompt tokens}
+
\text{scored continuation tokens}.
\]

Do not call this exact FLOPs or total model forward calls.

Deprecate or remove:

```text
vdra_generation_forward_calls
vdra_total_model_forward_calls
```

unless actual engine calls are measured.

---

## 8. Add canonical accounting to every expanded parent

Every node for which an expansion width is considered must receive canonical
fields before expansion.

This includes:

- root uniform expansion;
- near-leaf uniform expansion;
- direct expansion;
- compatibility paths.

For uniform expansion:

```python
write_node_accounting(
    node,
    default_k=default_k,
    predicted_k=default_k,
    allocated_k=default_k,
    k_min=n_min,
    dispersion_C=0.0,
)
```

Near-leaf and root expansions must contribute to:

```text
main_expansion_requested_branches
main_expansion_allocated_branches
main_expansion_built_branches
```

Do not omit them from aggregate accounting.

For terminal nodes that never become allocation-eligible, use a separate flag:

```python
node["vdra_expansion_skipped_terminal"] = True
```

Do not create an accounting record that violates `n_min`.

---

## 9. Use canonical helpers in treetune

Replace manual field assignments after a queue flush with:

```python
write_node_accounting(
    node,
    default_k=item.default_branch_factor,
    predicted_k=node["vdra_predicted_k"],
    allocated_k=result.summary.allocations[node_id],
    k_min=self.gear_n_min,
    dispersion_C=node["vdra_dispersion_C"],
    allocation_weight=result.summary.weights[node_id],
)
```

Then in strict mode:

```python
validate_node_accounting(node, k_min=self.gear_n_min)
```

This must be identical in `verl` and `treetune`.

---

## 10. Fix treetune allocation timing

Remove the zero-duration pattern:

```python
t_alloc = time.time()
allocation_seconds += time.time() - t_alloc
```

The actual allocator time already exists in:

```python
result.allocation_seconds
```

Use:

```python
allocation_seconds_total += result.allocation_seconds
```

Prefer queue-level allocation timing over fabricated per-depth timing.

If per-depth attribution is retained, document how shared flush time is assigned.

---

## 11. Persist queue flush records

`QueueFlushResult.to_record()` exists but must be written to:

```text
queue_flushes.jsonl
```

Each record must include:

```json
{
  "run_id": "...",
  "tree_id": "...",
  "queue_id": 0,
  "policy_snapshot_id": "...",
  "flush_reason": "capacity|timeout|final_drain",
  "queue_wait_seconds": 0.0,
  "queue_size_at_flush": 1,

  "default_queue_budget": 0,
  "total_saved_budget": 0,
  "total_unmet_demand": 0,

  "reserve_before_flush": 0,
  "reserve_drawn": 0,
  "reserve_after_flush": 0,

  "allocated_residual_budget": 0,
  "unallocated_residual_budget": 0,

  "allocation_seconds": 0.0
}
```

Write one record for every flush in both backends.

---

## 12. Persist per-node records

Write:

```text
nodes.jsonl
```

for every allocation-eligible parent.

Required fields:

```text
run_id
tree_id
node_id
parent_id
depth

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

pilot_children_generated
pilot_children_reused
pilot_children_discarded
additional_children_generated

pilot_generated_tokens
main_expansion_generated_tokens
scored_tokens

queue_id
queue_wait_seconds
flush_reason
```

Do not require reconstructing this data from occasional full-tree dumps.

---

## 13. Persist run-level summaries

Write:

```text
compute_summary.json
run_manifest.json
```

### `compute_summary.json`

Include:

```text
main_expansion_requested_branches
main_expansion_allocated_branches
main_expansion_built_branches

total_saved_branches
total_redistributed_branches
total_unallocated_reserve

pilot_children_generated
pilot_children_reused
pilot_children_discarded
additional_children_generated
pilot_reuse_rate

pilot_generated_tokens
main_expansion_generated_tokens
total_generated_tokens

scoring_request_count
scoring_prefill_tokens
scoring_continuation_tokens
total_scored_tokens

generation_request_count
token_equivalent_compute_proxy

pilot_generation_seconds
likelihood_scoring_seconds
dispersion_estimation_seconds
allocation_seconds
main_expansion_seconds
total_training_wall_seconds

timeout_flush_count
capacity_flush_count
final_drain_count

fallback_count
run_valid_for_main_results
```

### `run_manifest.json`

Include executed values after calibration resolution:

```text
algorithm_requested
algorithm_executed
run_valid_for_main_results
strict_vdra

tree_shape
segment_length

pilot_branch_factor
likelihood_samples_per_distribution
first_phase_tokens
second_phase_tokens

allocation_runtime
queue_count
queue_capacity
queue_timeout_seconds

root_allocation
use_residual_budget
n_min

tv_estimator
bound_form

eps_tail
eps_tail_calibration_path
eps_tail_calibration_metadata

budget_claim
compute_proxy_definition
```

---

## 14. Clarify root/reserve allocation scope in verl

Current `verl` online manager is created per tree/prompt, while
`root_allocation=true` implies cross-root allocation.

Choose one explicit behavior.

### Preferred main implementation

Use one shared runtime context per frozen-policy rollout minibatch:

```python
@dataclass
class VDRAMinibatchRuntime:
    reserve_pool: SharedReservePool
    root_queue_manager: RootQueueManager
    policy_snapshot_id: str
```

All roots in the minibatch share:

```text
root budget
reserve pool
queue lifecycle
policy snapshot
```

### Temporary acceptable behavior

If cross-root orchestration is not implemented yet:

```yaml
root_allocation: false
```

and document:

```text
allocation scope = one tree
```

Do not claim cross-root allocation while each root owns a separate manager.

---

## 15. Strict invariant checks

At tree/minibatch completion, verify:

### Reserve

```python
reserve_pool.contributed == (
    reserve_pool.consumed + reserve_pool.value
)
```

### Queues

```python
all queues are empty
all queue futures are done
```

### Nodes

```python
validate_node_accounting(node)
```

for all allocation-eligible parents.

### Pilot accounting

```python
pilot_children_reused <= pilot_children_generated
pilot_children_discarded == (
    pilot_children_generated - pilot_children_reused
)
```

### Token accounting

```python
total_generated_tokens == (
    pilot_generated_tokens
    + main_expansion_generated_tokens
)

total_scored_tokens == (
    scoring_prefill_tokens
    + scoring_continuation_tokens
)
```

### Redistribution

```python
total_redistributed_branches == reserve_pool.consumed
```

when all reserve usage is represented by `additional_k`.

Fail loudly in strict mode.

---

## 16. Required tests

Add the following tests.

### A. Default main config

Default config passes all strict startup checks.

### B. Calibration round trip

Script output loads through the strict loader.

### C. Online timeout end-to-end

One queued node, capacity not reached, timeout flushes it, node expands, reward is
computed, and root backpropagation succeeds.

### D. Online capacity end-to-end

Queue reaches capacity and flushes with `flush_reason=capacity`.

### E. No premature reward aggregation

Parent waits until every queued child has a reward.

### F. Future exception propagation

Expansion error inside a flush reaches the waiting tree coroutine.

### G. Pilot accounting with duplicates

8 generated, 3 unique, 2 reused gives:

```text
generated=8
reused=2
discarded=6
reuse_rate=0.25
```

### H. Uniform near-leaf accounting

Near-leaf branch counts appear in requested, allocated, and built totals.

### I. Treetune timing

Mock allocator delay and verify the queue result time is recorded.

### J. Queue flush persistence

One timeout and one capacity flush produce two JSONL records.

### K. Node persistence

`nodes.jsonl` contains the full pruning-allocation trace.

### L. Compute summary identities

All branch and token identities hold.

### M. Root allocation scope

If enabled, verify budget transfer across at least two roots.

### N. Online backend parity

With deterministic mocked generation/scoring, `verl` and `treetune` produce the
same:

```text
predicted_k
dispersion_C
base_k
saved_k
unmet_demand
additional_k
allocated_k
pilot counts
reserve consumed
final branch count
```

---

## 17. Files to inspect

At minimum:

```text
scripts/calibrate_tail_divergence.py

vdra_core/logging_schema.py
vdra_core/online_budget.py
vdra_core/calibration.py

verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/tree_rollout.py
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/gear_gate.py
verl/recipe/gear_tree/tree_logging.py
verl/recipe/gear_tree/gear_tree_worker.py

treetune/inference_strategies/gear_inference_strategy.py
treetune/episode_generators/gear_episode_generator.py
treetune/gear/logging_helpers.py

tests/test_vdra_alignment.py
tests/test_online_gear.py
tests/test_tv_estimators.py
tests/test_logging_helpers.py

verl/recipe/gear_tree/tests/test_async_tree_rollout.py
verl/recipe/gear_tree/tests/test_gear_gate.py
```

Consider adding:

```text
vdra_core/runtime_context.py
vdra_core/logging_sink.py
```

to share queue lifecycle and artifact persistence.

---

## 18. Acceptance criteria

The fix is complete only when:

1. Default main config starts in strict mode.
2. Calibration script output is accepted by the strict loader.
3. Queued nodes cannot return before expansion and reward completion.
4. Timeout works without a later node arrival.
5. Final drain leaves no pending queue item or future.
6. All generated pilots are included in compute accounting.
7. Pilot reuse rate uses all generated pilots as denominator.
8. Compute metrics do not mix tokens and request counts.
9. Every expanded parent has canonical accounting fields.
10. Near-leaf and uniform expansions are included in budget totals.
11. Treetune timing uses actual queue allocation time.
12. `nodes.jsonl` is written.
13. `queue_flushes.jsonl` is written.
14. `compute_summary.json` is written.
15. `run_manifest.json` is written.
16. Root allocation scope is implemented or disabled honestly.
17. End-to-end `online_timeout` tests pass.
18. Reserve, branch, token, pilot, and node invariants pass.
19. No silent fallback occurs.
20. The experiment can be described accurately as:

```text
VDRA uses pilot-based branch-demand prediction for pruning and
value-dispersion-guided residual-budget allocation under a fixed
main-expansion budget, while reporting pilot and scoring overhead separately.
```
