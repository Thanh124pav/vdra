# VDRA Algorithm and Runtime Alignment Fix Request

Repository: `Thanh124pav/vdra`  
Target branch: `main`  
Reviewed commit: `8cc8a9db3626c02cfae0f780ee379d90e2996045`

## Scope

Fix the remaining algorithm and runtime inconsistencies between the `verl` and
`treetune` implementations of VDRA.

Do not redesign the logging layer in this task unless a runtime fix requires a
small accompanying field update. Logging is being handled separately.

The canonical mathematical implementation in `vdra_core` should remain the
single source of truth.

---

# 1. Canonical meanings of pilot-related parameters

Use the following definitions consistently everywhere.

## 1.1 Pilot branch factor

```text
pilot_branch_factor = k0
```

This is the number of first-phase pilot children generated from one parent node.

Let these pilot prefixes be

\[
u_1,\ldots,u_{k_0}.
\]

The current VDRA predictor clusters these pilot prefixes according to their
pairwise short-horizon TV distances. Therefore,

\[
\widehat{k}^{\mathrm{need}}
=
\text{number of distinct pilot clusters}.
\]

Consequently,

\[
1
\le
\widehat{k}^{\mathrm{need}}
\le
k_0.
\]

## 1.2 Likelihood samples per distribution

```text
likelihood_samples_per_distribution = r
```

For each pilot prefix \(u_i\), generate \(r\) second-phase continuation samples:

\[
z_{i,1},\ldots,z_{i,r}
\sim
\pi(\cdot\mid u_i).
\]

For pair \((i,j)\), estimate TV only using samples originating from \(i\) or
\(j\):

\[
\{z_{i,1},\ldots,z_{i,r},z_{j,1},\ldots,z_{j,r}\}.
\]

## 1.3 Total sampled support count

The total number of generated second-phase support samples is

\[
N_{\mathrm{support}}
=
k_0 r.
\]

The current code uses `n_tv_estimates` for this derived quantity:

```python
self.n_tv_estimates = (
    self.pilot_branch_factor
    * self.likelihood_samples_per_distribution
)
```

Therefore, in the current implementation:

```text
n_tv_estimates is not an independent pilot-count configuration.
```

It is a legacy or derived quantity equal to:

```python
n_tv_estimates = pilot_branch_factor * likelihood_samples_per_distribution
```

## Required cleanup

Remove `n_tv_estimates` as an independent public runtime knob from the main VDRA
path.

Preferred API:

```python
ConditionalTVEstimator(
    pilot_branch_factor=k0,
    likelihood_samples_per_distribution=r,
    ...
)
```

Inside the estimator, expose:

```python
self.total_support_samples = k0 * r
```

For backward compatibility only, `n_tv_estimates` may remain as a deprecated
alias.

If the caller passes all three values, validate:

```python
assert n_tv_estimates == (
    pilot_branch_factor
    * likelihood_samples_per_distribution
)
```

Otherwise raise an explicit configuration error.

Do not silently reinterpret `n_tv_estimates` as either \(k_0\) or \(r\).

---

# 2. Fix the predictor-demand range

The current predictor is:

\[
\widehat{k}^{\mathrm{need}}
=
\text{number of unique clusters among }k_0\text{ pilots}.
\]

Therefore:

\[
\widehat{k}^{\mathrm{need}}\le k_0.
\]

If:

\[
k_0
\le
k^{\mathrm{default}},
\]

then:

\[
\widehat{k}^{\mathrm{need}}
\le
k^{\mathrm{default}},
\]

and hence:

\[
\mathrm{unmet\_demand}
=
\max(
\widehat{k}^{\mathrm{need}}
-
k^{\mathrm{default}},
0
)
=
0.
\]

In that case VDRA can prune, but it cannot identify any node as needing more
branches than the default width. Residual redistribution becomes inactive.

## Required main-method behavior

For the current cluster-count predictor, enforce:

\[
k_0 > \max_d k_d^{\mathrm{default}}
\]

when:

```text
use_residual_budget = true
allocation_mode = budget_allocation
```

For a `6,6,6` tree, use at least:

```yaml
pilot_branch_factor: 8
```

For general tree shapes, prefer a dynamic rule:

```python
pilot_branch_factor = max(tree_shape) + pilot_margin
```

with:

```yaml
pilot_margin: 2
```

or require an explicit configured \(k_0\) and validate it at startup.

## Strict validation

Add:

```python
if (
    strict_vdra
    and use_residual_budget
    and allocation_mode == "budget_allocation"
    and pilot_branch_factor <= max_default_branch_factor
):
    raise ValueError(
        "The current cluster-count k predictor requires "
        "pilot_branch_factor > max default branch factor "
        "for positive unmet demand and residual redistribution."
    )
```

## Alternative future design

A future predictor may estimate demand above \(k_0\), for example through
species-richness or unseen-cluster extrapolation. If such a predictor is added,
the above validation may be relaxed for that explicitly named predictor.

Do not pretend the current observed-cluster count extrapolates beyond \(k_0\).

---

# 3. Unify `verl` and `treetune` default configurations

The main VDRA defaults must represent the same algorithm in both backends.

Use one canonical main config:

```yaml
enabled: true
k_algorithm: hierarchical
allocation_mode: budget_allocation

pilot_branch_factor: 8
likelihood_samples_per_distribution: 2

n_min: 1
use_residual_budget: true

queue_count: 4
queue_capacity: 8
queue_timeout_seconds: 1.0

strict_vdra: true
invalid_support_policy: error

bound_form: linear
tv_estimator: tanh

budget_mode: fixed_main
allocation_proxy: vdra

root_allocation: true
enable_share: false
```

The exact queue count may be changed for hardware reasons, but the semantic
defaults must match.

## Required changes

- `verl` must not use `pilot_branch_factor: null` in the main method.
- `verl` must not use `queue_timeout_seconds: 0.0`.
- `verl` and `treetune` must use the same `root_allocation` default.
- `treetune` constructor defaults must use `n_min=1`, not `0`.
- Legacy `simple`, `perplexity`, and entropy-only predictors remain ablations,
  not main defaults.

---

# 4. Make the reserve pool budget-conserving

Current behavior can draw more reserve than the queue can use.

Let:

\[
R_{\mathrm{available}}
\]

be the queue's available share of the reserve and

\[
D_{\mathrm{queue}}
=
\sum_s \mathrm{unmet\_demand}_s.
\]

The queue may consume at most:

\[
R_{\mathrm{used}}
=
\min(
R_{\mathrm{available}},
D_{\mathrm{queue}}
).
\]

## Required implementation

Preferred approach: limit the draw before allocation.

```python
total_unmet_demand = sum(
    max(
        int(item.node["vdra_predicted_k"])
        - min(
            int(item.default_branch_factor),
            int(item.node["vdra_predicted_k"]),
        ),
        0,
    )
    for item in items
)

reserve_draw = await reserve_pool.draw_queue_share(
    max_amount=total_unmet_demand
)
```

Update `SharedReservePool.draw_queue_share` to accept:

```python
max_amount: Optional[int]
```

and draw:

```python
amount = min(queue_share, max_amount, self.value)
```

## Required invariants

For each flush:

```python
used_reserve = (
    summary.allocated_budget
    - sum(summary.base_allocations.values())
)
```

Require:

```python
used_reserve == reserve_draw
```

or explicitly return the unused amount to the pool.

At all times:

```python
reserve_pool.contributed == (
    reserve_pool.consumed
    + reserve_pool.value
)
```

unless an explicitly documented cross-iteration transfer exists.

Do not count an unused draw as consumed.

---

# 5. Make the `verl` queue genuinely online

The current `verl` path creates a queue manager inside one depth-batch call,
enqueues the batch, and immediately calls `drain()`.

That is not an online timeout queue.

## Required architecture

Create one long-lived VDRA runtime context per frozen-policy rollout iteration
or minibatch:

```python
@dataclass
class VDRARuntimeContext:
    reserve_pool: SharedReservePool
    queue_manager: RootQueueManager
    policy_snapshot_id: str
```

Create it before tree construction begins and reuse it while nodes arrive.

Each node must follow:

```text
pilot estimation completes
    -> predict k
    -> compute base/saved/unmet demand
    -> contribute saved budget immediately
    -> if unmet demand > 0, enqueue immediately
    -> call flush_ready()
```

Queues flush on:

```text
capacity
timeout
```

Final drain occurs only when the rollout iteration or tree-construction phase
ends.

## Prohibited pattern

Do not implement:

```python
manager = RootQueueManager(...)
enqueue_all_nodes_at_depth(...)
await manager.drain()
```

inside `allocate_batch_async`.

## Timeout worker

A real timeout requires a periodic coroutine or event-driven timer. Merely
calling `flush_ready()` when another node arrives is insufficient because an
isolated node may never receive another event.

Implement either:

```python
asyncio.create_task(queue_timeout_worker(...))
```

or one scheduled timer per nonempty queue.

The timeout worker must stop cleanly at final drain.

## Policy consistency

All nodes in one queue must share the same:

```python
policy_snapshot_id
```

Mixed policy snapshots must raise.

---

# 6. Align the `verl` tree builder with online execution

The current `async_build_tree_batch_alloc` is level-synchronous. It waits for the
whole frontier and allocates by depth.

Replace or bypass this for main VDRA.

## Required main path

Use an asynchronous work queue of tree nodes.

Conceptual control flow:

```text
pending_tree_nodes
    -> process node
    -> generate/reuse pilot
    -> estimate predicted_k and C_s
    -> direct-prune or enqueue
    -> queue flush determines final allocated_k
    -> expand node
    -> append generated children to pending_tree_nodes
```

The whole frontier must not be required before an early node can be allocated.

The old level-synchronous builder may remain only as an explicitly named
ablation or compatibility path:

```text
allocation_runtime = depth_batch
```

The main method must use:

```text
allocation_runtime = online_timeout
```

---

# 7. Keep one reserve pool across the intended allocation scope

Do not create a new reserve pool inside every depth-batch allocation call.

The main method must define the scope explicitly.

Preferred scope:

```text
one frozen-policy rollout minibatch
```

Within that scope:

- savings from low-demand nodes enter the shared reserve;
- queued high-demand nodes may consume that reserve;
- all nodes use the same frozen policy snapshot.

At final drain, report or retain the remaining reserve according to an explicit
policy.

Do not silently discard remaining reserve when a local function returns.

---

# 8. Reuse pilots without changing segment semantics

Pilot reuse is required, but the reused branch must obey the same segment length
as the baseline branch.

Let:

```text
M = main segment length
m_pilot = first-phase pilot length
```

Require:

\[
m_{\mathrm{pilot}}
\le M.
\]

## Preferred behavior

Generate pilot prefix of length \(m_{\mathrm{pilot}}\), then complete retained
pilots with:

\[
M-m_{\mathrm{pilot}}
\]

additional tokens before inserting them as normal tree children.

For a retained pilot:

```text
pilot prefix
    + completion to segment boundary M
    = reusable main-expansion child
```

A discarded pilot is not completed.

## Strict validation

Add:

```python
if strict_vdra and tv_first_phase_tokens > segment_length:
    raise ValueError(
        "VDRA pilot length cannot exceed the main segment length "
        "when pilots are reused as tree children."
    )
```

## Simpler valid configuration

It is also valid to set:

```yaml
tv_first_phase_tokens: <same as segment_length>
```

and reuse the pilot directly.

Do not reuse a 120-token pilot as a child in a tree whose baseline segment
length is 100 tokens.

---

# 9. Apply tail calibration in every `verl` runtime path

The launcher requires:

```text
EPS_TAIL_CALIBRATION_PATH
```

but the resolved calibration must actually be applied before constructing the
gate.

## Required change

Before every `GearGate` construction:

```python
from recipe.gear_tree.calibration import resolve_gear_calibration

g = resolve_gear_calibration(g)
```

Apply this in:

```text
verl async agent-loop gate construction
verl SPMD worker gate construction
validation rollout construction
standalone rollout construction
```

The gate must receive calibrated:

```python
eps_tail
eps_tail_by_depth
eps_tail_source
```

Do not continue using the YAML placeholder after a calibration artifact is
provided.

---

# 10. Make calibration artifacts strictly compatible

The calibration script must output a top-level `metadata` object.

Required artifact schema:

```json
{
  "metadata": {
    "model": "...",
    "checkpoint": "...",
    "dataset": "...",
    "pilot_branch_factor": 8,
    "likelihood_samples_per_distribution": 2,
    "short_horizon": 60,
    "first_phase_tokens": 120,
    "full_horizon": 512,
    "quantile": 0.99,
    "seed": 0
  },
  "summary": {},
  "records": []
}
```

In strict mode, missing metadata is an error.

The loader must validate at least:

```text
model/checkpoint identity
pilot_branch_factor
likelihood_samples_per_distribution
short_horizon
quantile
```

Do not treat missing metadata as compatible.

---

# 11. Fix SPMD scorer initialization

The SPMD path currently constructs `GearGate` before attaching a scorer, while
the main `budget_allocation` gate rejects `scorer=None`.

## Required change

Construct the scorer first:

```python
scorer = EngineLPScorer(
    rollout.inference_engine,
    tokenizer,
)

gate = _build_gate(
    gt,
    scorer=scorer,
)
```

The scorer is required for `budget_allocation`, regardless of `enable_share`.

Do not gate scorer construction on:

```python
gate.enable_share
```

Share and VDRA scoring are separate features.

---

# 12. Make root allocation real and consistent

Use the same default in `verl` and `treetune`.

If:

```text
root_allocation = true
```

then roots from the same rollout minibatch must be estimated and allocated under
one shared root budget before expansion.

The `verl` builder must not hardcode:

```python
if depth > 0:
    allocate(...)
```

while advertising root allocation.

If a backend cannot batch roots across prompts, fail explicitly in strict mode.

Do not leave a configuration flag with no runtime effect.

---

# 13. Strict handling of invalid TV estimates

`pairwise_tv_tanh` currently returns zero when it finds no valid values.

In strict VDRA, no valid likelihood-ratio samples means the estimate is invalid,
not that TV equals zero.

## Required behavior

Change the estimator to return both:

```python
value
valid_count
```

or raise directly when:

```python
valid_count == 0
```

Strict mode:

```python
raise ValueError(
    "No valid pair-specific likelihood-ratio samples for TV estimation"
)
```

Non-strict ablations may exclude the pair only when explicitly configured.

Never silently map invalid support to \(D_{\mathrm{TV}}=0\).

---

# 14. Normalize the hierarchical mode spelling

Replace all instances of:

```text
hierachical
```

with:

```text
hierarchical
```

Support the typo only as a deprecated config alias at the outermost config
parser.

Internally, store exactly:

```python
self.mode = "hierarchical"
```

Do not propagate the typo through estimator logic.

---

# 15. Required runtime parity tests

Current adapter parity tests only prove that both adapters import the same
allocator. Add end-to-end tests.

## Test A: parameter semantics

Use:

```text
k0 = 8
r = 2
```

Verify:

```text
pilot children = 8
second-phase support samples = 16
total support samples = 16
```

Verify that a conflicting explicit `n_tv_estimates=8` raises because:

```text
8 != 8 * 2
```

## Test B: positive unmet demand is possible

Use:

```text
default_k = 6
pilot_branch_factor = 8
all eight pilots are distinct
```

Expected:

```text
predicted_k = 8
base_k = 6
unmet_demand = 2
```

## Test C: no redistribution with insufficient k0

Use:

```text
default_k = 6
pilot_branch_factor = 6
```

Verify that the current cluster-count predictor cannot produce positive unmet
demand.

In strict main mode, startup validation must reject this configuration when
residual allocation is enabled.

## Test D: reserve conservation

Create:

```text
reserve = 10
queue total unmet demand = 3
```

Expected:

```text
reserve_draw = 3
reserve_consumed = 3
reserve_remaining = 7
```

## Test E: isolated timeout

Enqueue one node and do not enqueue later nodes.

Advance the clock or run the timeout worker.

Expected:

```text
flush_reason = timeout
```

without requiring a later frontier event.

## Test F: no immediate final drain

Verify that enqueuing a node does not immediately produce:

```text
flush_reason = final
```

during normal tree construction.

## Test G: pilot segment equivalence

Use:

```text
M = 100
m_pilot = 60
```

Verify that a reused child has exactly the same final segment budget as a newly
generated baseline child.

## Test H: calibration applied

Construct a calibration artifact with:

```text
eps_tail = 0.23
```

Verify that the `verl` gate receives `0.23`, not the YAML placeholder.

## Test I: metadata mismatch

Use an artifact calibrated with:

```text
k0 = 4
```

and run with:

```text
k0 = 8
```

Expected: strict startup failure.

## Test J: SPMD construction

Build the SPMD gate with `budget_allocation` and `enable_share=false`.

Verify that the likelihood scorer is still attached and construction succeeds.

## Test K: root allocation

Use two roots with different \(C_s\) and predicted demand.

Verify that the root allocation flag changes their allocations while preserving
the shared root main-expansion budget.

## Test L: backend parity

Feed equivalent deterministic mocked generation and scoring into `verl` and
`treetune`.

Verify identical:

```text
predicted_k
dispersion_C
base_k
saved_k
unmet_demand
additional_k
allocated_k
pilot children reused
final child segment lengths
reserve remaining
```

---

# 16. Files to modify

At minimum inspect and update:

```text
vdra_core/core.py
vdra_core/online_budget.py
vdra_core/calibration.py

treetune/gear/tv_estimators.py
treetune/inference_strategies/gear_inference_strategy.py
configs/gear_defaults.libsonnet
configs/gear_overlay.libsonnet

verl/recipe/gear_tree/gear_gate.py
verl/recipe/gear_tree/tree_rollout.py
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/gear_tree_worker.py
verl/recipe/gear_tree/calibration.py
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/run_gear_tree.sh

scripts/calibrate_tail_divergence.py

tests/test_vdra_alignment.py
tests/test_online_gear.py
tests/test_budget_allocation.py
verl/recipe/gear_tree/tests/test_gear_gate.py
verl/recipe/gear_tree/tests/test_async_tree_rollout.py
```

---

# 17. Acceptance criteria

The task is complete only when:

1. \(k_0\), \(r\), and total support count have one unambiguous meaning.
2. `n_tv_estimates` is derived or deprecated, not an independent conflicting knob.
3. The main cluster-count predictor can produce `predicted_k > default_k`.
4. Main configs reject `k0 <= default_k` when residual allocation is expected.
5. Reserve budget is never lost after being drawn.
6. `verl` uses a real online capacity/timeout queue.
7. A lone queued node flushes by timeout without a later node arrival.
8. Final drain occurs only at the end of the allocation scope.
9. Reserve and queues live across the intended frozen-policy minibatch scope.
10. Pilot reuse preserves the main segment semantics.
11. Calibration artifacts are actually loaded in every `verl` path.
12. Missing or incompatible calibration metadata fails in strict mode.
13. SPMD VDRA initializes with a scorer even when share is disabled.
14. Root allocation has a verified runtime effect in both backends.
15. Invalid TV samples never silently become zero in strict mode.
16. `verl` and `treetune` pass end-to-end runtime parity tests.
17. The main method is accurately described as:

```text
pruning by predicted branch demand,
followed by value-dispersion-guided redistribution
under a fixed main-expansion budget.
```
