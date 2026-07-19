# Claude Fix Guide — MEDIUM Tasks

> Scope: cross-function production refactors that touch replay lifecycle, strict identity, and manifest semantics. Do these after `CLAUDE_FIX_EASY.md` passes.

Do not change the mathematical objective. Canonical training remains an equal average over selected replay segment slots.

## M1 — Introduce one explicit reserved-update transaction

### Problem

The current trainer performs reservation, shortcuts, tensorization, validation, actor execution, replay commit, counter mutation, and manifest mutation in several disconnected blocks. Exceptions before `update_actor()` may leave rows reserved; exceptions after model mutation can leave driver state partially committed.

### Required design

Extract one production helper, for example:

```python
_execute_reserved_actor_update(
    reservation,
    sampled_edges,
    metrics,
    manifest_strict,
) -> ActorUpdateResult
```

The helper should make state transitions explicit:

```text
RESERVED
  ↓ replay-row validation
VALIDATED
  ↓ optional all-zero handling
TENSORIZED
  ↓ actor update
ACTOR_RETURNED
  ↓ cross-worker and expected-step verification
VERIFIED
  ↓ replay commit + driver counters + manifest
COMMITTED
```

Any exception before the actor begins:

```text
rollback replay reservation
leave counters unchanged
leave manifest success facts unchanged
```

Any actor exception:

```text
rollback replay reservation
increment failed-update diagnostics
re-raise
```

Any invalid actor result after model mutation:

```text
DO NOT pretend the model was rolled back
DO NOT commit replay rows or driver counters
mark the run irrecoverably invalid
save an explicit failure record
abort strict training
```

A model update cannot be transactionally undone by replay rollback. Encode that distinction clearly.

### Suggested result type

```python
@dataclass
class ActorUpdateResult:
    actor_output: DataProto | None
    selected_edges: int
    optimizer_steps: int
    replay_committed: bool
    skipped_all_zero: bool
    actor_started: bool
```

### Required files

```text
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/replay_buffer.py
verl/recipe/gear_tree/run_manifest.py
verl/recipe/gear_tree/tests/test_trainer_transaction_lifecycle.py
```

### Acceptance tests

Inject failures at each stage:

1. replay validation;
2. tensorization;
3. actor RPC;
4. inconsistent worker step reports;
5. actual/expected step mismatch;
6. replay commit.

For each injection, assert exact replay reservation state, counter state, manifest state, and whether the model may already have changed.

---

## M2 — Enforce canonical tree and edge identity strictly

### Problem

Strict paths still accept `tree_instance_id or tree_id`, and derived edge IDs can be bypassed by caller-supplied IDs. Collision checks based only on duplicate `(tree_id, child_segment_id)` pairs do not prove that every stochastic tree has a unique identity.

### Canonical strict contract

Every generated stochastic tree must carry:

```text
tree_instance_id
policy_snapshot_id
rollout_iteration
stable question_id
per-tree UUID/counter component
```

Strict extraction must require `tree_instance_id` specifically. A generic legacy `tree_id` is not sufficient.

Every edge ID must be deterministically derived from:

```text
tree_instance_id
parent_group_id
child_segment_id
```

Suggested contract:

```python
derived_edge_id = derive_edge_id(
    tree_instance_id,
    parent_group_id,
    child_segment_id,
)

if supplied_edge_id is not None and supplied_edge_id != derived_edge_id:
    raise ValueError(...)
record["edge_id"] = derived_edge_id
```

Do not use `setdefault()` in strict mode.

### Collision checks

At construction time, verify:

- every edge has nonempty `tree_instance_id`;
- all edges of one generated tree agree on that ID;
- two independent stochastic-tree records in the same generated batch do not share an ID;
- no identity equals an ambiguous `(snapshot, question)` fallback;
- all derived edge IDs are globally unique within the generated batch;
- replay insertion remains transactionally duplicate-safe.

Legacy fallback behavior may remain only in explicit non-strict compatibility paths.

### Required files

```text
verl/recipe/gear_tree/vllm_rollout_tree.py
verl/recipe/gear_tree/tree_advantage.py
verl/recipe/gear_tree/tree_data.py
verl/recipe/gear_tree/manifest_lifecycle.py
verl/recipe/gear_tree/tests/test_strict_tree_identity.py
```

### Acceptance tests

- Two trees with same question/snapshot/iteration receive different IDs.
- Reusing `tree_id="t0"` with disjoint child IDs fails strict validation.
- Missing `tree_instance_id` fails even when legacy `tree_id` exists.
- A caller-supplied mismatching `edge_id` fails.
- Non-strict legacy fixtures continue to load only when compatibility mode is explicit.

---

## M3 — Decouple canonical manifest invariants from obsolete objective weights

### Problem

Canonical `DataProto` no longer carries `objective_weights` or `segment_objective_weights`, but construction-time manifest code still computes tree-normalized segment weights and uses them to certify canonical segment invariants.

### Required canonical checks

For `vdra_segment_mean_ppo`, `segment_count_invariants_passed` should depend on observed construction facts such as:

```text
all realized non-placeholder segments counted exactly once
fresh_iid realized_child_count == allocated_k
queue segment counts sum to tree_total_segment_count
edge IDs unique
stored log-probs align with response tokens
no silent truncation in tensorization
```

It must not depend on:

```text
compute_segment_objective_weights
validate_segment_objective_weights
compute_objective_weights
parent/tree normalized float coefficients
```

Only the explicit `vdra_node_balanced_ppo` ablation may compute and validate those weights.

### Manifest monotonicity

All failure facts must be cumulative and non-healing:

```text
group_integrity_failures
segment_count_failures
replay_batch_failures
optimizer_step_accounting_failures
actor_result_failures
```

Success bits may become true only after an observed successful canonical update and must never erase historical failures.

### Required files

```text
verl/recipe/gear_tree/manifest_lifecycle.py
verl/recipe/gear_tree/run_manifest.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/tests/test_manifest_observed_facts.py
verl/recipe/gear_tree/tests/test_run_manifest.py
```

### Acceptance tests

- Canonical manifest validation never calls segment/objective weight helpers.
- Node-balanced ablation still validates its weights.
- A historical failure keeps the run invalid after later clean iterations.
- Manifest save/load preserves all cumulative failures.

---

## M4 — Make Hydra/dataclass validation reject misplaced or unknown canonical fields

### Problem

A composition test that removes unknown keys before dataclass instantiation cannot prove the real runtime config is schema-correct.

### Required behavior

The pre-GPU config check must:

1. compose the real canonical Hydra config;
2. instantiate the same typed config path used by the worker;
3. fail on unknown or misplaced canonical fields;
4. verify `mean` and `sum` overrides reach `actor.policy_loss`;
5. verify no duplicate top-level actor field silently overrides `policy_loss`.

Do not sanitize the config into a hand-picked subset before the only schema test. If upstream runtime intentionally ignores extra runtime fields, test the exact runtime conversion function and separately assert canonical fields are located correctly.

### Acceptance tests

- Moving `ratio_threshold` to the wrong level fails or is demonstrably ignored with an explicit warning.
- Invalid `segment_token_reduction` fails before training.
- The full composed actor config round-trips through the real worker conversion.

---

# Medium-task completion gate

Run the easy guide first, then add and run:

```bash
python -m pytest \
  verl/recipe/gear_tree/tests/test_trainer_transaction_lifecycle.py \
  verl/recipe/gear_tree/tests/test_strict_tree_identity.py \
  verl/recipe/gear_tree/tests/test_manifest_observed_facts.py \
  verl/recipe/gear_tree/tests/test_run_manifest.py -q

python scripts/check_hydra_composition.py
```

Do not mark a medium task complete from source-string guards alone. At least one test must execute the production helper or runtime conversion path changed by that task.
