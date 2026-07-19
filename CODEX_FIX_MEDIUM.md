# Claude Fix Guide — MEDIUM Tasks

> Scope: cross-function production fixes that preserve the host framework while
> repairing VDRA-specific counter separation, replay failure handling, strict
> identity, manifest semantics, and config validation.

Complete `CLAUDE_FIX_EASY.md` first.

## Mandatory impact review

Before each medium task, list:

```text
symbols changed
all known consumers
semantics preserved
failure behavior before and after
new tests
```

Stop and discuss with the user if implementation would require changing:

```text
VERL global_step semantics
scheduler cadence
total_training_steps unit
save/eval/checkpoint unit
policy objective
replay consumption after an actor has already changed the model
FSDP/DDP scaling
```

---

## M1 — Restore and verify the three-counter separation

### Problem

The current branch changed `global_step` from VERL's outer-update counter into
an internal PPO optimizer-batch counter. That conflicts with the host
framework's `total_training_steps`, scheduler, checkpoint, and save/eval logic.

### Required contract

```text
rollout_iteration
    one generation / replay-fill cycle
    replay-age unit

global_step
    one successful outer update_actor call
    host-framework loop/checkpoint/log/save/eval unit

num_optimizer_steps_total
    accumulated internal PPO optimizer-batch count
    observational metric only
```

For a normal 512/128/1 update:

```text
rollout_iteration         += 1
global_step               += 1
successful_actor_updates  += 1
num_optimizer_steps_total += 4
scheduler                 += 1   # unchanged VERL behavior
```

### Required implementation

After a successful actor RPC and replay commit:

```python
self.successful_actor_updates += 1
self.global_steps += 1
self.num_optimizer_steps_total += reported_internal_steps
```

Do not set:

```python
self.num_optimizer_steps_total = self.global_steps
```

Do not modify base VERL `total_training_steps` derivation or move the scheduler
inside `DataParallelPPOActor`.

The internal count may remain the number currently reported by the actor. Until
a separate approved change distinguishes attempts from successful updates, do
not let that ambiguity control training.

### Checkpoint/resume

Persist and restore independently:

```text
global_step
rollout_iteration
num_optimizer_steps_total
successful_actor_updates
```

Checkpoint directory naming remains keyed by outer `global_step`.

### Acceptance tests

- Five successful outer updates produce `global_step=5`.
- If each reports four internal optimizer batches, total internal count is 20.
- `total_training_steps=5` permits five outer updates, not approximately two.
- Scheduler advances five times, not twenty.
- Resume restores all counters without changing their units.
- Save/eval/checkpoint consumers still use outer `global_step`.

---

## M2 — Guarantee rollback for every pre-actor failure

### Problem

Rows can be reserved before replay validation and tensorization. Exceptions in
those stages must not leave the reservation stuck.

### Required production flow

```text
reserve
→ validate sampled replay rows
→ tensorize
→ actor RPC
→ commit replay
→ update outer counters and manifest
```

Wrap the pre-actor stages and actor RPC so that:

```text
validation failure     → rollback reservation, no counter mutation
tensorization failure  → rollback reservation, no counter mutation
actor RPC exception    → rollback reservation, increment failed-update metric
actor RPC success      → commit rows, global_step += 1
```

### Important boundary

An actor RPC can change model parameters before returning metrics. Do not claim
that replay rollback can undo a model update.

Internal optimizer-step metric disagreement is diagnostic under the preserved
host contract. Do not change replay commit or outer `global_step` semantics
based on that metric without discussing the policy with the user first.

### Suggested helper

```python
_execute_reserved_actor_update(...)
```

is allowed if it only consolidates the existing semantics. It must not become
a vehicle for scheduler/counter redesign.

### Acceptance tests

Inject failures in:

```text
replay validation
tensorization
actor RPC
```

For each, assert reservation state, replay size, outer counters, and manifest
facts.

Do not add a post-model-update rollback fiction.

---

## M3 — Enforce canonical tree and edge identity strictly

### Canonical strict contract

Every stochastic tree must carry an explicit:

```text
tree_instance_id
policy_snapshot_id
rollout_iteration
stable question_id
per-tree UUID/counter component
```

Strict extraction must require `tree_instance_id` specifically. A legacy
`tree_id` alone is not sufficient.

Every edge ID must be derived deterministically from:

```text
tree_instance_id
parent_group_id
child_segment_id
```

Required strict behavior:

```python
derived_edge_id = derive_edge_id(...)
if supplied_edge_id is not None and supplied_edge_id != derived_edge_id:
    raise ValueError(...)
record["edge_id"] = derived_edge_id
```

Do not use `setdefault()` for strict IDs.

Legacy fallback behavior may remain only behind an explicit non-strict
compatibility path.

### Acceptance tests

- Two stochastic trees for the same question/snapshot/iteration get different IDs.
- Missing `tree_instance_id` fails even if `tree_id` exists.
- Reused generic `tree_id="t0"` cannot masquerade as two distinct trees.
- A mismatching caller-supplied `edge_id` fails.
- Replay insertion remains transactionally duplicate-safe.

If enforcing this breaks a public serialization/checkpoint format, stop and
report the compatibility impact before migrating it.

---

## M4 — Decouple canonical manifest invariants from obsolete weights

For `vdra_segment_mean_ppo`, manifest validity may depend on observed facts:

```text
realized non-placeholder segments counted exactly once
fresh_iid realized child count matches allocation
queue counts match tree segment count
IDs are valid and unique
stored log-probs align with response tokens
no silent truncation
replay validation passed
at least one successful outer actor update occurred
```

It must not depend on:

```text
compute_segment_objective_weights
validate_segment_objective_weights
compute_objective_weights
parent/tree-normalized float coefficients
```

Only the explicit node-balanced ablation may calculate those weights.

Failure counters must be monotonic and non-healing. Success on a later
iteration must not erase a historical failure.

Do not add internal optimizer-step equality as a canonical run requirement
unless the user approves that stronger contract. The internal count is a
runtime diagnostic, not the outer training unit.

### Acceptance tests

- Canonical manifest path does not call objective-weight helpers.
- Node-balanced ablation still validates its weights.
- Historical failures remain after later clean updates.
- Save/load preserves all counters and units.

---

## M5 — Test the real Hydra/dataclass conversion path

The pre-GPU gate must:

1. compose the real canonical Hydra config;
2. instantiate the same typed path used by the worker;
3. verify canonical fields are at the correct level;
4. verify `mean` and `sum` overrides reach `actor.policy_loss`;
5. fail or warn explicitly for misplaced canonical fields;
6. avoid proving correctness only after silently deleting unknown fields.

Do not redesign the public config schema merely because the test exposes an
upstream runtime field. Report the conflict and distinguish:

```text
canonical VDRA field misplaced
versus
legitimate upstream runtime-only field
```

---

# Medium completion gate

Run easy tests first, then at minimum:

```bash
python -m pytest \
  verl/recipe/gear_tree/tests/test_trainer_state_checkpoint.py \
  verl/recipe/gear_tree/tests/test_trainer_transaction_lifecycle.py \
  verl/recipe/gear_tree/tests/test_strict_tree_identity.py \
  verl/recipe/gear_tree/tests/test_manifest_observed_facts.py \
  verl/recipe/gear_tree/tests/test_run_manifest.py -q

python scripts/check_hydra_composition.py
```

Completion report must include a counter table like:

```text
five outer updates:
rollout_iteration = ...
global_step = 5
scheduler steps = 5
num_optimizer_steps_total = 20  # when four were reported per update
```

Do not proceed to a hard production change automatically. Hard changes require
explicit user approval as described in `CLAUDE_FIX_HARD.md`.