# Claude Fix Guide — EASY Tasks

> Scope: small, local production fixes. Do these first. Each task should be one focused commit. Do not redesign the optimizer, scheduler, replay format, or tree builder in this file.

Read `PLAN.md` first. The canonical main path is:

```text
edge-level replay
512 selected edges per rollout iteration
128 selected edges per global optimizer batch
one PPO epoch
batch-slot segment mean
```

## E1 — Validate replay rows before any shortcut or tensorization

### Current risk

The trainer can currently inspect an all-zero batch or tensorize rows before replay-batch validation. A malformed reserved row can therefore bypass validation or raise without a clean rollback.

### Required order

Inside `RayGearTreeTrainer.fit`, after reservation and divisibility checks:

```text
validate_replay_batch(sampled_edges)
→ all-zero reservation check
→ edges_to_dataproto(sampled_edges)
→ actor update
```

Validation must run before:

```text
batch_has_zero_learning_signal(...)
_edges_to_update_batch(...)
```

### Required files

```text
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/replay_buffer.py
verl/recipe/gear_tree/tests/test_zero_adv_production_skip.py
verl/recipe/gear_tree/tests/test_construction_vs_replay_validation.py
```

### Acceptance tests

- A row missing `advantage` raises before the all-zero shortcut.
- A row with invalid age raises before tensorization.
- Duplicate IDs and log-prob/response mismatch cannot be committed by the all-zero path.
- A valid all-zero reservation is still consumed according to the documented policy.

---

## E2 — Missing training advantage is an error, not zero

### Current bug

Do not use:

```python
float(edge.get("advantage") or 0.0) == 0.0
```

That treats a missing or `None` advantage as a legitimate zero contribution.

### Required behavior

```python
for edge in edges:
    if "advantage" not in edge or edge["advantage"] is None:
        raise ValueError("sampled edge is missing training advantage")

return all(float(edge["advantage"]) == 0.0 for edge in edges)
```

Do not use `pav_advantage`, `prover_advantage`, or a diagnostic field. The exact scalar broadcast into the actor `advantages` tensor is authoritative.

### Acceptance tests

- `advantage=0.0` is zero signal.
- `advantage=-0.0` is zero signal.
- Missing `advantage` raises.
- `advantage=None` raises.
- Nonzero `advantage` prevents the whole-reservation zero shortcut.

---

## E3 — Require identical optimizer-step counts from all workers

### Current bug

Do not collapse worker reports using:

```python
n_optim_steps = max(step_ints)
```

A report such as `[4, 3]` must be a hard consistency failure, not silently interpreted as four steps.

### Required behavior

For canonical strict runs:

```python
if not step_ints:
    raise RuntimeError("actor did not report optimizer-step count")
if len(set(step_ints)) != 1:
    raise RuntimeError(
        f"inconsistent optimizer-step counts across actor workers: {step_ints}"
    )
n_optim_steps = step_ints[0]
```

Legacy fallback-to-one may remain only behind an explicit non-canonical compatibility path.

### Acceptance tests

- `[4, 4]` resolves to four.
- `[4, 3]` raises before replay commit and counter mutation.
- Missing metric raises in strict canonical mode.

---

## E4 — Verify expected step count before replay/counter commit

### Current bug

The actor-result consistency assertion must not run after replay rows are removed and counters are incremented.

### Required order after actor returns

```text
parse all worker reports
→ verify all workers agree
→ compute expected steps
→ verify actual == expected
→ only then commit replay
→ only then update counters and manifest
```

Expected count is valid only after divisibility enforcement:

```python
expected = (
    selected_count // ppo_mini_batch_size
) * ppo_epochs
```

### Important limitation

This does not undo model parameters if the actor already performed an incorrect number of steps. The medium guide introduces the transaction/state-transition helper. This easy fix only prevents replay and driver counters from becoming inconsistent as well.

### Acceptance tests

- An actor report mismatch leaves replay reservation uncommitted.
- Driver counters remain unchanged on mismatch.
- A valid 512/128/1 result commits and advances by four.

---

## E5 — Make optimizer accounting failures monotonic in the manifest

### Current bug

Do not overwrite a historical failure with a later successful iteration:

```python
optimizer_step_accounting_valid = actual == expected
```

### Required manifest field

Add:

```python
optimizer_step_accounting_failures: int = 0
```

Update after each observed actor result:

```python
if actual_optimizer_steps != expected_optimizer_steps:
    manifest.optimizer_step_accounting_failures += 1

manifest.optimizer_step_accounting_valid = (
    manifest.num_optimizer_steps_total > 0
    and manifest.optimizer_step_accounting_failures == 0
)
```

`validate_main_run()` must reject any manifest with a positive failure count.

### Acceptance tests

- First mismatch sets the counter to one and invalidates the run.
- A later correct iteration does not restore validity.
- Save/load preserves the failure counter.

---

## E6 — Remove stale canonical comments and obsolete weight claims

Update comments/documentation that still claim canonical `global_segment_mean` means a full-tree-normalized objective or requires `segment_objective_weights`.

Canonical replay objective:

```text
one selected replay slot = one equal outer-weight segment sample
one optimizer batch = equal average over its selected slots
```

Tree/queue counts remain construction diagnostics only.

### Target files

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/tree_data.py
verl/recipe/gear_tree/policy_loss.py
verl/recipe/gear_tree/run_manifest.py
```

### Acceptance test

A source/config guard should fail if canonical comments again state that `segment_objective_weights` are required by `vdra_segment_mean_ppo`.

---

# Easy-task completion command

Add the new regressions to the existing CPU gate, then run at minimum:

```bash
python -m pytest \
  verl/recipe/gear_tree/tests/test_zero_adv_production_skip.py \
  verl/recipe/gear_tree/tests/test_construction_vs_replay_validation.py \
  verl/recipe/gear_tree/tests/test_optimizer_step_accounting.py \
  verl/recipe/gear_tree/tests/test_manifest_observed_facts.py -q
```

Do not mark this guide complete unless the tests exercise production functions and the production trainer ordering—not a rewritten synthetic mirror.
