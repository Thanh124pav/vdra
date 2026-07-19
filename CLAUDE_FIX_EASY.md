# Claude Fix Guide — EASY Tasks

> Scope: small, local fixes that must preserve VERL's existing training-loop,
> scheduler, checkpoint, and distributed semantics.

Read `PLAN.md` first.

## Non-negotiable safety rule

Before editing, confirm that the task does **not** change:

```text
global_step meaning
total_training_steps unit
scheduler cadence
checkpoint/save/eval unit
policy objective
replay sampling unit
FSDP/DDP scaling
```

If it might, stop and discuss with the user instead of implementing it here.

Canonical host contract to preserve:

```text
rollout_iteration         += 1 per generation/replay cycle
global_step               += 1 per successful outer actor update
scheduler                 += 1 per successful update_actor call
num_optimizer_steps_total += internal PPO optimizer-batch count for logging only
```

---

## E1 — Disable/remove the canonical all-zero shortcut

### Why

The whole-reservation shortcut was introduced as an optional optimization. It
created two avoidable problems:

```text
missing advantage could be interpreted as zero
rows could be committed before replay validation
```

Per-mini-batch zero skipping would introduce additional conflicts with AdamW,
entropy/KL terms, scheduler cadence, and replay consumption. It is not approved.

### Required behavior

Canonical main should use the dense actor path:

```text
reserve
→ validate
→ tensorize
→ update_actor
→ commit
```

Remove or disable the production call to:

```python
batch_has_zero_learning_signal(...)
```

The helper may remain for experiments/tests, but it must not alter canonical
main execution.

### Acceptance tests

- An all-zero but valid reservation still reaches the normal actor path.
- No special zero shortcut commits replay rows.
- No individual 128-row zero batch is skipped.
- Scheduler and global-step behavior remain unchanged.

---

## E2 — Validate replay rows before tensorization

### Required order

After reservation and divisibility checks:

```text
validate_replay_batch(sampled_edges)
→ edges_to_dataproto(sampled_edges)
→ update_actor
```

Replay validation must check at least:

```text
edge_id exists and is unique
question_id exists
generation_rollout_iteration exists
age is valid
advantage exists and is not None
stored log-probs align with response tokens
selected count <= target
per-question count <= resolved cap
```

### Missing advantage rule

Do not use:

```python
float(edge.get("advantage") or 0.0)
```

Use explicit validation:

```python
if "advantage" not in edge or edge["advantage"] is None:
    raise ValueError("sampled edge is missing training advantage")
```

The authoritative value is the exact `advantage` scalar that tensorization
broadcasts into the actor tensor.

### Acceptance tests

- Missing `advantage` raises before tensorization.
- `advantage=None` raises before tensorization.
- `advantage=0.0` is valid data and follows the dense actor path.
- Invalid age, duplicate ID, or log-prob mismatch cannot reach actor update.

---

## E3 — Correct stale comments and specification claims

Remove comments that claim any of the following:

```text
global_step is the internal optimizer-step count
scheduler must step once per internal PPO mini-batch
total_training_steps must use optimizer-step units
canonical segment mean requires segment_objective_weights
zero-signal mini-batches must be skipped
```

Canonical wording:

```text
global_step = outer VERL actor-update count
scheduler = one step per outer update_actor call
num_optimizer_steps_total = separate observational metric
one selected replay slot = one equal outer-weight segment sample
tree/queue counts = construction diagnostics only
```

Target files include:

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/policy_loss.py
verl/recipe/gear_tree/run_manifest.py
scripts/pre_gpu_check.sh
```

### Acceptance guard

A source/config test should fail if canonical documentation again says:

```text
global_step += 4 for 512/128
scheduler += 4 for one update_actor
segment_objective_weights are required by vdra_segment_mean_ppo
```

---

## E4 — Add host-contract regression tests

Add focused tests that lock the unchanged framework contract before the medium
counter fix is implemented.

Required expectations:

```text
five successful outer actor updates
→ global_step increases by five
→ scheduler advances five times
→ total_training_steps=5 permits five updates

if each actor update reports four internal PPO optimizer batches
→ num_optimizer_steps_total increases by twenty
→ global_step is still five
```

These tests may use a lightweight production trainer/worker harness, but they
must exercise the real counter mutation code rather than a rewritten formula.

Do not modify the scheduler or base VERL trainer merely to make the tests pass.
If the test exposes a wider contract conflict, report it and stop.

---

# Easy completion gate

Run at minimum:

```bash
python -m pytest \
  verl/recipe/gear_tree/tests/test_zero_adv_production_skip.py \
  verl/recipe/gear_tree/tests/test_construction_vs_replay_validation.py \
  verl/recipe/gear_tree/tests/test_threshold_crossing_and_logging.py \
  verl/recipe/gear_tree/tests/test_actor_update_control_flow.py -q
```

Completion report must state:

```text
what changed
what framework semantics were intentionally preserved
which production path was executed
whether any cross-cutting conflict was found
```

Do not begin a medium task automatically if a conflict is found.