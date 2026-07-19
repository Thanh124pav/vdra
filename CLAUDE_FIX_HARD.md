# Claude Fix Guide — HARD Tasks

> Scope: architecture-level refactors. Do not patch these as isolated one-line fixes. Complete `CLAUDE_FIX_EASY.md` and `CLAUDE_FIX_MEDIUM.md` first so failure handling, IDs, and manifest semantics are stable.

The hard work has one central goal:

```text
one unit called "optimizer step" must mean the same thing
in the actor, scheduler, trainer loop, checkpoint, logs, and manifest
```

## H1 — Unify training duration, optimizer-step counting, and LR scheduling

### Current inconsistency

The current system mixes three units:

```text
total_training_steps = dataloader/rollout iterations
trainer global_step  = reported optimizer-step calls
scheduler.step()     = update_actor calls
```

With 512 selected edges and mini-batch 128:

```text
one rollout iteration
→ four optimizer batches
→ four optimizer.step() calls
```

If `total_training_steps` remains the number of rollout batches while `global_step` increases by four, training can end about four times early. If the scheduler advances once per `update_actor`, its schedule is also four times slower than the optimizer-step counter.

### Non-negotiable final semantics

```text
rollout_iteration
    One generation/replay-fill cycle.
    Used for replay age and rollout-frequency reporting.

global_step
    One successful actual optimizer.step().
    Used for LR schedule, optimizer-step checkpoints, and optimizer-step logs.

total_training_steps
    Total successful optimizer steps planned for the run.
```

A separate config/derived field may express planned rollout iterations:

```text
total_rollout_iterations
```

Do not overload `total_training_steps` with both units.

### Required design choices

Choose one explicit configuration contract:

#### Preferred contract

```yaml
trainer:
  total_rollout_iterations: <derived from dataloader × epochs>
  total_training_steps: null  # resolved to optimizer-step budget
```

Resolve optimizer-step budget from the planned canonical sample schedule, or require an explicit optimizer-step value when exact derivation is impossible because replay underfill/zero batches can vary.

At minimum, never compare an optimizer-step `global_step` against a rollout-iteration total.

### Scheduler placement

Move scheduler advancement to the same place as each successful optimizer step. Do not call `actor_lr_scheduler.step()` once after the whole `update_policy()` call when that call can perform multiple optimizer steps.

Possible implementation:

```python
did_step, grad_norm = self._optimizer_step()
if did_step:
    self.actor_lr_scheduler.step()
    num_optimizer_steps += 1
```

If scheduler ownership must remain in the worker, pass it into the actor or add a callback/helper that runs inside the optimizer-batch loop.

### Successful-step accounting

Change `_optimizer_step()` to report whether a parameter update occurred:

```python
@dataclass
class OptimizerStepResult:
    grad_norm: torch.Tensor
    did_step: bool
```

For non-finite gradients:

```text
zero gradients
no optimizer.step()
no scheduler.step()
no global-step increment
record a failure/skip metric
```

Do not increment `num_optimizer_steps` merely because `_optimizer_step()` was called.

### Required files

```text
verl/verl/trainer/ppo/ray_trainer.py
verl/verl/workers/actor/dp_actor.py
verl/verl/workers/fsdp_workers.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/trainer_state.py
verl/recipe/gear_tree/run_manifest.py
```

### Required metrics

```text
training/rollout_iteration
training/global_step
training/optimizer_steps_this_iteration
training/planned_total_optimizer_steps
training/planned_total_rollout_iterations
actor/optimizer_step_attempts
actor/optimizer_steps_successful
actor/optimizer_steps_skipped_nonfinite
actor/scheduler_steps
```

### Acceptance tests

- Four successful optimizer batches produce four optimizer steps and four scheduler steps.
- One non-finite batch produces one attempt, zero successful steps, zero scheduler steps, and no global-step increment.
- One epoch over `N` planned rollout iterations does not terminate after approximately `N/4` iterations.
- Resume preserves scheduler state, optimizer-step global step, and rollout iteration.
- Warmup/cosine schedule receives the same total-step unit as `global_step`.

---

## H2 — Skip zero-signal data at the optimizer-batch level

### Current limitation

A whole 512-edge reservation may contain one all-zero 128-edge optimizer batch and three nonzero batches. A whole-reservation zero check cannot detect this. The zero batch can still call optimizer/scheduler logic and be counted as a step.

### Required behavior

Inside the actor's optimizer-batch loop:

```python
for mini_batch in mini_batches:
    if mini_batch_has_zero_training_signal(mini_batch):
        record skipped-zero-batch metric
        continue

    zero_grad
    backward over microbatches
    successful optimizer step
    scheduler step
```

Use the exact `advantages` tensor and effective response/probability mask used by the loss. A batch is zero signal only if its canonical policy-loss numerator is guaranteed to be zero.

Do not infer this from one scalar metadata field after tensorization if masking can remove all active contributions.

### Counter consequences

Expected optimizer steps can no longer always be computed only from:

```text
selected_count / ppo_mini_batch_size
```

Track separately:

```text
optimizer_batches_planned
optimizer_batches_zero_signal
optimizer_steps_attempted
optimizer_steps_successful
```

The manifest should compare observed successful steps with a runtime-derived expected-successful count after zero-batch classification, not a precomputed floor formula that assumes every batch is trainable.

### Replay policy

Document and test whether rows from a zero-signal optimizer batch are consumed. Recommended:

```text
commit consumed zero-signal rows
```

so they are not repeatedly sampled until expiration.

### Acceptance tests

- Four mini-batches with one zero batch produce three optimizer and scheduler steps.
- Global step advances by three.
- Zero-batch rows are handled according to the documented replay policy.
- A fully zero reservation produces zero actor optimizer steps.
- Mean and sum token reductions agree on whether the batch is zero signal when masks are identical.

---

## H3 — Verify actual FSDP/FSDP2 production gradient semantics

### Current evidence

Two-process Gloo/DDP parity is useful and should remain. It verifies equal-sized disjoint shards under an averaging reducer.

It does not fully prove:

```text
actual FSDP/FSDP2 reducer behavior
VERL actor dispatch and sharding manager
sequence-parallel interaction
mixed precision reduction
production optimizer loop
```

### Required integration coverage

Add at least one production-oriented distributed test or short GPU smoke that uses:

```text
actual DataParallelPPOActor
actual FSDP or FSDP2 wrapper
actual local mini-batch normalization from VERL worker config
actual vdra_segment_mean_ppo loss
mean and sum reductions
```

Compare against a single-rank reference using the same initial weights and exact selected rows.

Test at minimum:

- world size 1 versus 2;
- equal shards;
- uneven token lengths;
- microbatch splitting;
- BF16/FP32 accumulation behavior where supported;
- gradient norm and parameter delta after one step.

Do not claim FSDP parity from the toy DDP test alone.

### Acceptance criteria

- Single-rank and distributed parameter deltas match within documented tolerance.
- No extra world-size division or multiplication appears.
- Reported optimizer-step counts agree on all ranks.
- Scheduler advances exactly once per successful global optimizer step.

---

## H4 — Update checkpoint and interval semantics after the step refactor

After H1/H2, re-audit:

```text
checkpoint folder naming
trainer state JSON
scheduler state
next save threshold
next eval threshold
last-step detection
manifest total-step fields
```

A jump in successful global step must still trigger crossed save/eval thresholds. A zero/non-finite skipped batch must not falsely cross a threshold.

### Acceptance tests

- Resume at global step 400 and rollout iteration 100 preserves both units.
- Scheduler resumes at step 400, not actor-update count 100.
- A transition 8→12 fires a step-10 threshold once.
- A skipped optimizer batch leaves step-based thresholds unchanged.

---

# Hard-task completion gate

The following are required before long experiments:

```bash
bash scripts/pre_gpu_check.sh
```

The gate must include new tests for:

```text
total-step unit consistency
scheduler step count
non-finite did_step accounting
per-optimizer-batch zero skip
transaction lifecycle
all-worker count agreement
FSDP/FSDP2 production parity or a recorded GPU smoke artifact
```

Then run a short GPU smoke for at least five rollout iterations and report:

```text
rollout iterations completed
optimizer batches planned
optimizer steps successful
scheduler steps
zero-signal batches skipped
non-finite steps skipped
global step
```

Do not begin a long paper experiment until these counts are mutually consistent and the run manifest remains valid.
