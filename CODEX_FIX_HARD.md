# Claude Fix Guide — HARD / Discussion-Gated Tasks

> Scope: changes that may alter host-framework semantics, optimizer behavior,
> scheduler cadence, replay consumption, or distributed scaling.
>
> **Do not implement these tasks automatically.** Test and analyze first, then
> discuss the conflict and options with the user.

Complete the easy and medium guides first.

---

# 0. Approval gate

Before proposing a hard code change, provide:

```text
current behavior
why it is incorrect or insufficient
whether it is a correctness issue, metric issue, or optional optimization
all affected consumers
option A: preserve framework
option B: redesign framework
migration and test cost of each option
recommended option
```

Wait for explicit user approval before editing production code.

A failing toy/unit test is not sufficient reason to alter FSDP, scheduler,
checkpoint, replay, or global-step semantics.

---

## H1 — Verify actual FSDP/FSDP2 semantics before changing scaling

### Current evidence

The two-process Gloo/DDP parity test is useful. It does not fully prove:

```text
actual FSDP/FSDP2 reducer behavior
VERL actor dispatch and sharding manager
sequence-parallel interaction
mixed-precision reduction
production parameter delta
```

### Required action

Test first. Do not patch scaling first.

Use a short production-oriented test or GPU smoke with:

```text
actual DataParallelPPOActor
actual FSDP or FSDP2 wrapper
actual VERL batch normalization
vdra_segment_mean_ppo
mean and sum reductions
same initial weights and selected rows for reference comparison
```

Report:

```text
single-rank loss and parameter delta
distributed loss and parameter delta
local/global batch sizes
reducer behavior
gradient norm tolerance
any world-size discrepancy
```

### If a mismatch is found

Stop and discuss before adding any world-size multiplication or division.
Present the exact reducer and dispatch evidence. Do not infer a production fix
from the toy DDP test alone.

---

## H2 — Scheduler-per-internal-optimizer-step redesign

### Status

```text
NOT APPROVED
```

Preserved VERL behavior is:

```text
one update_actor call
→ one outer global_step
→ one scheduler.step()
```

Moving `scheduler.step()` into the PPO mini-batch loop would change:

```text
warmup length
cosine/constant schedule horizon
resume state
learning-rate values for existing configs
comparison with previous runs
meaning of total_training_steps
```

Do not implement this as a fix for the internal count being four. The separate
internal optimizer metric already provides the TreeTune-style x-axis.

If the user later considers this redesign, first provide two experiment plans:

```text
A. preserve scheduler per outer update
B. scheduler per internal optimizer update with migrated total-step/warmup units
```

No production edit before approval.

---

## H3 — Per-mini-batch zero-signal skipping

### Status

```text
NOT APPROVED / OPTIONAL OPTIMIZATION
```

Skipping an internal PPO mini-batch because its policy-gradient numerator is
zero may still change behavior through:

```text
AdamW weight decay
entropy loss
KL loss
optimizer state
scheduler cadence
replay row consumption
reported optimizer counts
```

Canonical submission-first behavior is dense processing with no zero shortcut.

Do not add:

```python
if mini_batch_has_zero_signal:
    continue
```

unless the user explicitly chooses a precise definition of zero signal and a
row-consumption/scheduler policy.

If discussed later, the proposal must specify behavior for mean/sum reduction,
masked rows, entropy/KL terms, AdamW, and replay consumption.

---

## H4 — Redefining `global_step` or `total_training_steps`

### Status

```text
PROHIBITED WITHOUT EXPLICIT APPROVAL
```

Canonical preserved contract:

```text
global_step = successful outer VERL actor updates
total_training_steps = planned outer VERL updates
scheduler = one step per successful outer update
```

Do not redefine `global_step` as internal optimizer-step count. Do not convert
`total_training_steps` to optimizer-step units merely to make the names match.

For optimizer-update plots, use:

```text
num_optimizer_steps_total
```

as a separate observational axis.

Any future redesign must migrate all consumers together:

```text
outer loop
scheduler
warmup
checkpoint names
resume
save/eval frequencies
logging
policy snapshot IDs
manifest
analysis scripts
```

This is an architecture change, not a local bug fix.

---

## H5 — Changing post-actor replay/abort semantics

After actor RPC returns, model parameters may already have changed. A metric
mismatch cannot be undone by replay rollback.

The following choices have different scientific and operational meanings:

```text
commit rows and mark run invalid
rollback rows but keep changed model
abort immediately after saving failure state
continue non-strictly with warning
```

Do not choose among them automatically.

Under the current preserved contract, internal optimizer-step reports are
observational and do not drive outer `global_step`. Pre-actor failures must
rollback; post-actor metric anomalies should be recorded and discussed before
becoming a strict abort policy.

---

# Hard verification gate

Hard verification may be run after easy/medium tasks:

```bash
bash scripts/pre_gpu_check.sh
```

Then run a short GPU smoke and report:

```text
rollout_iteration
global_step
scheduler/LR observations
num_optimizer_steps_total
selected edges and replay ages
single-rank/distributed comparison where available
manifest verdict
```

A hard task is complete only when either:

```text
verification confirms no production change is needed
```

or:

```text
the user approved a specific design after conflict review,
the complete migration was implemented,
and compatibility tests passed
```

Do not turn an optional optimization into a P0 correctness requirement.