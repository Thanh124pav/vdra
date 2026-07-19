# Claude Fix Instructions — Conflict-Safe Index

Read `PLAN.md` first.

The previous instruction set allowed local fixes to redefine host-framework
semantics. That is no longer permitted.

## Mandatory rule

```text
PRESERVE VERL SEMANTICS BY DEFAULT.
DO NOT CREATE A NEW CROSS-CUTTING CONTRACT SILENTLY.
```

Before changing code, prepare an impact map:

```text
symbol / behavior changed
current VERL meaning
all known consumers
semantics preserved
compatibility risks
tests
```

If a change can affect any of the following, stop and discuss with the user
before implementation:

```text
global_step
total_training_steps
scheduler cadence
checkpoint/save/eval units
policy objective or weights
replay sampling/age/consumption
zero-batch handling
optimizer ownership
FSDP/DDP scaling
public config schema
```

Do not broaden a patch just because a test reveals an unrelated conflict.
Report the conflict and wait.

---

# Canonical counter contract

```text
rollout_iteration
    one generation/replay cycle
    replay-age unit

global_step
    one successful outer VERL actor update
    training loop/checkpoint/save/eval unit

num_optimizer_steps_total
    separate observational internal PPO-update metric
    never drives the outer loop or scheduler

scheduler
    one step per successful update_actor call
```

For one normal 512/128/1 update:

```text
rollout_iteration         += 1
global_step               += 1
scheduler steps           += 1
num_optimizer_steps_total += 4
```

---

# Execution order

```text
1. CLAUDE_FIX_EASY.md
2. CLAUDE_FIX_MEDIUM.md
3. CPU integration gate
4. short GPU smoke
5. CLAUDE_FIX_HARD.md verification only
```

Hard production changes require explicit user approval.

## Easy

Use `CLAUDE_FIX_EASY.md` for:

```text
disabling/removing the all-zero shortcut
replay validation before tensorization
strict missing-advantage validation
stale comments/spec corrections
host-counter regression tests
```

Easy tasks must not alter scheduler, global-step, total-step, FSDP, replay-unit,
or objective semantics.

## Medium

Use `CLAUDE_FIX_MEDIUM.md` for:

```text
restoring three-counter separation
pre-actor reservation rollback
strict tree_instance_id / derived edge_id
canonical manifest cleanup
real Hydra/dataclass validation
```

Medium changes require production success-path and failure-path tests.

## Hard / discussion-gated

Use `CLAUDE_FIX_HARD.md` for analysis and verification of:

```text
actual FSDP/FSDP2 parity
scheduler-per-internal-step proposals
per-mini-batch zero skipping
global-step/total-step redesign
post-actor anomaly policy
distributed scaling changes
```

Except for verification, do not implement these automatically.

---

# Completion rule

A task is not complete merely because:

```text
a commit has the task name
a helper test passes
a source-string guard passes
a config contains the desired value
```

A task is complete only when:

```text
production success path passes
production failure path passes
host-framework semantics were preserved or explicitly approved
all affected consumers were audited
no new conflict was introduced
```

Every completion report must include:

```text
files changed
behavior changed
behavior preserved
consumers audited
tests run
unresolved risks
whether a conflict discussion is required
```

If a conflict is discovered, do not guess the user's preferred architecture.