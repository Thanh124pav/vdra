# PLAN.md — Remaining Work Before GPU Smoke

## Purpose

This is the current source of truth for branch:

```text
claude/plan-tasks-execution-iih9zc
```

The branch has implemented most historical P0.A–P0.K items, but the latest production audit found several cross-cutting regressions that are not covered by the old task-completion claims.

Do not treat a commit name, passing helper test, or source-string guard as proof of completion. A task is complete only when the production path and its failure path satisfy the contract below.

Implementation work is split by difficulty:

```text
CLAUDE_FIX_EASY.md
CLAUDE_FIX_MEDIUM.md
CLAUDE_FIX_HARD.md
```

Complete them in that order unless a dependency explicitly requires otherwise.

---

# 1. Canonical behavior

## 1.1 Rollout and replay

```text
ONE ROLLOUT ITERATION

Generate complete stochastic trees
    ↓
Validate full-tree construction invariants
    ↓
Convert every realized non-placeholder segment into one replay edge
    ↓
Stamp generation_rollout_iteration and unique tree/edge identity
    ↓
Insert edges transactionally
    ↓
Expire by rollout_iteration age
    ↓
Reserve individual edges, capped by question_id
    ↓
Select at most 512 total edges
    ↓
Validate sampled replay rows
    ↓
Train optimizer batches of 128 selected edges
```

Canonical defaults:

```yaml
replay_buffer:
  target_edges_per_iteration: 512
  max_edge_age_iterations: 8
  max_edges_per_question_per_iteration: auto
  replay_sampling_unit: edge
  underfilled_update_policy: postpone_until_divisible

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 128
    ppo_epochs: 1
    policy_loss:
      loss_mode: vdra_segment_mean_ppo
      segment_token_reduction: mean

tree_policy:
  policy_aggregation: global_segment_mean
  segment_token_reduction: mean
```

## 1.2 Canonical policy objective

For one global optimizer batch `B`:

```text
N_B = number of selected replay slots in that optimizer batch
```

Normally `N_B=128`.

For each selected segment `s`, the loss first reduces over active tokens using either token mean or token sum. The outer objective is:

\[
L_B^{(r)}
=
\frac{1}{N_B}
\sum_{s\in B}L_s^{(r)},
\qquad
r\in\{\mathrm{mean},\mathrm{sum}\}.
\]

Every selected replay slot has equal outer weight.

The following must not affect canonical policy weight:

```text
tree size
parent group size
allocated_k
queue_flush_id
branch factor
replay age
segment_objective_weights
objective_weights
```

Tree and queue counts remain construction/theory diagnostics only.

## 1.3 Counter units

```text
rollout_iteration
    One generation/replay-fill cycle.
    Used for replay age and rollout-frequency reporting.

global_step
    One successful actual optimizer.step().
    Used for LR scheduling, optimizer-step checkpoints, and step-based logs.

optimizer_steps_this_iteration
    Successful optimizer steps during the current rollout iteration.
```

For a fully trainable 512-edge reservation with mini-batch 128 and one PPO epoch:

```text
rollout_iteration += 1
global_step += 4
scheduler steps += 4
```

The phrase “successful optimizer step” excludes:

```text
non-finite gradient attempts that skip optimizer.step()
zero-signal optimizer batches that are intentionally skipped
failed actor calls
```

---

# 2. Current audit status

## 2.1 Implemented correctly enough to preserve

| Area | Status | Notes |
|---|---|---|
| Canonical edge-level replay dispatch | DONE | Strictness no longer chooses complete-tree replay. |
| Auto per-question cap | DONE on edge path | `666→33`, `888→73`; hard target cap is active. |
| Complete-tree replay | ABLATION ONLY | No longer canonical. |
| Mean/sum schema and resolver | DONE | Reads `actor.policy_loss.*`. |
| Canonical DataProto weight tensors | DONE | Main path omits float objective-weight tensors. |
| Default 512/128/1 configuration | DONE | Config and actor control-flow tests exist. |
| Checkpointed `rollout_iteration` | DONE | Legacy checkpoint replay reset is documented. |
| Crossed save/eval thresholds | DONE for reported global-step jumps | Must be re-audited after hard step refactor. |
| Actor-observed stored old log-probs | DONE | Manifest bit is no longer inferred from tensor presence alone. |
| Two-process DDP parity | DONE as DDP evidence | Not sufficient proof for FSDP/FSDP2 production. |

## 2.2 Claimed complete but still incomplete

| Area | Real status | Remaining issue |
|---|---|---|
| P0.B validation split | PARTIAL | Replay validation still occurs after tensorization; pre-actor exceptions may not rollback. |
| P0.C obsolete weights | PARTIAL | DataProto is clean, but canonical manifest still validates tree-normalized segment weights. |
| P0.D optimizer-step accounting | PARTIAL | Worker counts may disagree; mismatch is checked after replay/counters mutate. |
| P0.G zero-adv handling | BLOCKED | Missing advantage can be treated as zero; whole-reservation shortcut can bypass validation; zero mini-batches are not skipped individually. |
| P0.H strict IDs | PARTIAL | Strict path can still accept generic `tree_id`; caller-supplied `edge_id` may bypass derivation; some collisions remain undetected. |
| P0.I distributed scaling | PARTIAL | Toy DDP parity exists, but actual FSDP/FSDP2 production path is unverified. |
| P0.J observed manifest | PARTIAL | Accounting validity can heal after a later success; canonical segment invariant still depends on obsolete weight validation. |
| P0.K pre-GPU gate | PARTIAL | Gate does not detect the current total-step/scheduler/transaction regressions. |

## 2.3 Newly discovered P0 regressions

### R1 — Training duration uses the wrong unit

Base VERL derives `total_training_steps` from dataloader/rollout iterations, while the VDRA loop compares it with optimizer-step `global_step`.

With four optimizer steps per rollout iteration:

```text
planned rollout iterations = N
global_step budget used by loop = N
actual rollouts before stop ≈ N/4
```

Unless an explicit optimizer-step budget is supplied, training can terminate about four times early.

### R2 — LR scheduler advances once per actor call, not per optimizer step

A single `update_policy()` may execute four `optimizer.step()` calls, but the worker advances the scheduler once afterward.

Current mismatch:

```text
4 optimizer steps
1 scheduler step
```

### R3 — Non-finite optimizer attempts are counted as successful steps

`_optimizer_step()` can skip `optimizer.step()` when gradient norm is non-finite, but the actor still increments `num_optimizer_steps` after the call.

### R4 — Reserved-update failure handling is not fully transactional

Failures in replay validation or tensorization can occur after reservation without guaranteed rollback. Actor-result mismatch is checked after model/replay/counter mutation.

### R5 — All-zero shortcut can bypass replay validation

A missing `advantage` can be interpreted as zero, and the all-zero reservation path can commit rows before replay validation.

### R6 — Zero-signal optimizer batches inside a mixed reservation still step

A 512-edge reservation can contain one zero-signal 128-edge optimizer batch and three trainable batches. The current whole-reservation shortcut cannot skip only the zero batch.

---

# 3. Work split by difficulty

## 3.1 Easy tasks

Source of truth:

```text
CLAUDE_FIX_EASY.md
```

| ID | Task | Why it is easy |
|---|---|---|
| E1 | Move replay validation before zero shortcut and tensorization | Local control-flow reorder. |
| E2 | Missing/`None` advantage must raise | Small predicate fix. |
| E3 | Require identical worker optimizer-step reports | Local actor-result validation. |
| E4 | Verify actual/expected steps before replay/counter commit | Small state-mutation reorder. |
| E5 | Add monotonic optimizer-accounting failure counter | Local manifest schema/lifecycle change. |
| E6 | Remove stale comments and canonical weight claims | Documentation/source guards only. |

Easy tasks must not redesign scheduler ownership, optimizer loop structure, or FSDP behavior.

## 3.2 Medium tasks

Source of truth:

```text
CLAUDE_FIX_MEDIUM.md
```

| ID | Task | Scope |
|---|---|---|
| M1 | Create explicit reserved-update transaction/state machine | Cross-function replay/actor/manifest lifecycle. |
| M2 | Enforce strict `tree_instance_id` and derived `edge_id` | Tree builder, extraction, normalization, manifest. |
| M3 | Decouple canonical manifest from objective-weight validation | Manifest and construction invariants. |
| M4 | Make real Hydra/dataclass validation reject misplaced fields | Runtime config conversion and pre-GPU gate. |

Medium tasks require production-path failure-injection tests. Source-string tests alone are insufficient.

## 3.3 Hard tasks

Source of truth:

```text
CLAUDE_FIX_HARD.md
```

| ID | Task | Scope |
|---|---|---|
| H1 | Unify total-step unit, successful optimizer steps, scheduler and global step | Base trainer, actor, FSDP worker, checkpoint, manifest. |
| H2 | Skip zero signal at each 128-edge optimizer batch | Actor optimizer loop and runtime expected-step accounting. |
| H3 | Verify actual FSDP/FSDP2 production gradient semantics | Distributed integration/GPU smoke. |
| H4 | Re-audit checkpoint/save/eval semantics after H1/H2 | Scheduler/counter resume and interval thresholds. |

Do not implement H1 as independent one-line patches. `total_training_steps`, scheduler stepping, `did_step`, global counters, and resume semantics must change together.

---

# 4. Required execution order

```text
Stage 1 — EASY
E1 → E2 → E3 → E4 → E5 → E6

Stage 2 — MEDIUM
M1 → M2 → M3 → M4

Stage 3 — HARD
H1 → H2 → H4 → H3

Stage 4 — INTEGRATION
full CPU gate
short GPU smoke
manifest review
long experiment only after all pass
```

Dependency notes:

- M1 relies on E1–E4 to define correct validation and commit order.
- H1 relies on E3/E5 and M1 so step reports and failure state are trustworthy.
- H2 must be implemented after or together with H1 because zero-batch skips change the number of successful optimizer and scheduler steps.
- H4 follows H1/H2 because checkpoint thresholds depend on the final step semantics.
- H3 is last because distributed parity must test the final optimizer/scheduler loop.

---

# 5. Production transaction contract

The final reserved-update flow must be:

```text
reserve replay rows
    ↓
validate sampled replay rows
    ↓
classify zero/trainable optimizer batches
    ↓
tensorize
    ↓
actor forward/backward/optimizer loop
    ↓
collect all worker reports
    ↓
verify rank agreement and runtime step accounting
    ↓
commit replay rows
    ↓
update driver counters and manifest
```

Failure semantics:

```text
before actor starts:
    rollback replay
    no counter mutation
    no success manifest facts

during actor RPC:
    rollback replay
    record failed update
    re-raise

after model may have changed but actor result is invalid:
    do not claim rollback of model
    do not commit replay/counters as successful
    mark run irrecoverably invalid
    abort strict training
```

---

# 6. Definition of done

The branch is ready for a GPU smoke only when:

```text
[ ] Every EASY task passes its production regression tests
[ ] Every MEDIUM task passes failure-injection tests
[ ] total_training_steps uses optimizer-step units or is explicitly separated from total_rollout_iterations
[ ] scheduler steps exactly once per successful optimizer.step()
[ ] non-finite attempts do not increment scheduler/global step
[ ] zero-signal 128-edge optimizer batches are skipped individually
[ ] replay validation always precedes zero shortcuts and tensorization
[ ] all pre-actor failures rollback reservations
[ ] all worker optimizer-step reports must agree
[ ] actor-result mismatch is detected before replay/counter commit
[ ] canonical manifest never depends on objective-weight normalization
[ ] strict main requires tree_instance_id and derived edge_id
[ ] accounting failures are monotonic and non-healing
[ ] DDP evidence remains green
[ ] FSDP/FSDP2 production parity or an equivalent GPU smoke is recorded
[ ] scripts/pre_gpu_check.sh prints PRE_GPU_CHECK=PASS
```

Then run at least five rollout iterations and report:

```text
rollout iterations completed
optimizer batches planned
zero-signal optimizer batches skipped
optimizer-step attempts
successful optimizer steps
non-finite steps skipped
scheduler steps
global step
```

For a fully trainable 512/128/1 run, the expected relation is:

```text
5 rollout iterations
20 successful optimizer steps
20 scheduler steps
global_step = 20
```

Do not launch long paper experiments until these counts are consistent after both a fresh run and checkpoint resume.
