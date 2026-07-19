# PLAN.md — Conflict-Safe Work Before GPU Smoke

## Purpose

This is the current source of truth for branch:

```text
claude/plan-tasks-execution-iih9zc
```

The previous plan incorrectly required the repository to redefine VERL's
`global_step` as the number of internal `optimizer.step()` calls. That change
created unit conflicts with `total_training_steps`, the LR scheduler,
checkpoint naming, and save/eval intervals.

This plan corrects that decision. The default rule from now on is:

```text
PRESERVE HOST-FRAMEWORK SEMANTICS.
MAKE THE SMALLEST COMPATIBLE CHANGE.
DO NOT REDESIGN A CROSS-CUTTING CONTRACT WITHOUT USER APPROVAL.
```

Implementation work remains split into:

```text
CLAUDE_FIX_EASY.md
CLAUDE_FIX_MEDIUM.md
CLAUDE_FIX_HARD.md
```

---

# 0. Mandatory conflict-review rule

Before changing production code, write a short impact map containing:

```text
1. exact symbol / behavior being changed
2. its current meaning in VERL
3. every known consumer
4. intended new meaning, if any
5. compatibility risks
6. tests proving no existing contract was broken
```

## Stop and discuss with the user before implementation if a change may affect

```text
global_step or total_training_steps
LR scheduler ownership or cadence
checkpoint directory naming or resume units
save/eval frequency semantics
policy objective or sample weights
replay sampling unit, replay age, or row-consumption policy
optimizer-step ownership
zero-signal batch consumption
FSDP/DDP/world-size scaling
public Hydra/dataclass configuration schema
```

Claude must not silently choose a new architecture in these areas.

If a requested local fix exposes a cross-cutting conflict:

```text
STOP
summarize the conflict
show at least two compatible options
state which files and semantics each option changes
wait for user approval
```

Passing tests does not waive this rule when the change alters a framework
contract.

---

# 1. Canonical behavior to preserve

## 1.1 Host-framework counters and schedules

The canonical counter contract is:

```text
rollout_iteration
    One generation / replay-fill cycle.
    Used for replay age and rollout diagnostics.

global_step
    One successful outer actor update in the VERL trainer.
    Preserves the host framework's training-loop, checkpoint, logging,
    save/eval, and scheduler unit.

total_training_steps
    Number of outer VERL training updates planned for the run.
    It remains compatible with len(train_dataloader) * total_epochs.

num_optimizer_steps_total
    Observational metric for internal PPO optimizer-batch updates.
    It must NOT control the outer loop, scheduler, checkpoint naming,
    save/eval frequency, or policy snapshot numbering.
```

The existing VERL scheduler remains:

```text
one scheduler.step() per successful update_actor call
```

Do not move the scheduler into the internal PPO mini-batch loop unless the user
explicitly approves a scheduler redesign after an impact review.

For 512 selected replay edges, PPO mini-batch 128, and one PPO epoch:

```text
rollout_iteration         += 1
global_step               += 1
scheduler steps           += 1
num_optimizer_steps_total += 4   # diagnostic internal count
```

If TreeTune-style optimizer-update curves are needed, log a separate axis such
as:

```text
training/num_optimizer_steps_total
```

Do not rename or repurpose VERL's `global_step` to obtain that plot.

## 1.2 Rollout and replay

```text
Generate complete stochastic trees
    ↓
Validate full-tree construction invariants
    ↓
Convert every realized non-placeholder segment into one replay edge
    ↓
Stamp generation_rollout_iteration and unique identities
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
Tensorize
    ↓
Run one outer actor update
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

Complete-tree replay remains an explicitly labeled ablation only.

## 1.3 Canonical policy objective

For each internal PPO optimizer batch `B`:

\[
L_B^{(r)}
=
\frac{1}{N_B}
\sum_{s\in B}L_s^{(r)},
\qquad r\in\{\mathrm{mean},\mathrm{sum}\}.
\]

Normally `N_B=128`. Every selected replay slot has equal outer weight.

The following must not affect canonical policy weight:

```text
tree size
parent group size
allocated_k
queue_flush_id
branch factor
replay age
objective_weights
segment_objective_weights
```

Tree and queue counts remain construction/theory diagnostics.

---

# 2. Correction of the previous regression analysis

The previous plan listed R1-R6 as newly discovered P0 regressions. Their correct
status is:

## R1 — Training duration wrong unit

```text
CAUSE: previous instruction changed global_step from outer update count
       to internal optimizer-step count.
```

This is a regression caused by that instruction, not a defect in VERL's
`total_training_steps` derivation.

Correct fix:

```text
restore global_step += 1 per successful outer actor update
keep total_training_steps in outer-update units
keep num_optimizer_steps_total separate and observational
```

## R2 — Four optimizer updates but one scheduler step

This is **not a bug under the preserved VERL contract**.

```text
one update_actor call = one outer policy update = one scheduler step
```

Do not change scheduler cadence without user approval.

## R3 — Non-finite attempts counted in internal optimizer metric

The optimizer's finite-gradient safety behavior is valid. The ambiguity is in
the newly added metric name/interpretation.

For now:

```text
internal optimizer count is diagnostic only
it does not drive global_step or schedules
```

Renaming it to attempts versus successful updates, or changing `_optimizer_step`
to return `did_step`, requires a separate discussion because it can affect
metrics and tests but is not a training-loop blocker.

## R4 — Reserved-update failure handling

This contains one real production issue independent of the counter redesign:

```text
validation or tensorization failure after reservation must rollback rows
```

Pre-actor rollback must be fixed.

An actor-result metric mismatch after the model may already have changed is not
a normal replay transaction. Do not invent rollback/commit semantics for that
case without discussing it with the user.

## R5 — All-zero shortcut bypasses validation

This regression was introduced by an optional shortcut. Submission-first
resolution:

```text
disable/remove the canonical whole-reservation all-zero shortcut
retain dense actor behavior
```

No zero shortcut may run before validation.

## R6 — Zero-signal internal optimizer batch still steps

This is not a canonical correctness bug. Per-mini-batch zero skipping changes:

```text
AdamW weight decay behavior
entropy/KL behavior
optimizer-update counts
scheduler semantics
replay row-consumption policy
```

It is an optional redesign and must not be implemented without user approval.

---

# 3. Current status

## 3.1 Preserve these completed fixes

| Area | Status | Rule |
|---|---|---|
| Edge-level replay dispatch | DONE | Keep canonical `replay_sampling_unit=edge`. |
| Auto per-question cap | DONE | Keep `666→33`, `888→73`. |
| Hard target cap | DONE | Never forward more than 512 rows. |
| Mean/sum policy-loss wiring | DONE | Keep `actor.policy_loss.*` as source. |
| Canonical DataProto float weights | DONE | Main path carries neither objective-weight tensor. |
| Stored old log-probs | DONE | Actor must use generation-time denominator. |
| `rollout_iteration` checkpoint state | DONE | Keep for replay age. |
| DDP two-process parity evidence | USEFUL | Keep as evidence; do not infer FSDP proof. |

## 3.2 Remaining production work

| ID | Status | Work |
|---|---|---|
| E1 | REQUIRED | Disable/remove canonical all-zero shortcut. |
| E2 | REQUIRED | Validate replay rows before tensorization; missing advantage must fail. |
| E3 | REQUIRED | Correct stale comments/specs. |
| M1 | REQUIRED | Restore VERL `global_step += 1` and separate optimizer metric. |
| M2 | REQUIRED | Guarantee rollback for all pre-actor failures. |
| M3 | REQUIRED | Enforce strict tree/edge identities without legacy fallback in strict mode. |
| M4 | REQUIRED | Decouple canonical manifest from obsolete weight normalization. |
| M5 | REQUIRED | Validate real Hydra/dataclass path without silently sanitizing canonical fields. |
| H1 | VERIFY | Run actual FSDP/FSDP2-oriented smoke/parity evidence. Test first. |

The following are not approved implementation tasks:

```text
changing scheduler to step per internal optimizer batch
redefining global_step as optimizer-step count
changing total_training_steps to optimizer-step units
skipping individual zero-signal optimizer batches
adding world-size scaling based only on toy DDP inference
```

They belong in `CLAUDE_FIX_HARD.md` as discussion-gated proposals.

---

# 4. Work split by difficulty

## Easy

Source: `CLAUDE_FIX_EASY.md`

```text
E1 remove/disable all-zero shortcut from canonical main
E2 validate rows before tensorization and reject missing advantage
E3 remove stale comments and wrong step/weight claims
E4 add regression tests for the unchanged host contract
```

## Medium

Source: `CLAUDE_FIX_MEDIUM.md`

```text
M1 restore and verify three-counter separation
M2 pre-actor replay transaction / rollback
M3 strict tree_instance_id and derived edge_id
M4 manifest invariant cleanup
M5 real Hydra/dataclass validation
```

## Hard / discussion-gated

Source: `CLAUDE_FIX_HARD.md`

```text
H1 FSDP/FSDP2 verification: test first, report before changing production
H2 scheduler-per-internal-step redesign: not approved
H3 per-mini-batch zero skip: not approved
H4 global-step / total-step unit redesign: prohibited without approval
H5 distributed scaling modifications: test and discuss before patching
```

---

# 5. Required execution order

```text
Stage 1 — EASY
E1 → E2 → E3 → E4

Stage 2 — MEDIUM
M1 → M2 → M3 → M4 → M5

Stage 3 — INTEGRATION
full CPU gate
short GPU smoke using preserved VERL semantics
manifest and metric review

Stage 4 — HARD VERIFICATION
H1 only

Any other hard change requires a user discussion first.
```

---

# 6. Conflict-safe completion rule

A task is complete only when:

```text
production success path passes
production failure path passes
existing host-framework semantics remain unchanged unless approved
no counter, scheduler, checkpoint, replay, objective, or distributed contract
was silently redefined
```

For every completed task, report:

```text
files changed
behavior changed
behavior intentionally preserved
consumers audited
regression tests run
known unresolved risks
```

If a test can only be made to pass by changing an unrelated contract, stop and
discuss the conflict instead of broadening the patch.

---

# 7. Definition of done before GPU smoke

```text
[ ] global_step increases by one per successful outer actor update
[ ] total_training_steps remains in the same outer-update unit
[ ] scheduler remains one step per successful update_actor call
[ ] num_optimizer_steps_total is separate and observational
[ ] rollout_iteration is restored and used for replay age
[ ] canonical all-zero shortcut is disabled or removed
[ ] replay rows are validated before tensorization
[ ] validation/tensorization/actor-RPC failures rollback reservations
[ ] canonical manifest does not depend on objective-weight normalization
[ ] strict main requires canonical tree/edge identities
[ ] edge-level replay/caps remain correct
[ ] stored old log-probs remain in use
[ ] CPU gate passes
```

Then run a short GPU smoke and report separately:

```text
rollout_iteration
global_step
successful_actor_updates
num_optimizer_steps_total
scheduler step / LR observations
selected replay edges
replay ages
manifest verdict
```

Do not start a long paper experiment if those counters use inconsistent units.