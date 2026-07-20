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
CODEX_FIX_EASY.md
CODEX_FIX_MEDIUM.md
CODEX_FIX_HARD.md
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

Codex must not silently choose a new architecture in these areas.

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

## 1.2 Rollout and replay (2026-07-21 sparse-execution contract)

```text
Generate complete stochastic trees
    ↓
Validate full-tree construction invariants
    ↓
Compute rewards/values for ALL realized children; compute the parent
baseline from the COMPLETE sibling set; compute advantages for all
children (never after filtering)
    ↓
Convert every realized non-placeholder segment into ONE LOGICAL SLOT:
  - advantage != 0  → slot + full trainable payload
  - advantage == 0  → metadata-only slot (slot identity, tree_id,
    parent_group_id, question_id, response_token_count,
    advantage_is_zero=true, trainable_edge_id=null)
    ↓
Stamp generation_rollout_iteration and unique identities (slots)
    ↓
Insert slots transactionally (ledger = all realized slots;
payload store = nonzero slots only)
    ↓
Expire by rollout_iteration age (slots)
    ↓
Reserve individual LOGICAL SLOTS, capped by question_id
    ↓
Select at most 512 total logical slots
    ↓
Partition the 512 reserved slots into logical optimizer batches of
ppo_mini_batch_size BEFORE tensor filtering; compute the exact
pre-filter denominators M_B (slot count) and T_B (sum of
response_token_count over ALL slots in B, zero slots included)
    ↓
Validate sampled replay rows
    ↓
Tensorize ONLY the nonzero trainable payloads; stamp the per-batch
denominators into DataProto.meta_info; order rows rank-major per
logical batch and pad each batch's rows to a multiple of the DP size
with collective-safe dummy rows (zero gradient, counted nowhere)
    ↓
Run one outer actor update
  (if EVERY logical batch of the update has only zero-advantage slots,
   the trainer records an explicit skipped update instead: no
   update_actor RPC, no global_step, no scheduler.step)
```

Canonical defaults:

```yaml
replay_buffer:
  target_edges_per_iteration: 512       # counts LOGICAL SLOTS (zero + nonzero)
  max_edge_age_iterations: 8
  max_edges_per_question_per_iteration: auto   # counts logical slots
  replay_sampling_unit: edge            # slot-level; parent groups MAY split
  underfilled_update_policy: postpone_until_divisible  # in logical slots

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 128            # logical slots per optimizer batch
    ppo_epochs: 1
    policy_loss:
      loss_mode: vdra_segment_mean_ppo
      policy_aggregation: segment_mean
      segment_token_reduction: mean

tree_policy:
  policy_aggregation: segment_mean
  segment_token_reduction: mean
  only_adv_greater_than_zero: true      # sparse TENSOR EXECUTION only —
                                        # never removes slots from the
                                        # objective denominator
```

Zero-advantage sparsity is an EXECUTION policy, not an objective change:
zero slots count toward `target_edges_per_iteration`, per-question caps,
optimizer-batch divisibility and the `M_B`/`T_B` denominators; they do not
count toward tensor rows, model forward compute, or nonzero-gradient rows.

Complete-tree replay remains an explicitly labeled ablation only.

## 1.3 Canonical policy objective (USER DECISION 2026-07-20/21 — paper objective)

The canonical objectives follow the paper's Segment-Level Policy
Optimization section. `tree_policy.policy_aggregation` (duplicated and
must-agree with `actor.policy_loss.policy_aggregation`) selects one of:

```text
token_mean
    Every original valid token has equal weight.
segment_mean            (canonical DEFAULT)
    Every original segment/response slot has equal weight.
tree_balanced_segment_mean
    Labeled ABLATION only — every tree has equal total weight; segments
    averaged within tree (the historical w = 1/(N_T·N_seg) objective).
legacy_token_mean / vdra_node_balanced
    Unchanged baseline / ablation overlays.
```

For each LOGICAL optimizer batch `B` (formed from reserved slots BEFORE
zero filtering, §1.2):

```text
M_B = number of logical segment slots in B (zero slots included)
T_B = sum of response_token_count over every logical slot in B

segment_mean:  L_B = sum_{u in B, A_u != 0} [token-mean of segment u] / M_B
token_mean:    L_B = sum_{u in B, A_u != 0} sum_t token_loss(u,t) / T_B
```

Denominator rules (all violations must fail fast, never fall back):

```text
- M_B and T_B are stamped into DataProto.meta_info by the trainer
  (original_logical_segment_count / original_logical_token_count),
  ONE VALUE PER LOGICAL BATCH — never one value for a whole
  update_actor call, never reconstructed from retained rows/tokens,
  never approximated by tree_total_segment_count or parent-group
  proportional attribution.
- Every DP rank and every micro-batch of B reuses the same fixed value.
- Zero slots belong to the logical batch they were RESERVED into —
  membership is decided before filtering, not attributed afterwards.
- Distributed scaling: the reducer AVERAGES gradients across DP ranks
  (measured, docs/h1_fsdp_parity_report.md), so each rank computes
  local_loss = dp_size * local_numerator / denominator with the ACTUAL
  data-parallel group size.
- example: advantages [pos, neg, 0, 0] → tensor rows [L1, L2],
  M_B = 4, L_B = (L1 + L2) / 4.
```

All-zero logical batches (valid slots, no trainable payload):

```text
skip model forward/backward, optimizer.step, and that batch's
num_optimizer_steps_total contribution; if EVERY batch of the update is
all-zero, the trainer records an explicit skipped update (no
update_actor RPC, no global_step increment, no scheduler.step).
Record vdra/all_zero_logical_batches and
vdra/skipped_zero_gradient_updates. Never run an AdamW step on a
mathematically zero update (decoupled weight decay would still move
parameters).
```

`segment_mean` must never be described as Dr. GRPO (a fixed
maximum-length denominator would be a separate `dr_grpo_fixed_length`
option). The old name `global_segment_mean` refers to the tree-balanced
ablation ONLY: strict mode rejects it with a rename error; non-strict
legacy loading maps it to `tree_balanced_segment_mean` with a deprecation
warning; it NEVER maps to the new uniform `segment_mean`.

The following must not affect canonical policy weight:

```text
parent group size
allocated_k
queue_flush_id
branch factor
replay age
objective_weights
segment_objective_weights
unique-tree count N_T            (tree-balanced ablation only)
tree_total_segment_count N_seg   (tree-balanced ablation only)
```

Tree and queue counts remain construction/theory diagnostics and must not
be reconnected to the manifest validity gate.

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

> SUPERSEDED 2026-07-21 (user decision, §1.2/§1.3): the canonical path is
> now SPARSE TENSOR EXECUTION over a pre-filter logical-slot ledger. Zero
> slots are removed from model execution only — never from reservation,
> caps, divisibility, or the `M_B`/`T_B` denominators — and validation
> still runs on the reserved slots before tensorization. The old
> whole-reservation shortcut (which skipped validation and changed the
> objective) remains forbidden.

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

> USER APPROVAL GRANTED 2026-07-21 for exactly ONE narrow form (§1.3):
> a logical batch whose slots ALL have exactly zero advantage skips
> forward/backward/optimizer.step (no AdamW decay on a mathematically
> zero update); a fully-zero update becomes an explicit skipped update
> (no global_step, no scheduler.step). Any broader zero-signal skipping
> (thresholds, per-micro-batch, partial batches) remains unapproved.

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
skipping zero-signal optimizer batches BEYOND the approved all-zero
  logical-batch skip of §1.3 (2026-07-21)
adding world-size scaling beyond the approved dp_size factor of §1.3,
  which is backed by the measured FSDP2 reducer evidence
  (docs/h1_fsdp_parity_report.md), not by toy DDP inference alone
```

They belong in `CODEX_FIX_HARD.md` as discussion-gated proposals.

## 3.3 Hard-stage decisions locked on 2026-07-20/21 (user-approved)

```text
H1 evidence complete: docs/h1_fsdp_parity_report.md (FSDP2 reducer =
  average; segment_mean parity; tree_balanced non-parity → ablation).
Canonical objective: paper token_mean / segment_mean (§1.3);
  tree_balanced_segment_mean demoted to labeled ablation;
  batch_slot_mean_ablation flag removed (its math IS segment_mean).
Sparse-execution contract (§1.2): logical-slot ledger, pre-filter
  M_B/T_B stamped per logical batch, dummy-row collective safety,
  all-zero skip policy, strict validation with fail-fast (no retained-row
  fallback, no proportional attribution, no tree-count approximation).
```

---

# 4. Work split by difficulty

## Easy

Source: `CODEX_FIX_EASY.md`

```text
E1 remove/disable all-zero shortcut from canonical main
E2 validate rows before tensorization and reject missing advantage
E3 remove stale comments and wrong step/weight claims
E4 add regression tests for the unchanged host contract
```

## Medium

Source: `CODEX_FIX_MEDIUM.md`

```text
M1 restore and verify three-counter separation
M2 pre-actor replay transaction / rollback
M3 strict tree_instance_id and derived edge_id
M4 manifest invariant cleanup
M5 real Hydra/dataclass validation
```

## Hard / discussion-gated

Source: `CODEX_FIX_HARD.md`

```text
H1 FSDP/FSDP2 verification: DONE 2026-07-20 (docs/h1_fsdp_parity_report.md)
H2 scheduler-per-internal-step redesign: not approved
H3 per-mini-batch zero skip: not approved, EXCEPT the all-zero
   logical-batch skip approved 2026-07-21 (§1.3)
H4 global-step / total-step unit redesign: prohibited without approval
H5 distributed scaling: dp_size factor of §1.3 approved on measured
   FSDP2 evidence; anything further requires new discussion
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
[x] global_step increases by one per successful outer actor update
[x] total_training_steps remains in the same outer-update unit
[x] scheduler remains one step per successful update_actor call
[x] num_optimizer_steps_total is separate and observational
[x] rollout_iteration is restored and used for replay age
[x] canonical all-zero shortcut is disabled or removed
[x] replay rows are validated before tensorization
[x] validation/tensorization/actor-RPC failures rollback reservations
[x] canonical manifest does not depend on objective-weight normalization
[x] strict main requires canonical tree/edge identities
[x] edge-level replay/caps remain correct
[x] stored old log-probs remain in use
[x] CPU gate passes
```

Hard-stage additions (§1.2/§1.3, 2026-07-21), all verified on CPU:

```text
[x] canonical objective = paper segment_mean (default) / token_mean over
    pre-filter logical M_B / T_B; tree_balanced_segment_mean is ablation
[x] sparse tensor execution over the logical-slot ledger; advantages from
    the complete sibling set; zero slots count in reservation/caps/M_B/T_B
[x] dp_actor loss_scale_factor = dp_size (measured average-reducer
    compensation); real 2-rank FSDP2 segment_mean parity + no-hang
[x] all-zero logical batch = explicit skipped update (no global_step, no
    scheduler.step, no AdamW drift); strict fail-fast on missing denominator
[x] batch_slot_mean_ablation flag retired; global_segment_mean rename gated
[x] pre-server sweep clean (docs/pre_server_sweep_report.md)
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