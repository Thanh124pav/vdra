# PLAN.md — Remaining Work Before GPU Smoke

## Purpose

This document is the current source of truth for the VDRA implementation after auditing `main` following PR #7.

Do not treat every historical P0 item as unfinished. Several foundations are already implemented. The remaining work is to make the **production path** match the new configuration and loss semantics end to end.

The repository is not GPU-ready until every item marked **BLOCKED** below is closed by production-path tests.

---

# 1. Canonical behavior

```text
ONE ROLLOUT ITERATION

Generate stochastic trees
    ↓
Convert every realized non-placeholder segment into one replay edge
    ↓
Stamp generation_rollout_iteration
    ↓
Add edges transactionally to replay
    ↓
Expire edges by rollout age
    ↓
Sample individual edges, grouped/capped by question_id
    ↓
Select at most 512 edges total
    ↓
Train global optimizer batches of 128 edges
    ↓
One optimizer.step() per 128-edge batch
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

Counters:

```text
rollout_iteration
    One generation/replay-fill cycle.
    Used for replay age.

global_step
    One successful actual optimizer.step().
    Used for optimizer-step schedules, checkpoints, and logs.

optimizer_steps_this_iteration
    Actual optimizer steps completed during the current rollout iteration.
```

For 512 selected edges, mini-batch 128, one PPO epoch:

```text
rollout_iteration += 1
global_step += 4
optimizer_steps_this_iteration = 4
```

---

# 2. Canonical policy objective

For one optimizer batch `B`, let:

```text
N_B = number of selected replay slots in that optimizer batch
```

Normally `N_B=128`.

For segment `s`, after response/probability masking:

\[
Z_s = \sum_t M_{s,t}.
\]

Supported within-segment reductions:

\[
L_s^{\mathrm{mean}}
=
\begin{cases}
\frac{1}{Z_s}\sum_t M_{s,t}\ell_{s,t}, & Z_s>0,\\
0, & Z_s=0,
\end{cases}
\]

\[
L_s^{\mathrm{sum}}
=
\sum_t M_{s,t}\ell_{s,t}.
\]

Canonical replay-training objective:

\[
L_B^{(r)}
=
\frac{1}{N_B}
\sum_{s\in B}L_s^{(r)},
\qquad r\in\{\mathrm{mean},\mathrm{sum}\}.
\]

Every selected replay slot has equal outer weight.

The following must not affect main-path policy weight:

```text
tree size
parent group size
allocated_k
queue_flush_id
branch factor
replay age
```

The full-tree/queue identity remains useful for the paper derivation and rollout diagnostics, but it is **not** the replay optimizer weighting rule.

---

# 3. Current status

| Area | Status | Notes |
|---|---|---|
| `segment_token_reduction=mean|sum` schema | DONE | Declared and validated in `PolicyLossConfig`. |
| Production mean/sum resolver | MOSTLY DONE | Still fix other policy-loss fields read from the wrong config level. |
| Main batch-slot loss | DONE | Uses `original_optimizer_batch_slot_count`. |
| Default PPO mini-batch 128 | DONE | Main config updated. |
| Actor optimizer-step counting | DONE | Actor returns actual step count. |
| `global_step += actual optimizer steps` | DONE for uninterrupted runs | Resume and interval-trigger behavior remain broken. |
| Replay age field | DONE for uninterrupted runs | Uses `generation_rollout_iteration`. |
| Auto-cap formula | DONE | `666→33`, `888→73` helper exists. |
| Same-server scorer config | DONE at config level | GPU/runtime handshake still requires smoke verification. |
| Canonical edge-level replay | BLOCKED | Strict main still reserves complete trees. |
| Auto cap in canonical production | BLOCKED | Complete-tree reservation can ignore/overshoot cap. |
| Edge-level integrity validation | BLOCKED | Current validation still requires complete parent groups. |
| Main DataProto without float objective weights | BLOCKED | Old tree/parent weights are still tensorized and validated. |
| Exact replay target/remainder handling | BLOCKED | Complete-tree packing can return 516 and create a 4-row optimizer step. |
| Counter checkpoint/resume | BLOCKED | `rollout_iteration` is not restored. |
| Save/eval threshold crossing | BLOCKED | Step jumps can skip frequencies. |
| Optional zero-row sparsification | PARTIAL | Unit identity exists; production filter/skip path is incomplete. |
| Strict unique tree IDs | PARTIAL | Production still has permissive fallbacks; manifest uniqueness check is weak. |
| Distributed scaling | BLOCKED | Only algebraic simulation exists; production FSDP/DDP behavior is unverified. |
| Manifest observed facts | BLOCKED | Several fields still become true from unrelated checks or before actor use. |
| Pre-GPU integration gate | BLOCKED | Still skips production trainer contracts and relies on synthetic mirrors. |

---

# P0.A — Make canonical replay genuinely edge-level

## Current bug

The main config declares:

```text
replay_sampling_unit = edge
```

but `strict_group_integrity=true` causes the trainer to call:

```text
reserve_complete_trees_for_update()
```

This contradicts the canonical replay design.

For a `666` tree:

```text
full tree edges = 258
auto per-question cap = 33
```

Complete-tree reservation can still select all 258 edges for that question, so the automatic cap is effectively bypassed.

## Required production change

Canonical VDRA must always call the edge reservation path:

```python
reservation = replay_buffer.reserve_for_update(
    current_rollout_iteration=self.rollout_iteration,
)
```

Do not select reservation mode from `strict_group_integrity`.

If complete-tree replay is retained, expose it only as an explicit ablation:

```yaml
replay_sampling_unit: complete_tree
```

It must never be selected implicitly by strict mode.

## Edge sampler contract

1. Expire unreserved edges with age `>= max_edge_age_iterations`.
2. Group available edges by stable `question_id`.
3. Randomly choose at most the resolved per-question cap.
4. Combine candidates across questions.
5. Select at most `target_edges_per_iteration` total.
6. Never return more than the target.
7. Reserve exact selected edge IDs.
8. Commit only after successful actor execution.
9. Roll back unchanged after failure.
10. Never duplicate edges to fill the target.

## Acceptance tests

- Strict main config calls `reserve_for_update`, not complete-tree reservation.
- A `666` tree contributes at most 33 edges/question/iteration when `R=1`, age 8.
- A `888` tree contributes at most 73.
- Sample size is never greater than 512.
- Sampling can intentionally split trees and parent groups.
- Commit and rollback are transactional.

---

# P0.B — Separate tree-construction integrity from replay-batch integrity

## Current bug

The current batch validator requires:

```text
rows from parent group in sampled batch == allocated_k
```

That is valid when checking a freshly generated full tree, but invalid for edge-level replay. Edge replay is allowed to select only some siblings from a parent group.

## Required separation

### Tree construction validation

Run once immediately after a full tree is generated and before inserting its edges into replay:

```text
realized_child_count == allocated_k under fresh_iid
sample_multiplicity == 1 under fresh_iid
all edge IDs unique within generated tree batch
sum_q queue_released_segment_count[q] == tree_total_segment_count
no pruned placeholders counted as trainable segments
```

This validation sees the complete generated tree.

### Replay batch validation

Run on sampled edge rows before tensorization:

```text
required row metadata exists
edge_id values are unique
question_id exists
generation_rollout_iteration exists
stored old log-probs align with response tokens
actual training advantage exists
sampled edge count <= 512
sampled per-question count <= resolved cap
all sampled ages are valid
```

It must **not** require:

```text
complete tree
complete parent group
row_count == allocated_k
queue totals reconstructed from the partial replay sample
```

## Acceptance tests

- A replay batch containing 2 of 6 siblings passes replay validation.
- The original full generated tree still fails construction validation if it realized only 5 of allocated 6 children.
- Partial replay sampling does not increment `group_integrity_failures`.

---

# P0.C — Remove obsolete objective-weight tensors from main path

## Current bug

`edges_to_dataproto()` still computes and attaches:

```text
objective_weights
segment_objective_weights
```

The main loss no longer consumes these coefficients. It uses the equal replay-slot mean.

Manifest lifecycle also still validates tree-normalized and parent-normalized weights on canonical batches.

## Required changes

For `loss_mode=vdra_segment_mean_ppo`:

- Do not compute or attach `objective_weights`.
- Do not compute or attach `segment_objective_weights`.
- Do not validate parent/tree weight normalization.
- Keep integer/identity metadata only when needed for diagnostics.
- The actor must receive `original_optimizer_batch_slot_count` for the current 128-edge optimizer batch.

For the explicit `vdra_node_balanced_ppo` ablation, old tensors may be generated only on that path.

## Acceptance tests

- Canonical main `DataProto.batch` contains neither float objective-weight tensor.
- Node-balanced ablation still receives its required weights.
- Canonical loss/gradient is unchanged after deleting old weight tensors.

---

# P0.D — Enforce exact optimizer-batch cardinality

## Current bug

Complete-tree packing can return more than 512 rows. Example:

```text
258 + 258 = 516
```

Actor splitting may then produce:

```text
128 + 128 + 128 + 128 + 4
```

This creates an unintended 4-row optimizer step and conflicts with floor-based expected-step accounting.

## Canonical behavior

- The edge sampler must never return more than 512.
- If selected count is 512: perform 4 optimizer steps.
- If selected count is 384, 256, or 128: the underfilled update may run because it is divisible by 128.
- If selected count is not divisible by 128: roll back/postpone.
- No smaller tail optimizer batch in canonical mode.

Expected step count:

\[
N_{\mathrm{steps}}
=
\frac{N_{\mathrm{selected}}}{128}
\times
\mathrm{ppo\_epochs}.
\]

This formula is valid only after divisibility has been enforced.

## Acceptance tests

- 516 candidates produce exactly 512 selected rows.
- 512 produces four optimizer steps.
- 384 produces three.
- 516 is never forwarded to actor.
- 130 rows are postponed and reservation rolled back.

---

# P0.E — Finish counter persistence and interval triggers

## Current bug 1: resume

`global_step` is restored from checkpoint folder naming, but `rollout_iteration` and `num_optimizer_steps_total` are reset.

Restored replay edges can therefore have negative ages:

```text
restored rollout_iteration = 0
edge generation_rollout_iteration = 98
age = -98
```

## Required checkpoint state

Save a trainer state file in every checkpoint:

```json
{
  "global_step": 400,
  "rollout_iteration": 100,
  "num_optimizer_steps_total": 400
}
```

Restore it before replay expiration or generation begins.

For legacy checkpoints without this state:

- either reset replay explicitly, or
- set `rollout_iteration` to at least the maximum restored edge generation iteration;
- emit a visible migration warning.

Never silently continue with negative replay ages.

## Current bug 2: save/eval frequency

A trainer call may advance:

```text
global_step: 8 → 12
```

A check based only on:

```python
global_step % 10 == 0
```

misses the step-10 event.

## Required trigger semantics

Use crossed thresholds or `next_*_step` counters:

```python
while previous_global_step < next_save_step <= current_global_step:
    save_checkpoint()
    next_save_step += save_freq
```

Apply equivalent logic for evaluation, profiling, and any optimizer-step schedule owned by the trainer.

## Logging cleanup

Do not log contradictory keys such as:

```text
training/optimizer_step = 8
training/global_step = 12
```

in the same final event.

Use:

```text
training/global_step_before_update
training/global_step_after_update
training/global_step
training/rollout_iteration
training/optimizer_steps_this_iteration
```

## Acceptance tests

- Save/load restores all counters exactly.
- Restored replay ages are non-negative and expire at the correct iteration.
- A jump from 8 to 12 fires the threshold at 10.
- Logs contain one unambiguous final `training/global_step`.

---

# P0.F — Complete policy-loss config wiring

`segment_token_reduction` is wired correctly, but other fields declared inside `PolicyLossConfig` are still read from the actor top level.

Canonical reads must use:

```python
config.policy_loss.use_prob_mask
config.policy_loss.ratio_threshold
config.policy_loss.segment_token_reduction
```

Do not read these as:

```python
config.get("use_prob_mask", ...)
config.get("ratio_threshold", ...)
```

unless the function explicitly receives a `PolicyLossConfig` object instead of `ActorConfig`.

Acceptance:

- Overrides at `actor.policy_loss.*` change production behavior.
- Wrong-level duplicates are removed or rejected.

---

# P0.G — Finish optional zero-contribution sparsification

The default main run currently keeps zero-advantage rows. That dense path is acceptable.

If sparse execution remains supported:

- Filter using the exact scalar/tensor broadcast into policy `advantages`.
- Do not filter using `pav_advantage` or another diagnostic value.
- Preserve the original optimizer-slot denominator `N_B`.
- If all selected slots have zero contribution, do not call `optimizer.step()`.
- Do not increment `global_step`.
- Define and test whether those replay rows are committed or retained.

The production path, not only a standalone loss test, must exercise all-zero skip behavior.

---

# P0.H — Strengthen unique tree and edge identity

Strict main generation must require a globally unique `tree_instance_id`.

Do not silently derive strict IDs from only:

```text
policy snapshot + question ID
generic segment ID
```

Required tree identity should include or embed:

```text
policy snapshot
rollout iteration
stable question ID
per-tree UUID/counter
```

`edge_id` must derive from that unique tree identity plus child identity.

Manifest validation must verify more than “the tree ID set is non-empty.” It must detect collisions among multiple stochastic trees for the same question and snapshot.

---

# P0.I — Verify and fix actual distributed gradient scaling

The existing test demonstrates the algebraic world-size trap but does not prove production FSDP/DDP behavior.

Do not blindly multiply by world size.

First determine production dispatch semantics:

```text
Does every DP rank receive disjoint rows or a replicated optimizer batch?
Does the reducer sum or average gradients?
What denominator is passed on each rank?
```

Then implement scaling so the distributed gradient equals the single-rank 128-slot mean.

Required evidence:

- Real two-process `torch.distributed` CPU test, or a minimal FSDP/DDP integration test using the same production reducer.
- Both `mean` and `sum` token reductions.
- Uneven token lengths.
- Explicit comparison with a one-process 128-row reference.

A pure single-process algebra simulation is not sufficient for P0 completion.

---

# P0.J — Make manifest fields genuinely observed

Canonical manifest validation must not depend on complete-tree replay.

Remove complete-tree/complete-parent fields from main validity.

Required observed facts:

```text
replay_sampling_unit == edge
selected edge count <= target
per-question cap respected
replay ages based on rollout_iteration
optimizer-step count equals actual actor steps
stored old log-probs actually used by actor
no truncation occurred during tensorization
unique IDs verified
scorer/rollout versions verified
```

Specific fixes:

- Do not set `segment_count_invariants_passed` from a generic group-integrity pass.
- Do not set node-balanced and segment-mean invariant bits together.
- Do not mark `stored_old_log_probs_used=True` merely because the tensor exists; actor output must confirm it used the stored denominator.
- Set `no_truncation=True` only after strict tensorization succeeds.
- Do not run queue/tree normalization checks on a partial replay sample.
- Main manifest becomes valid only after at least one successful canonical optimizer step.

---

# P0.K — Replace synthetic-only gate with production integration evidence

`pre_gpu_check.sh` must continue to run unit tests, but it must also test the actual production wiring.

Required additions:

1. Real Hydra composition of the main config and `sum` override.
2. Instantiate complete actor config, not only `PolicyLossConfig`.
3. Execute actual canonical replay reservation with strict main settings.
4. Verify strict main uses edge reservation.
5. Pass sampled edges through actual `edges_to_dataproto()`.
6. Assert old objective-weight tensors are absent on main path.
7. Exercise actual `DataParallelPPOActor.update_policy()` control flow using a minimal mock model/optimizer, not a rewritten mirror.
8. Verify 512/128 causes four real `_optimizer_step()` calls.
9. Test checkpoint/resume counter state.
10. Run a two-process distributed gradient parity test.
11. Stop ignoring `test_trainer_contracts.py`, or split its CPU-safe production contracts into an included module.
12. Remove complete-tree replay tests from canonical acceptance; retain them only under an ablation label.

GitHub CI on Python 3.10 and 3.12 must run the same gate and report a real successful workflow run.

---

# 4. Implementation order

Do not work on all files at once. Close tasks in this order:

```text
1. P0.A canonical edge reservation
2. P0.B split construction vs replay integrity
3. P0.C remove main-path objective weights
4. P0.D exact sample size and divisibility
5. P0.E checkpoint/resume and threshold crossing
6. P0.F remaining config wiring
7. P0.G zero sparsification production path
8. P0.H strict IDs
9. P0.I real distributed scaling
10. P0.J manifest observed facts
11. P0.K production integration gate
```

After steps 1–4, re-audit replay and actor batch semantics before proceeding.

---

# 5. Definition of done

GPU Smoke D may start only when all are true:

```text
[ ] strict canonical main uses edge-level reservation
[ ] complete trees/parents are not required in sampled replay batches
[ ] auto per-question cap is enforced in production
[ ] selected edge count never exceeds 512
[ ] every actor update receives a count divisible by 128
[ ] no 4-row or other tail optimizer step occurs
[ ] canonical DataProto contains no float objective-weight tensors
[ ] one optimizer batch uses equal replay-slot averaging
[ ] global_step counts actual optimizer.step() calls
[ ] rollout_iteration controls replay age
[ ] all counters survive checkpoint/resume
[ ] save/eval thresholds are not skipped by multi-step jumps
[ ] zero-row sparse mode is either production-correct or explicitly disabled
[ ] strict tree IDs cannot fall back to ambiguous identities
[ ] real distributed gradient parity passes
[ ] manifest fields are observed from the correct runtime stage
[ ] real Hydra and production integration tests pass
[ ] GitHub CPU CI is green on Python 3.10 and 3.12
[ ] scripts/pre_gpu_check.sh prints PRE_GPU_CHECK=PASS
```

Then run at least five rollout iterations and report:

```text
rollout iterations completed
optimizer global steps completed
selected edges per iteration
optimizer steps per iteration
replay age histogram
resolved per-question cap
manifest validity
```

For five full 512-edge iterations with batch 128 and one epoch, the expected optimizer-step total is 20.
