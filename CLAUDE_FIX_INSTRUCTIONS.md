# Claude Fix Instructions — Finish the Production Path

> Follow this file literally. Do not redesign the method. Do not claim completion from config changes, helper tests, or synthetic mirrors alone. Every task must be verified on the production path.

`PLAN.md` is the full specification. This file is the execution guide for the current repository state after PR #7.

---

# 0. What is already done

Do not reimplement these unless a regression test fails:

```text
DONE

✓ PolicyLossConfig declares segment_token_reduction
✓ accepted values are mean and sum
✓ main loss supports token mean and token sum
✓ main loss supports optimizer-batch slot averaging
✓ ppo_mini_batch_size defaults to 128
✓ target_edges_per_iteration defaults to 512
✓ actor counts actual optimizer.step() calls
✓ trainer increases global_step by returned optimizer-step count
✓ replay edges can store generation_rollout_iteration
✓ replay age helper uses rollout iteration
✓ auto-cap helper exists
✓ [6,6,6], age 8 resolves to 33
✓ [8,8,8], age 8 resolves to 73
✓ default scorer topology is same-server mode
```

Do not write another synthetic test that only repeats these identities. Fix the remaining production mismatches below.

---

# 1. The actual remaining blocker

Current configuration says:

```yaml
replay_sampling_unit: edge
max_edges_per_question_per_iteration: auto
strict_group_integrity: true
```

But the trainer still does:

```python
if manifest_strict:
    reserve_complete_trees_for_update(...)
else:
    reserve_for_update(...)
```

Therefore the canonical strict main run still uses complete-tree replay.

This is wrong.

For a `666` tree:

```text
full tree = 258 edges
auto cap  = 33 edges/question/iteration
```

Complete-tree replay can select all 258 edges and ignore the intended cap.

It can also pack two trees:

```text
258 + 258 = 516
```

and then actor splitting can create:

```text
128 + 128 + 128 + 128 + 4
```

That unintended 4-row optimizer step is a new regression.

Fix this chain end to end.

---

# 2. Final canonical flow

```text
ROLLOUT ITERATION

Generate complete trees
    ↓
Validate full-tree construction invariants
    ↓
Convert realized non-placeholder segments to replay edges
    ↓
Add edges transactionally
    ↓
Expire by rollout_iteration age
    ↓
Sample individual edges by question cap
    ↓
Select at most 512 total
    ↓
Require selected_count % 128 == 0
    ↓
Split into optimizer batches of 128
    ↓
One optimizer.step() per batch
```

Counters:

```text
rollout_iteration = generation/replay-fill cycles
global_step       = successful optimizer.step() calls
```

For 512 edges:

```text
rollout_iteration += 1
global_step += 4
```

---

# 3. Phase 1 — Switch strict main to real edge-level replay

## Target files

```text
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/replay_buffer.py
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
```

## Required change

Canonical main must call:

```python
reservation = replay_buffer.reserve_for_update(
    current_rollout_iteration=self.rollout_iteration,
)
```

Do not choose complete-tree reservation because `strict_group_integrity` is true.

Strictness controls validation, not sampling unit.

If complete-tree replay remains supported, require an explicit ablation config:

```yaml
replay_sampling_unit: complete_tree
```

Then dispatch by `replay_sampling_unit`, not by manifest strictness.

## Edge sampler requirements

```text
- group available edges by question_id
- apply resolved per-question cap
- sample individual edges
- return at most target_edges_per_iteration
- never exceed 512
- never duplicate rows
- reserve exact IDs
- commit only after successful actor update
- rollback unchanged on failure
```

## Required production tests

Use the actual `RayGearTreeTrainer` replay-selection helper or a directly extracted production function.

Test:

```text
strict main + replay_sampling_unit=edge
→ reserve_for_update is called
→ reserve_complete_trees_for_update is not called
```

Also test:

```text
666 tree → at most 33 edges for one question
888 tree → at most 73 edges for one question
516 available candidates → exactly 512 selected
```

Do not mark Phase 1 complete if only `compute_max_edges_per_question()` passes.

---

# 4. Phase 2 — Split construction validation from replay validation

## Problem

Current `validate_group_integrity()` expects a sampled batch to contain all siblings:

```text
sampled parent row count == allocated_k
```

That conflicts with edge-level replay.

## Required architecture

Create two separate validators.

### A. Full generated-tree validator

Run immediately after tree-to-edge extraction, before replay insertion.

Validate:

```text
realized_child_count == allocated_k under fresh_iid
sample_multiplicity == 1 under fresh_iid
edge IDs unique
no pruned placeholder counted as trainable segment
sum queue segment counts == tree total segment count
old log-probs align with response tokens
```

This validator sees the complete generated tree.

### B. Sampled replay-batch validator

Run before `edges_to_dataproto()`.

Validate only row-local and replay constraints:

```text
required fields exist
edge IDs unique
question IDs exist
generation_rollout_iteration exists
age is in [0, max_edge_age_iterations)
per-question selected count <= resolved cap
selected count <= target
old log-probs align with response tokens
actual training advantage exists
```

Do not require:

```text
complete tree
complete parent group
sampled row count == allocated_k
queue totals reconstructed from sampled rows
```

## Regression test

A sampled batch containing only 2 of 6 children from one parent must pass replay validation.

The original full tree with only 5 realized children when `allocated_k=6` must still fail construction validation.

---

# 5. Phase 3 — Remove old objective weights from canonical DataProto

## Problem

`edges_to_dataproto()` still always attaches:

```text
objective_weights
segment_objective_weights
```

The canonical main loss no longer uses them.

They preserve old tree/parent assumptions and keep manifest logic coupled to complete trees.

## Required change

Make tensorization aware of policy loss mode or pass an explicit flag.

For:

```text
loss_mode = vdra_segment_mean_ppo
```

do not compute or attach:

```text
objective_weights
segment_objective_weights
```

For:

```text
loss_mode = vdra_node_balanced_ppo
```

the ablation may still attach its required tensors.

Remove canonical manifest checks for tree-normalized and parent-normalized float weights.

Keep only metadata needed for diagnostics and strict construction validation.

## Test

```python
batch = edges_to_dataproto(..., loss_mode="vdra_segment_mean_ppo")
assert "objective_weights" not in batch.batch
assert "segment_objective_weights" not in batch.batch
```

Also verify the node-balanced ablation still works.

---

# 6. Phase 4 — Enforce exact sample and optimizer-batch sizes

## Canonical rule

```text
selected_count <= 512
selected_count % 128 == 0 before actor call
```

Allowed selected counts:

```text
128
256
384
512
```

Canonical path must postpone and roll back for counts such as:

```text
4
130
258
516
```

`516` must first be clipped/sampled to at most 512, never forwarded as 516.

## Required implementation

After reservation:

```python
n = len(sampled_edges)

if n > target_edges_per_iteration:
    raise AssertionError("sampler exceeded target")

if n % ppo_mini_batch_size != 0:
    replay_buffer.rollback(reservation)
    postpone()
```

Do not rely on this old condition:

```python
len(sampled_edges) < target and len(sampled_edges) % ppo_mini != 0
```

because it allows oversized non-divisible batches.

Expected optimizer steps:

```python
expected_steps = n // 128 * ppo_epochs
```

Only compute this after divisibility validation.

## Tests

```text
512 → 4 steps
384 → 3 steps
130 → postpone
516 candidates → select 512, then 4 steps
no final 4-row optimizer batch
```

---

# 7. Phase 5 — Fix checkpoint/resume counter state

## Current regression

The trainer restores `global_step` but resets `rollout_iteration` to zero.

Restored replay rows may then have negative age.

## Required state file

Save inside each checkpoint directory:

```text
gear_tree_trainer_state.json
```

Contents:

```json
{
  "global_step": 400,
  "rollout_iteration": 100,
  "num_optimizer_steps_total": 400
}
```

Restore this state before replay is used.

## Legacy checkpoint behavior

For a checkpoint without the new state file:

Choose one explicit safe behavior:

```text
A. reset replay and start rollout_iteration from 0
```

or:

```text
B. restore replay and set rollout_iteration >= max edge generation iteration
```

Emit a warning. Never silently create negative ages.

## Tests

```text
save at global_step=400, rollout_iteration=100
load
assert exact counter equality
assert restored replay edge ages are non-negative
assert expiry still happens after 8 rollout iterations
```

---

# 8. Phase 6 — Fix logging and save/eval threshold crossing

## Logging bug

The current event may contain a pre-update value under one key and a post-update value under another.

Use:

```text
training/global_step_before_update
training/global_step_after_update
training/global_step
training/rollout_iteration
training/optimizer_steps_this_iteration
```

The final `training/global_step` must equal the post-update value.

Remove or rename ambiguous `training/optimizer_step`.

## Frequency bug

A call may jump:

```text
8 → 12
```

A modulo check misses save/eval step 10.

Use crossed thresholds:

```python
previous = global_step_before_update
current = global_step_after_update

while previous < next_save_step <= current:
    save_checkpoint()
    next_save_step += save_freq
```

Apply the same principle to evaluation and profiling triggers.

## Tests

```text
8 → 12 with save_freq=10 triggers save
8 → 12 with test_freq=10 triggers evaluation
final log contains one consistent global_step=12
```

---

# 9. Phase 7 — Finish policy-loss config reads

`segment_token_reduction` is fixed.

Now fix fields that still live in `PolicyLossConfig` but are read from `ActorConfig` top level.

Read:

```python
config.policy_loss.use_prob_mask
config.policy_loss.ratio_threshold
```

Do not read:

```python
config.get("use_prob_mask", ...)
config.get("ratio_threshold", ...)
```

unless the function was passed the policy-loss subconfig directly.

Add production-path tests proving an override under:

```yaml
actor_rollout_ref.actor.policy_loss
```

changes behavior.

---

# 10. Phase 8 — Finish zero-advantage behavior

Default main path keeps zero-advantage rows. Keep this safe dense behavior unless sparse mode is fully implemented.

If sparse mode remains:

```text
filter using exact training advantage
not pav_advantage
preserve original N_B
skip optimizer.step() for all-zero batch
do not increment global_step
explicitly define replay commit/retain behavior
```

Current standalone loss test is not enough.

Add a production actor/trainer test proving the all-zero optimizer batch does not call `_optimizer_step()`.

If this cannot be completed quickly, disable sparse mode in canonical config and label it unfinished.

---

# 11. Phase 9 — Make strict IDs actually strict

Strict generation must require `tree_instance_id` containing a per-tree UUID/counter.

Do not silently fall back to:

```text
snapshot + question
generic segment ID
```

Required identity components:

```text
policy snapshot
rollout iteration
question ID
unique per-tree UUID/counter
```

Derive edge ID from the unique tree ID plus child identity.

Manifest verification must detect collisions among multiple stochastic trees for the same question and snapshot.

This is stronger than:

```python
bool(set(tree_ids))
```

Add a collision regression test on the production normalizer.

---

# 12. Phase 10 — Verify real distributed scaling

The current test only proves an algebraic identity in one process.

Before changing scale, inspect production dispatch:

```text
Are rows sharded or replicated across DP ranks?
Does FSDP/DDP sum or average gradients?
What denominator does each rank receive?
```

Then make the distributed gradient equal the single-rank 128-row reference.

Required test:

```text
real two-process torch.distributed CPU test
or minimal production FSDP/DDP integration
```

Test both:

```text
segment_token_reduction=mean
segment_token_reduction=sum
```

Do not claim this task complete from `test_distributed_grad_scaling.py` if it only manually sums/averages local losses inside one process.

---

# 13. Phase 11 — Repair manifest semantics

Canonical manifest must not require or infer complete trees in replay.

Remove complete-tree and complete-parent fields from canonical validity.

Do not do this:

```text
group integrity passed
→ set node_balanced_invariants_passed=True
→ set segment_count_invariants_passed=True
```

Those are different claims.

Required runtime observations:

```text
edge-level sampler used
selected_count <= target
per-question cap respected
all ages valid and based on rollout_iteration
actual optimizer steps returned by actor
stored old log-probs actually used by actor
strict tensorization completed without truncation
unique IDs verified
scorer/rollout version verified
```

Actor should return an observed metric such as:

```text
actor/used_stored_old_log_probs = 1
```

Only then set manifest `stored_old_log_probs_used=True`.

Set `no_truncation=True` after strict tensorization succeeds.

Do not reconstruct full-tree queue identities from a partial replay batch.

Main manifest becomes valid only after at least one successful canonical optimizer step.

---

# 14. Phase 12 — Upgrade pre-GPU gate

The gate must run actual production wiring, not only helper tests.

Required gate additions:

```text
✓ real Hydra composition of main config
✓ real Hydra composition of sum override
✓ instantiate complete ActorConfig
✓ strict main dispatches to edge reservation
✓ production replay sampler enforces cap and target
✓ actual edges_to_dataproto canonical path has no objective weights
✓ actual DataParallelPPOActor.update_policy control flow is exercised with a minimal mock model
✓ 512/128 performs four real _optimizer_step calls
✓ all-zero production batch skips optimizer step
✓ checkpoint/resume restores rollout_iteration
✓ threshold crossing triggers save/eval
✓ real two-process distributed gradient parity
✓ CPU-safe trainer contracts are no longer ignored
```

Remove complete-tree replay from canonical acceptance. Keep its test only under an explicit ablation section.

Do not call raw `yaml.safe_load()` a Hydra composition test.

GitHub Actions must produce a real green workflow run on both Python 3.10 and 3.12.

---

# 15. Required implementation order

Follow exactly:

```text
1. edge-level reservation in strict main
2. split construction/replay validators
3. remove main-path objective weights
4. exact target and divisibility handling
5. checkpoint/resume counters
6. logging and threshold crossing
7. remaining PolicyLossConfig reads
8. zero-adv production behavior
9. strict IDs
10. real distributed scaling
11. manifest observed facts
12. production integration gate
```

After steps 1–4, run a focused audit before continuing.

---

# 16. Do not do these things

```text
DO NOT restore parent-balanced main objective
DO NOT use queue ratios as actor policy weights
DO NOT keep complete-tree replay because strict mode is enabled
DO NOT claim auto cap works while complete-tree reservation bypasses it
DO NOT forward more than 512 edges
DO NOT permit a 4-row tail optimizer step in canonical mode
DO NOT use global_step for replay age
DO NOT reset rollout_iteration on resume
DO NOT infer stored old-log-prob use from tensor presence alone
DO NOT claim distributed correctness from a one-process algebra test
DO NOT claim Hydra composition from yaml.safe_load
DO NOT mark completion while production trainer tests are skipped
```

---

# 17. Completion report format

When finished, report:

```text
A. Changed production files
B. Changed test files
C. Exact canonical replay function now called
D. Sample counts tested: 128, 256, 384, 512, 516 candidates, 130 underfill
E. Exact optimizer-step counts observed
F. Checkpoint/resume counter values before and after
G. Save/eval threshold-crossing evidence
H. Real distributed test command and result
I. Hydra composition commands and result
J. Full pre_gpu_check output
K. GitHub Actions run status for Python 3.10 and 3.12
L. Any intentionally disabled unfinished feature
```

Do not report `P0 complete` unless all production-path requirements above are satisfied.
