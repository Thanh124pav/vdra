# Claude Implementation Guide — Finish VDRA Before GPU Smoke

> This file is an implementation brief. Follow it literally. Do not redesign the method, restore parent-balanced weighting, or reinterpret the replay semantics.
>
> `PLAN.md` contains the full specification. This file gives the shortest practical path to implement it correctly.

---

# 0. Final behavior in one picture

```text
ONE ROLLOUT ITERATION

Generate new trees
    │
    ├─ stamp rollout_iteration on every new edge
    ├─ add edges to replay
    └─ expire edges older than 8 rollout iterations

Replay sampling
    │
    ├─ sampling unit: edge / segment
    ├─ group by question_id
    ├─ per-question cap: computed automatically from tree_shape and age window
    └─ select up to 512 edges for this rollout iteration

Policy optimization
    │
    ├─ split selected edges into global optimizer batches of 128
    ├─ each 128-edge batch may be split into GPU microbatches
    ├─ accumulate gradients across those microbatches
    └─ call optimizer.step() exactly once per 128-edge batch

Counters for a full 512-edge iteration
    │
    ├─ rollout_iteration += 1
    ├─ optimizer_steps_this_iteration = 4
    └─ global_step += 4
```

Canonical defaults:

```yaml
target_edges_per_iteration: 512
ppo_mini_batch_size: 128
ppo_epochs: 1
max_edge_age_iterations: 8
max_edges_per_question_per_iteration: auto
replay_sampling_unit: edge
policy_aggregation: global_segment_mean
segment_token_reduction: mean
```

---

# 1. Non-negotiable semantics

## 1.1 Counters

Use three separate concepts:

```text
rollout_iteration
    Number of generation/replay-fill cycles.
    Used for replay age.

global_step
    Number of successful actual optimizer.step() calls.
    Used for LR scheduler, optimizer-step checkpoints, and optimizer-step logs.

optimizer_steps_this_iteration
    Number of successful optimizer.step() calls during the current rollout iteration.
```

Do not increment `global_step` once per call to `update_actor()` if that call contains several optimizer steps.

Do not use `global_step` to expire replay rows.

## 1.2 Replay

Replay sampling unit is one edge, which is one training segment.

Do not require complete-tree replay in the canonical path.

The old TreeTune diversity mechanism is intentional:

```text
large tree
→ its edges remain in replay for several rollout iterations
→ each question contributes at most an automatically resolved cap per iteration
```

## 1.3 Policy objective

The generated-tree derivation remains:

\[
\widehat g_T
=
\frac{1}{N_{\mathrm{seg}}(T)}
\sum_{s\in\mathcal S(T)}A_sH_s
=
\sum_q\frac{n_q}{N_{\mathrm{seg}}(T)}
\left(\frac1{n_q}\sum_{s\in\mathcal S_q}A_sH_s\right).
\]

Queue regrouping is analysis only. It must not produce queue weights in the actor loss.

During replay training, every selected segment slot in one optimizer batch has equal outer weight.

For one optimizer batch `B` of 128 selected slots:

\[
L_B^{(r)}
=
\frac{1}{N_B}
\sum_{s\in\operatorname{retained}(B)}L_s^{(r)},
\qquad N_B=128
\]

for a full batch.

Supported within-segment reductions:

```text
mean: average active token losses inside the segment
sum:  sum active token losses inside the segment
```

`mean` is default. `sum` is a supported ablation.

Do not use parent-balanced `objective_weights` in the canonical path.

---

# 2. Implement in this order

Follow the phases in order. Do not start GPU work before Phase 8 passes.

---

# Phase 1 — Fix configuration wiring

## Problem

`segment_token_reduction` was added to YAML but is not reliably declared/read through the real actor config schema. The `sum` option may be rejected or silently fall back to `mean`.

## Required changes

### A. Declare the field in the real schema

Target:

```text
verl/verl/workers/config/actor.py
```

Add to `PolicyLossConfig`:

```python
segment_token_reduction: str = "mean"
```

Validate it is exactly one of:

```python
{"mean", "sum"}
```

### B. Use one actor-side source of truth

Canonical source:

```text
actor_rollout_ref.actor.policy_loss.segment_token_reduction
```

The loss must read:

```python
config.policy_loss.segment_token_reduction
```

not:

```python
config.get("segment_token_reduction", "mean")
```

at the wrong config level.

### C. Validate duplicate config fields

If this remains:

```text
tree_policy.segment_token_reduction
```

then startup must assert:

```text
tree_policy.segment_token_reduction
== actor.policy_loss.segment_token_reduction
```

### D. Canonical config

Target:

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
```

Set:

```yaml
tree_policy:
  policy_aggregation: global_segment_mean
  segment_token_reduction: mean

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 128
    ppo_epochs: 1
    policy_loss:
      loss_mode: vdra_segment_mean_ppo
      segment_token_reduction: mean
```

## Tests

- Hydra-compose the real main config.
- Instantiate the real `ActorConfig` and `PolicyLossConfig`.
- Repeat with `segment_token_reduction=sum`.
- Verify production loss code receives `sum`.
- Invalid value such as `average` must fail before training.

---

# Phase 2 — Replace replay configuration semantics

## Problem

Current names imply that 512 edges are one optimizer update. They are actually the target edge count consumed in one rollout iteration.

Complete-tree replay also conflicts with the intended TreeTune-style diversity behavior.

## Required config

Replace or deprecate:

```text
target_edges_per_update
max_edge_age
max_edges_per_question
```

with:

```text
target_edges_per_iteration
max_edge_age_iterations
max_edges_per_question_per_iteration
replay_sampling_unit
```

Canonical values:

```yaml
replay_buffer:
  target_edges_per_iteration: 512
  max_edge_age_iterations: 8
  max_edges_per_question_per_iteration: auto
  replay_sampling_unit: edge
```

Backward-compatible aliases may be accepted temporarily, but canonical configs, manifests, logs, and tests must use the new names.

## Replay age

Each new edge must store:

```text
generation_rollout_iteration
```

Expire when:

```python
current_rollout_iteration - generation_rollout_iteration >= max_edge_age_iterations
```

Never use optimizer `global_step` for this calculation.

## Sampling behavior

Canonical sampler:

```text
1. remove expired unreserved edges
2. group remaining edges by stable question_id
3. select at most resolved_cap edges per question
4. combine candidates
5. sample up to target_edges_per_iteration
6. reserve selected edge IDs
7. commit only after successful actor execution
8. rollback unchanged on actor failure
```

Sampling unit is one edge. Do not pack complete trees.

Do not duplicate rows to fill 512.

---

# Phase 3 — Compute the per-question cap automatically

## Formula

For:

```text
tree_shape = [b1, b2, ..., bD]
```

compute maximum non-root edges per full tree:

\[
E_{\max}
=
\sum_{d=1}^{D}\prod_{\ell=1}^{d}b_\ell.
\]

If one question may create `R` stochastic trees per rollout iteration:

\[
E_{\max}^{\text{question/iteration}}=R E_{\max}.
\]

Resolve:

\[
C_{\text{question}}
=
\left\lceil
\frac{E_{\max}^{\text{question/iteration}}}
{\text{max edge age iterations}}
\right\rceil.
\]

Examples for `R=1`, age 8:

```text
[6,6,6]
E_max = 6 + 36 + 216 = 258
cap   = ceil(258 / 8) = 33

[8,8,8]
E_max = 8 + 64 + 512 = 584
cap   = ceil(584 / 8) = 73
```

## Implementation requirements

- Resolve `auto` during startup after the final tree config is known.
- Do not retain a hidden hard-coded fallback of 32 in the main path.
- Numeric override may exist for compatibility or ablation only.
- Log both configured and resolved values.

Required logs:

```text
replay/tree_max_edges
replay/trees_per_question_per_iteration
replay/max_edge_age_iterations
replay/resolved_max_edges_per_question_per_iteration
```

## Tests

- `666 → 33`.
- `888 → 73`.
- Changing tree shape must change the resolved cap.
- Sampling never exceeds the resolved cap for one question.

---

# Phase 4 — Fix optimizer-step control flow

## Required behavior

Given:

```text
selected edges = 512
ppo_mini_batch_size = 128
ppo_epochs = 1
```

production must perform:

```text
4 actual optimizer.step() calls
```

Each 128-edge optimizer batch may be split into microbatches for memory, but those microbatches belong to one optimizer step.

Correct structure:

```python
optimizer_steps_this_iteration = 0

for epoch in range(ppo_epochs):
    for optimizer_batch in split(selected_edges, global_batch_size=128):
        optimizer.zero_grad()

        for microbatch in split_for_memory(optimizer_batch):
            partial_loss = compute_partial_loss(
                microbatch,
                original_optimizer_batch_size=len(optimizer_batch),
            )
            partial_loss.backward()

        optimizer.step()
        lr_scheduler.step()

        global_step += 1
        optimizer_steps_this_iteration += 1
```

Do not normalize each 128-edge optimizer step using a denominator of 512.

Do not call `optimizer.step()` after every GPU microbatch.

## Underfilled behavior

Canonical behavior:

```text
postpone until selected edge count is divisible by 128
```

Alternative smaller final optimizer batch is allowed only if explicitly configured, tested, and logged.

Do not silently treat a 17-edge tail as if it had denominator 128.

## Counter propagation

The actor must return the number of successful optimizer steps to the trainer.

Example actor metadata:

```python
actor_output.meta_info["num_optimizer_steps"] = 4
```

The trainer must increment its persisted `global_step` by the returned value, not by one.

Better: make the actor/trainer share one authoritative counter update contract so double-increment is impossible.

---

# Phase 5 — Fix loss normalization

## One optimizer batch

For each 128-edge optimizer batch:

```text
N_B = selected segment slots before optional zero-row sparsification
```

For each retained row:

```python
if reduction == "mean":
    row_loss = masked_token_loss.sum() / active_token_count
elif reduction == "sum":
    row_loss = masked_token_loss.sum()
```

Then:

```python
batch_loss = sum(row_losses) / N_B
```

When split into microbatches, each microbatch contributes:

```python
partial_loss = sum(row_losses_in_microbatch) / N_B
```

and all partial gradients are accumulated before one optimizer step.

## Remove float objective tensors from the main path

Do not create or attach these for canonical VDRA:

```text
objective_weights
segment_objective_weights
```

Keep legacy tensors only inside explicit legacy-ablation code paths when those modes are selected.

Use integer counts and compute denominators in the actor.

Suggested dtype handling:

```python
row_loss_fp32 = row_loss.float()
loss_fp32 = row_loss_fp32.sum() / float(original_batch_slot_count)
```

Model forward/backward may remain BF16/FP16.

## Tests

For both `mean` and `sum`:

- 128 rows processed directly.
- The same 128 rows processed as 2, 4, and 8 microbatches.
- Same loss and parameter gradients.
- Same result after row permutation.
- Tests must execute `optimizer.step()` in the production location.

Mode distinction:

```text
mean: duplicate identical tokens inside one segment → row loss unchanged
sum:  duplicate identical tokens inside one segment → row loss doubles
```

---

# Phase 6 — Fix zero-advantage handling

## Canonical safe path

It is acceptable to keep all rows initially:

```yaml
only_adv_greater_than_zero: false
```

Get the dense path correct first.

## Optional sparse path

If zero-contribution rows are removed before model execution:

- Decide using the exact advantage used by the policy loss.
- Do not filter using `pav_advantage` if training uses another `advantage`.
- Preserve the original optimizer-batch slot count `N_B`.
- Dense and sparse execution must produce identical loss and gradients.

Example:

```text
selected slots = 128
nonzero retained rows = 20
N_B remains 128
loss = sum(20 nonzero row losses) / 128
```

All-zero selected batch:

- do not call `optimizer.step()`;
- do not increment `global_step`;
- log the event;
- use one explicit tested replay policy for those selected rows.

## Required cleanup

Update all validators so they distinguish:

```text
pre-filter realized rows
post-filter retained rows
```

Do not require:

```text
retained_row_count == allocated_k
```

after sparse filtering.

Allowed invariant:

```text
retained_row_count <= realized_child_count == allocated_k
```

---

# Phase 7 — Fix IDs, scorer contract, distributed scaling, and manifest

## 7.1 Unique IDs

Strict main path requires a unique `tree_instance_id` containing enough information to distinguish stochastic trees for the same question and snapshot:

```text
policy_snapshot_id
rollout_iteration
stable_question_id
UUID or monotonic counter
```

No fallback to only:

```text
(snapshot, question)
```

Replay insertion remains transactional and rejects duplicate `edge_id`.

## 7.2 Scorer and rollout endpoints

Canonical config must resolve one valid topology:

```text
same server
```

or:

```text
two explicit endpoints
```

Strict mode must fail before training when endpoints or reported model versions are missing/mismatched.

Do not keep a default config that necessarily fails its own validator.

## 7.3 Distributed scaling

`ppo_mini_batch_size=128` means global optimizer batch size.

It does not mean 128 rows independently on every DP rank.

Required equivalence:

```text
single rank, global batch 128
==
multi-rank sharding of the same 128 rows after actual DDP/FSDP gradient averaging
```

Verify whether the framework averages or sums gradients and compensate exactly once.

Do not divide by world size twice.

## 7.4 Manifest

Remove these as canonical requirements:

```text
complete_tree_replay
complete_parent_microbatches
node_balanced_invariants_passed
```

Required fields include:

```text
policy_aggregation
segment_token_reduction
replay_sampling_unit
target_edges_per_iteration
resolved_max_edges_per_question_per_iteration
max_edge_age_iterations
ppo_mini_batch_size
ppo_epochs
rollout_iteration
global_step
optimizer_steps_last_iteration
num_optimizer_steps_total
replay_age_uses_rollout_iteration
optimizer_step_accounting_valid
stored_old_log_probs_used
rollout_scorer_weights_verified
no_truncation
unique_tree_ids_verified
```

Operational booleans must come from observed runtime evidence.

Do not set `stored_old_log_probs_used=True` merely because a tensor existed before actor execution. Confirm the actor used the stored denominator path.

Replay diagnostics:

```text
selected edges
unique questions
age histogram
mean age
max age
resolved cap
max selected count for one question
actual optimizer-step count
zero-contribution slots
```

Do not claim that each age bucket contributes exactly `1/8`; log the observed histogram.

---

# Phase 8 — Build a real pre-GPU gate

Update:

```text
scripts/pre_gpu_check.sh
.github/workflows/cpu-ci.yml
```

The gate must test production integration, not only helper functions.

Required checks:

```text
[ ] compileall passes
[ ] Hydra main config composes
[ ] real ActorConfig/PolicyLossConfig instantiate
[ ] mean mode reaches production loss
[ ] sum mode reaches production loss
[ ] 666 auto cap = 33
[ ] 888 auto cap = 73
[ ] edge-level replay respects cap
[ ] replay expiry uses rollout_iteration
[ ] 512 selected edges / 128 batch = four optimizer.step() calls
[ ] trainer global_step increases by four
[ ] scheduler steps four times
[ ] rollout_iteration increases by one
[ ] microbatch gradients match direct 128-row reference
[ ] two-rank gradients match one-rank reference
[ ] dense and sparse zero handling match
[ ] duplicate replay insertion is transactional
[ ] strict scorer endpoint/version tests pass
[ ] manifest records observed counters
[ ] Smoke A-D configs compose
```

Do not skip `test_trainer_contracts.py` if it contains relevant integration tests.

A helper test that computes:

```python
(loss_a + loss_b).backward()
```

is not sufficient evidence for production code that performs separate optimizer steps.

Only print:

```text
PRE_GPU_CHECK=PASS
```

after all required checks pass.

---

# 3. Expected canonical config after the fix

```yaml
gear_tree:
  tree_shape: [6, 6, 6]

  replay_buffer:
    enabled: true
    replay_sampling_unit: edge
    target_edges_per_iteration: 512
    max_edge_age_iterations: 8
    max_edges_per_question_per_iteration: auto
    underfilled_update_policy: postpone_until_divisible
    checkpoint: true

  only_adv_greater_than_zero: false

  gear:
    pilot_execution_mode: fresh_iid
    bound_form: linear
    tail_mode: none
    eps_tail: 0
    allocation_runtime: online_timeout
    allocation_scope: per_queue_flush_within_tree

tree_policy:
  advantage_mode: spo_local
  policy_aggregation: global_segment_mean
  segment_token_reduction: mean
  strict_group_integrity: true

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 128
    ppo_epochs: 1

    policy_loss:
      loss_mode: vdra_segment_mean_ppo
      segment_token_reduction: mean
```

For the `sum` ablation, change only:

```yaml
tree_policy.segment_token_reduction: sum
actor_rollout_ref.actor.policy_loss.segment_token_reduction: sum
```

Do not change replay, allocation, outer segment aggregation, or batch-size settings.

---

# 4. Files likely requiring changes

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/config/*.yaml
verl/recipe/gear_tree/replay_buffer.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/tree_advantage.py
verl/recipe/gear_tree/tree_data.py
verl/recipe/gear_tree/policy_loss.py
verl/recipe/gear_tree/run_manifest.py
verl/recipe/gear_tree/manifest_lifecycle.py
verl/recipe/gear_tree/gear_core/gear/vllm_scorer.py
verl/verl/workers/config/actor.py
verl/verl/workers/actor/dp_actor.py
scripts/pre_gpu_check.sh
.github/workflows/cpu-ci.yml
verl/recipe/gear_tree/tests/
```

Do not edit unrelated algorithms unless required for a shared config schema. Preserve the legacy SPO and node-balanced modes as labeled ablations when practical.

---

# 5. Completion checklist for Claude

Before reporting completion, provide this exact summary with evidence:

```text
CONFIG
[ ] canonical batch = 128
[ ] target edges/rollout iteration = 512
[ ] age window = 8 rollout iterations
[ ] cap = auto
[ ] replay unit = edge
[ ] token mean default
[ ] token sum supported

REPLAY
[ ] 666 resolves to 33
[ ] 888 resolves to 73
[ ] no complete-tree requirement in main path
[ ] age uses rollout_iteration
[ ] commit/rollback works

OPTIMIZER
[ ] 512/128 produces four optimizer.step() calls
[ ] global_step counts actual steps
[ ] rollout_iteration remains one
[ ] scheduler count equals optimizer count
[ ] microbatching does not create extra steps

LOSS
[ ] every 128-slot optimizer batch uses denominator 128
[ ] no denominator 512 reused across four steps
[ ] no main objective_weights tensor
[ ] mean/sum differ correctly
[ ] distributed gradient parity passes

RUNTIME
[ ] unique IDs strict
[ ] scorer/rollout version contract valid
[ ] manifest records observed facts
[ ] checkpoint restores both counters

GATE
[ ] scripts/pre_gpu_check.sh prints PRE_GPU_CHECK=PASS
[ ] Python 3.10 CI passes
[ ] Python 3.12 CI passes
```

Also report:

```text
files changed
commands/tests run
exact test counts
any remaining unsupported path
```

Do not claim GPU readiness if any item above is skipped, xfailed, mocked away from production control flow, or only checked by raw YAML parsing.
