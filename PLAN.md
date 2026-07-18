# PLAN.md — Pre-GPU Correctness Checklist

## Goal

Complete every P0 task below before launching a GPU smoke run. After this checklist passes, no known algorithmic, replay, batching, counter, metadata, config, or CPU-integration blocker should remain. GPU smoke may still expose CUDA, Ray, vLLM, FSDP, memory, or distributed-runtime issues.

Canonical main path:

```text
fixed-length SPO-style segments
+ VDRA online rollout allocation
+ fresh_iid final children
+ SPO local segment advantage
+ edge-level replay over recent rollout iterations
+ GRPO-style global segment-average PPO
+ configurable within-segment token reduction
```

The following concepts must remain separate:

```text
rollout_iteration   = one tree-generation / replay-fill cycle
global_step         = one actual optimizer.step()
segment aggregation = equal average over selected segment slots
within-segment loss = token mean or token sum
```

Default main settings:

```text
target_edges_per_iteration = 512
ppo_mini_batch_size = 128
ppo_epochs = 1
segment_token_reduction = mean
policy_aggregation = global_segment_mean
max_edge_age_iterations = 8
max_edges_per_question_per_iteration = auto
replay_sampling_unit = edge
```

With a full 512-edge replay sample and `ppo_mini_batch_size=128`, one rollout iteration executes four optimizer steps:

\[
512 / 128 = 4.
\]

`global_step` must therefore increase by four, while `rollout_iteration` increases by one.

---

## 1. Mathematical contract

### 1.1 Full generated tree and queue regrouping

For a generated tree `T`, let:

```text
S(T)       = all realized, non-placeholder segments in T
N_seg(T)   = |S(T)|
S_q        = segments released from queue flush q
n_q        = |S_q|
```

The full-tree gradient estimator used in the method derivation is:

\[
\widehat g_T
=
\frac{1}{N_{\mathrm{seg}}(T)}
\sum_{s\in\mathcal S(T)} A_s H_s
=
\sum_q
\frac{n_q}{N_{\mathrm{seg}}(T)}
\left(
\frac{1}{n_q}
\sum_{s\in\mathcal S_q} A_s H_s
\right).
\]

The queue expression is only a regrouping of the same segment average. It must not introduce parent-balanced weights, queue-specific policy weights, or a new optimizer.

### 1.2 Replay training batch

TreeTune-style replay does not train every full tree in one optimizer step. It samples segment/edge slots from recent rollout iterations. For one global optimizer batch `B`, define:

```text
N_B = number of selected segment slots before optional zero-contribution sparsification
```

For each segment `s`, after response/probability masking:

\[
Z_s = \sum_t M_{s,t}.
\]

The supported within-segment reductions are:

\[
L_s^{\mathrm{mean}}
=
\begin{cases}
\frac{1}{Z_s}\sum_t M_{s,t}\ell_{s,t}, & Z_s>0,\\
0, & Z_s=0,
\end{cases}
\]

and

\[
L_s^{\mathrm{sum}}
=
\sum_t M_{s,t}\ell_{s,t}.
\]

For either `r ∈ {mean, sum}`, the optimizer-batch objective is:

\[
L_B^{(r)}
=
\frac{1}{N_B}
\sum_{s\in\operatorname{retained}(B)} L_s^{(r)}.
\]

If zero-contribution segments are omitted from model execution, they remain counted in `N_B`. Parent IDs, queue IDs, branch factors, tree sizes, and replay ages must not change a retained segment's policy weight.

`mean` is the GRPO-style default. `sum` remains a first-class supported option and ablation. Both modes must share exactly the same rollout, replay, selected segment slots, advantages, batching, and optimizer-step control flow.

---

# P0.1 — Wire the canonical loss and token-reduction option correctly

**Targets**

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/policy_loss.py
verl/verl/workers/config/actor.py
verl/verl/workers/actor/dp_actor.py
verl/recipe/gear_tree/run_manifest.py
```

**Required changes**

- Main VDRA must use:

```text
loss_mode = vdra_segment_mean_ppo
policy_aggregation = global_segment_mean
segment_token_reduction = mean
```

- `segment_token_reduction` must accept exactly `mean` and `sum`.
- Declare the field in the actual instantiated config schema.
- Use one authoritative actor-side source of truth, preferably:

```text
actor_rollout_ref.actor.policy_loss.segment_token_reduction
```

- The loss must read that exact field; it must not silently fall back to `mean` when the config says `sum`.
- If `tree_policy.segment_token_reduction` remains for manifest/readability, startup validation must require equality with the actor field.
- `vdra_node_balanced_ppo` may remain only as a clearly labeled ablation.
- Remove `objective_weights`, `segment_objective_weights`, and parent-balanced float coefficients from the canonical main path.
- Keep `treetune_ppo` unchanged for the legacy SPO baseline.

**Acceptance tests**

- Hydra-compose and instantiate the real `ActorConfig` and `PolicyLossConfig` in both modes.
- A production-path `sum` config reaches the actor and executes token sum rather than token mean.
- Invalid strings fail during startup/config validation.
- Parent regrouping and queue-label permutation do not alter the main loss.
- `mean` and `sum` produce different values on non-uniform segment lengths.

---

# P0.2 — Restore TreeTune-style replay cadence and automatic per-question cap

**Targets**

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/replay_buffer.py
verl/recipe/gear_tree/gear_ray_trainer.py
```

**Required changes**

Rename the replay concepts so their meaning is explicit:

```text
target_edges_per_iteration
max_edge_age_iterations
max_edges_per_question_per_iteration
replay_sampling_unit = edge
```

Do not call 512 edges one optimizer update. It is the target number of segment samples consumed during one rollout iteration.

Main defaults:

```text
target_edges_per_iteration = 512
ppo_mini_batch_size = 128
ppo_epochs = 1
max_edge_age_iterations = 8
max_edges_per_question_per_iteration = auto
```

Replay behavior:

1. Add newly generated edge records with `generation_rollout_iteration`.
2. Expire an edge when:

\[
\text{current rollout iteration}
-
\text{generation rollout iteration}
\geq
\text{max edge age iterations}.
\]

3. Group available edges by stable `question_id`.
4. Sample at most the resolved per-question cap.
5. Sample individual edges/segments, not mandatory complete trees.
6. Remove only edges committed after a successful actor update.
7. Roll back the reservation unchanged on failure.
8. Never duplicate rows to fill an underfilled sample.

The automatic cap must be derived from the configured maximum tree size. For tree shape:

\[
[b_1,b_2,\ldots,b_D],
\]

the maximum number of non-root edges in one full tree is:

\[
E_{\max}
=
\sum_{d=1}^{D}
\prod_{\ell=1}^{d} b_\ell.
\]

If the config can generate `R` stochastic trees for the same question in one rollout iteration, use:

\[
E_{\max}^{\text{question/iteration}} = R E_{\max}.
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
tree_shape = [6,6,6] -> E_max = 258 -> cap = 33
tree_shape = [8,8,8] -> E_max = 584 -> cap = 73
```

The canonical main config must use `auto`; a numeric override may exist only for controlled compatibility/ablation runs and must be recorded in the manifest.

**Acceptance tests**

- `[6,6,6]`, age 8 resolves to 33.
- `[8,8,8]`, age 8 resolves to 73.
- Increasing tree size cannot leave the resolved cap unchanged because of a stale hard-coded 32.
- No sampled edge has age `>= max_edge_age_iterations`.
- Per-question selected count never exceeds the resolved cap.
- Selected IDs are removed only after commit; rollback restores all reserved rows.
- Replay logs an age histogram rather than claiming a forced `1/8` split across age buckets.

---

# P0.3 — Make counters, scheduler, and logging match actual optimizer steps

**Targets**

```text
verl/recipe/gear_tree/gear_ray_trainer.py
verl/verl/workers/actor/dp_actor.py
checkpoint / scheduler wiring
```

**Required changes**

Maintain separate counters:

```text
rollout_iteration
optimizer_steps_this_iteration
global_step
```

Semantics:

- `rollout_iteration += 1` once per generation/replay-fill cycle.
- `global_step += 1` after every successful actual `optimizer.step()`.
- `optimizer_steps_this_iteration` is reset at the start of each rollout iteration and incremented after each successful optimizer step.
- `num_optimizer_steps_total` must equal `global_step`.
- Replay age uses `rollout_iteration`, never `global_step`.
- Learning-rate scheduler stepping uses actual optimizer steps.
- Any setting expressed in optimizer steps—warmup, total training steps, save frequency, and optimizer-step-based evaluation—must use `global_step`.
- Rollout-frequency metrics and replay retention use `rollout_iteration`.

Required logs:

```text
training/rollout_iteration
training/global_step
training/optimizer_steps_this_iteration
training/num_optimizer_steps_total
training/selected_edges_this_iteration
training/expected_optimizer_steps_from_selected_edges
```

With 512 selected edges, mini-batch 128, epoch 1:

```text
optimizer_steps_this_iteration = 4
global_step increases by 4
rollout_iteration increases by 1
```

Do not label one call to `update_actor` as one optimizer step when it internally performs several steps.

Underfilled canonical behavior:

- Do not start an optimizer mini-batch with an undefined denominator.
- Prefer postponing until the selected edge count is divisible by the global mini-batch size 128.
- Any alternative final-smaller-batch behavior must be explicit, tested, and recorded.

**Acceptance tests**

- A synthetic 512-edge iteration with mini-batch 128 performs exactly four `optimizer.step()` calls.
- `global_step` advances by four and `rollout_iteration` by one.
- Replay age does not advance four times during that iteration.
- Scheduler `.step()` count equals successful optimizer-step count.
- A failed optimizer step does not increment `global_step` and rolls back the associated replay reservation.
- Checkpoint save/load restores both counters without conflating them.

---

# P0.4 — Define mini-batch and microbatch loss semantics exactly

**Targets**

```text
verl/recipe/gear_tree/tree_data.py
verl/recipe/gear_tree/policy_loss.py
verl/verl/workers/actor/dp_actor.py
```

**Required changes**

For each global optimizer mini-batch of 128 selected segment slots:

1. Compute the PPO-clipped token surrogate using stored generation-time old log-probabilities.
2. Apply response and probability masks.
3. Reduce within each segment using `mean` or `sum`.
4. Average equally across the 128 selected segment slots.
5. Execute one optimizer step after all microbatches belonging to that 128-slot optimizer batch have accumulated gradients.

Do not normalize a 128-edge optimizer step using a denominator computed over the whole 512-edge rollout iteration.

Do not execute four optimizer steps using four partial numerators that were all normalized as fractions of one 512-edge objective.

Microbatching is memory partitioning only:

```text
one global optimizer batch of 128
-> one or more per-GPU microbatches
-> accumulated gradient of the 128-slot mean
-> one optimizer.step()
```

Implementation requirements:

- Store counts as integer metadata/tensors.
- Do not store main-path float objective coefficients.
- Accumulate the final weighted reduction in FP32 when useful.
- Mixed-precision model forward/backward may remain BF16/FP16.
- Preserve row/metadata alignment under dynamic batching and permutation.
- `ratio_threshold` must not silently discard an arbitrary microbatch and change the batch denominator.

**Acceptance tests**

For both `mean` and `sum`:

- direct 128-row loss equals the sum of microbatch partial numerators divided by the original 128-slot count;
- parameter gradients match under one microbatch, several microbatches, dynamic batching, and row permutation;
- tests call `optimizer.step()` at the same location as production;
- a test that merely sums losses and calls `backward()` once is not sufficient evidence for a production path with different optimizer-step placement.

Mode-specific checks:

```text
token mean:
- duplicating identical active tokens leaves one segment loss unchanged

token sum:
- duplicating identical active tokens doubles one segment loss
```

---

# P0.5 — Preserve IDs, exact training rows, and optional zero sparsification

**Targets**

```text
verl/recipe/gear_tree/tree_rollout.py
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/tree_advantage.py
verl/recipe/gear_tree/tree_data.py
verl/recipe/gear_tree/replay_buffer.py
```

**Required changes**

Unique identity:

- Create one globally unique `tree_instance_id` when each stochastic tree starts.
- Include policy snapshot, rollout iteration, stable question ID, and UUID/counter.
- Strict main runs must raise if it is missing; never fall back to `(snapshot, question)` alone.
- Derive each `edge_id` from unique tree ID plus child identity.
- Replay insertion must reject duplicate IDs transactionally.

Segment records:

- Every realized, non-pruned child segment is one replay sampling slot.
- Administrative `pruned=True` placeholders are not replay/training slots.
- Preserve:

```text
tree_id
edge_id
question_id
queue_flush_id
generation_rollout_iteration
allocated_k
realized_child_count
sample_multiplicity
actor_shifted_log_probs
actual training advantage
```

Zero-contribution sparsification:

- Canonical correctness may retain all zero-advantage rows.
- An optional sparse path may avoid model execution for rows whose actual training contribution is zero.
- The filter must use the exact advantage tensor sent to the policy loss, not `pav_advantage` or another diagnostic scalar.
- Zero slots selected for an optimizer batch remain counted in that batch's `N_B`.
- A batch whose every selected slot has zero contribution should skip `optimizer.step()`, log the event, leave `global_step` unchanged, and commit/retain replay rows according to one explicit tested policy.

Full-tree counts remain useful for rollout/theory validation:

```text
tree_total_segment_count = N_seg(T)
queue_released_segment_count[q] = n_q
sum_q n_q = N_seg(T)
```

They must be computed before optional filtering, but they are not main policy weights under edge-level replay.

**Acceptance tests**

- Two stochastic trees for the same question/snapshot have distinct IDs and coexist in replay.
- Duplicate insertion leaves replay unchanged.
- Filter uses actual training advantage.
- Dense and sparse execution give identical loss and gradients when zero slots are counted in `N_B`.
- Pruned placeholders do not affect tree or batch segment counts.
- Missing required count/ID metadata raises in strict mode; never silently default to 1 or an ambiguous ID.

---

# P0.6 — Fix distributed scaling and runtime contracts

**Targets**

```text
verl/verl/workers/actor/dp_actor.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/gear_core/gear/vllm_scorer.py
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
```

**Required changes**

Main runtime contract:

```text
pilot_execution_mode = fresh_iid
bound_form = linear
tail_mode = none
eps_tail = 0
allocation_runtime = online_timeout
allocation_scope = per_queue_flush_within_tree
policy_aggregation = global_segment_mean
segment_token_reduction = mean
ppo_mini_batch_size = 128
ppo_epochs = 1
```

Verify at runtime:

```text
stored generation-time old log-probs are used
no silent prompt/response truncation
request sampling parameters are honored
mixed-depth queue flushes remain legal
allocated_k respects bounds and feasible budget slack
pilots are discarded before fresh_iid final generation
rollout and scorer use verified matching weights
selected token-reduction mode reaches the actor
resolved replay cap matches tree shape and age window
replay age is based on rollout_iteration
```

Scorer/rollout topology:

- Select exactly one explicit supported mode: same server or two explicit endpoints.
- Fetch server-reported versions.
- Fail strict mode on missing or mismatched versions.
- The canonical smoke config must not contain an internally invalid endpoint combination.

Distributed objective:

- `ppo_mini_batch_size=128` is the effective global optimizer batch size, not 128 independently on every DP rank.
- Shard the 128 selected segment slots across ranks.
- Account for DDP/FSDP gradient averaging so the final gradient equals the single-rank 128-slot mean.
- Do not accidentally divide by data-parallel world size twice.
- Do not accidentally multiply the learning rate effect by world size.

**Acceptance tests**

- Single-rank and simulated/real two-rank gradients match for both token reductions.
- Test actual reducer averaging semantics, not merely two losses in one process.
- Same-server and two-endpoint scorer contracts each have positive and negative tests.
- Default smoke config resolves one valid topology.
- Runtime detects a stale scorer version before allocation/training.

---

# P0.7 — Make the run manifest record observed facts

**Targets**

```text
verl/recipe/gear_tree/run_manifest.py
verl/recipe/gear_tree/manifest_lifecycle.py
verl/recipe/gear_tree/gear_ray_trainer.py
```

**Required manifest fields**

```text
policy_aggregation = global_segment_mean
segment_token_reduction = mean | sum
replay_sampling_unit = edge
target_edges_per_iteration = 512
resolved_max_edges_per_question_per_iteration
max_edge_age_iterations = 8
ppo_mini_batch_size = 128
ppo_epochs = 1
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
segment_count_invariants_passed
```

Operational booleans must be set from observed runtime behavior, not inferred from YAML.

Remove `complete_tree_replay` and `complete_parent_microbatches` as canonical main-run requirements. Edge-level replay is intentional and matches the TreeTune diversity mechanism.

Record replay diagnostics:

```text
selected edges
unique questions
edge-age histogram
mean/max edge age
resolved auto cap
per-question selected-count max
zero-contribution selected slots
actual optimizer steps
```

A main manifest remains invalid until at least one successful optimizer step passes all relevant runtime checks.

**Acceptance tests**

- A 512-edge synthetic iteration records four optimizer steps at batch 128.
- Manifest distinguishes rollout iteration from optimizer step.
- Each failed observed invariant makes the manifest invalid.
- Save/load preserves counters and resolved cap.
- Manifest never claims a uniform `1/8` age composition unless that composition was actually observed.

---

# P0.8 — Put all correctness checks behind one pre-GPU gate

**Targets**

```text
.github/workflows/cpu-ci.yml
scripts/pre_gpu_check.sh
verl/recipe/gear_tree/tests/
tests/
```

CPU CI and `scripts/pre_gpu_check.sh` must run:

```text
python -m compileall vdra_core verl/recipe/gear_tree
Hydra composition and real dataclass instantiation
main token-mean reference tests
supported token-sum reference tests
mean-vs-sum non-alias test
128-row optimizer-batch normalization tests
microbatch gradient parity tests
actual optimizer-step placement/accounting tests
512/128 -> four optimizer-step test
rollout_iteration vs global_step replay-age test
auto-cap tests for 666 and 888 trees
edge-level replay sampling and age-expiry tests
zero-contribution dense-vs-sparse tests
unique tree/edge ID tests
transactional replay tests
distributed gradient-scaling tests
scorer/rollout version-contract tests
manifest lifecycle tests
bounded-allocation slack tests
Smoke A-D config composition
```

Main config assertions:

```text
fresh_iid
linear bound
global_segment_mean
segment_token_reduction = mean
target_edges_per_iteration = 512
ppo_mini_batch_size = 128
ppo_epochs = 1
max_edge_age_iterations = 8
max_edges_per_question_per_iteration = auto
replay_sampling_unit = edge
no parent-balanced main loss
no main-path objective_weights
strict runtime checks enabled
```

A separate composition/integration test must verify that overriding `segment_token_reduction=sum` changes only the within-segment reduction.

Do not skip or xfail a known correctness blocker. Python 3.10 and 3.12 CI jobs must pass.

Print only after all checks succeed:

```text
PRE_GPU_CHECK=PASS
```

---

# Definition of done

The repository is ready for GPU Smoke D only when:

```text
[ ] P0.1-P0.8 are complete
[ ] scripts/pre_gpu_check.sh prints PRE_GPU_CHECK=PASS
[ ] CPU CI is green on Python 3.10 and 3.12
[ ] real Hydra/dataclass composition passes
[ ] main config uses global segment mean, not node balancing
[ ] token mean is default and token sum is a tested override
[ ] target_edges_per_iteration is 512
[ ] effective global ppo_mini_batch_size is 128
[ ] one full replay iteration produces four actual optimizer steps
[ ] global_step counts optimizer.step() calls
[ ] rollout_iteration alone controls replay age
[ ] per-question cap is automatically derived from tree shape and age window
[ ] edge-level replay is used intentionally; complete-tree replay is not required
[ ] no main-path float objective-weight tensors remain
[ ] single-rank and distributed gradients match the explicit reference
[ ] scorer/rollout versions are verified
[ ] manifest fields are based on observed runtime facts
[ ] no known correctness blocker is skipped or xfailed
```

Then run Smoke D for at least five rollout iterations and report both:

```text
rollout iterations completed
optimizer global steps completed
```

With a full 512-edge sample every iteration, five rollout iterations should normally produce twenty optimizer steps at batch size 128 and one PPO epoch.

Do not start long experiments until Smoke D confirms finite loss/gradients, valid manifest, correct replay-age behavior, correct optimizer-step accounting, no counter drift after resume, no scorer mismatch, and no truncation.

After the default `mean` smoke passes, run a short `sum` smoke as an objective ablation. The synthetic RQ3/RQ4 scripts are not pre-GPU blockers and are not paper evidence.
