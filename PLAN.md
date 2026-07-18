# PLAN.md — Pre-GPU Correctness Checklist

## Goal

Complete every task below before launching a GPU smoke run. After this checklist passes, no known algorithmic, batching, replay, metadata, config, or CPU-integration blocker should remain. GPU smoke may still expose CUDA, Ray, vLLM, FSDP, memory, or distributed-runtime issues.

Canonical main path:

```text
fixed-length SPO-style segments
+ VDRA online rollout allocation
+ fresh_iid final children
+ SPO local segment advantage
+ global segment-average PPO update
+ configurable within-segment token reduction
```

The two aggregation levels must remain separate:

```text
between segments: global_segment_mean
within a segment: segment_token_reduction = mean | sum
```

The default main setting is:

```text
policy_aggregation = global_segment_mean
segment_token_reduction = mean
```

`mean` is the GRPO-style default and avoids making longer segments automatically larger solely because they contain more active tokens. `sum` remains a first-class supported option for the formulation in which a segment score-function term is the sum of its token log-probability gradients. The two modes must share exactly the same rollout, advantage, segment denominator, replay, and batching logic.

For each realized segment `s`, define the number of active tokens after response/probability masking:

\[
Z_s = \sum_t M_{s,t}.
\]

The two supported segment contributions are:

\[
L_s^{\mathrm{mean}}
=
\begin{cases}
\frac{1}{Z_s}\sum_t M_{s,t}\,\ell_{s,t}, & Z_s>0,\\
0, & Z_s=0,
\end{cases}
\]

and

\[
L_s^{\mathrm{sum}}
=
\sum_t M_{s,t}\,\ell_{s,t}.
\]

For either reduction `r ∈ {mean, sum}`, the canonical tree objective is:

\[
L_T^{(r)}
=
\frac{1}{N_{\mathrm{seg}}(T)}
\sum_{s\in\operatorname{retained}(T)} L_s^{(r)}.
\]

Equivalently, when regrouped by queue flush:

\[
L_T^{(r)}
=
\sum_q
\frac{n_q}{N_{\mathrm{seg}}(T)}
\left(
\frac{1}{n_q}
\sum_{s\in\operatorname{retained}(T)\cap\mathcal S_q}
L_s^{(r)}
\right),
\]

where omitted zero-contribution segments are still counted in `n_q` and `N_seg(T)`.

```text
S(T)       = all realized, non-placeholder training segments in tree T
N_seg(T)   = |S(T)|
S_q        = realized segments released from queue flush q
n_q        = |S_q|
```

The queue expression is only a regrouping of the same global segment average. It must not introduce parent-balanced weights, queue-specific optimization weights, or a new optimizer.

For an update containing multiple trees, use the repository's intended outer tree/prompt average explicitly:

\[
L_{\mathrm{update}}^{(r)}
=
\frac{1}{N_T}
\sum_T L_T^{(r)}.
\]

Do not launch a long training run until `scripts/pre_gpu_check.sh` and CPU CI both pass.

---

## P0.1 — Remove the incorrect node-balanced main objective

**Targets**

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/policy_loss.py
verl/verl/workers/actor/dp_actor.py
verl/recipe/gear_tree/run_manifest.py
verl/recipe/gear_tree/manifest_lifecycle.py
```

**Required changes**

- Main VDRA must use a segment-average loss:

```text
loss_mode = vdra_segment_mean_ppo
policy_aggregation = global_segment_mean
segment_token_reduction = mean
```

- `segment_token_reduction` must accept exactly:

```text
mean
sum
```

- `mean` is the canonical/default main setting.
- `sum` is a supported ablation or alternative objective, not an error path.
- `vdra_node_balanced_ppo` must not be selected by the main config.
- It may be removed or retained only as a clearly labeled ablation.
- Remove `objective_weights` and all parent-balanced normalization requirements from the main path.
- `parent_group_id`, `allocated_k`, and `sample_multiplicity` may remain as rollout/integrity metadata, but must not determine main-policy importance.
- Queue ratios must not be passed as optimization weights.
- Keep `treetune_ppo` unchanged as the legacy SPO baseline.

**Acceptance tests**

- Main config selects `global_segment_mean`.
- Main config defaults to `segment_token_reduction=mean`.
- Both `mean` and `sum` modes execute through the same main loss implementation.
- Main loss is unchanged when the same segments are regrouped into different parent groups.
- Main loss is unchanged when queue labels are permuted.
- Main loss differs from the parent-balanced objective on a non-uniform tree.
- Invalid token-reduction strings fail clearly at startup.

---

## P0.2 — Count the correct segment set while allowing zero-advantage filtering

**Targets**

```text
verl/recipe/gear_tree/tree_rollout.py
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/tree_advantage.py
verl/recipe/gear_tree/tree_data.py
```

**Required changes**

- `only_adv_greater_than_zero` may remain enabled for compute efficiency.
- A realized segment with `A_s = 0` may be omitted from `DataProto`, because its contribution is zero in both token-reduction modes.
- The segment denominator must still count that segment.
- For every tree, compute before advantage filtering:

```text
tree_total_segment_count = N_seg(T)
```

- For every queue flush, compute before advantage filtering:

```text
queue_released_segment_count[q] = n_q
```

- Count only realized segment samples.
- Exclude administrative `pruned=True` placeholders from both numerator and denominator.
- Preserve `tree_total_segment_count` on every retained edge from that tree.
- Preserve `queue_flush_id` and queue counts for logging/theoretical validation only.
- In `fresh_iid`, preserve:

```text
allocated_k
realized_child_count
sample_multiplicity == 1
```

- Do not require retained-row count to equal `allocated_k` after zero-advantage filtering.
- Instead require:

```text
retained_row_count <= realized_child_count == allocated_k
```

**Acceptance tests**

For one tree with four realized segments and segment contributions `[2, 0, 0, 0]`:

```text
retained rows may contain only the first segment
tree_total_segment_count == 4
final tree contribution == 2 / 4
```

Run this test for both token-reduction modes.

Also verify:

- zero-advantage filtering does not change the loss;
- a pruned placeholder does not change `N_seg(T)`;
- queue counts sum to the tree count:

```text
sum_q n_q == N_seg(T)
```

---

## P0.3 — Make every stochastic tree instance globally unique

**Targets**

```text
verl/recipe/gear_tree/tree_rollout.py
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/tree_advantage.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/replay_buffer.py
```

**Required changes**

- Create one `tree_instance_id` when each stochastic tree starts.
- The ID must distinguish repeated trees for the same question and policy snapshot. Include:

```text
policy_snapshot_id
rollout_iteration
stable_question_id
per-tree UUID or monotonic counter
```

- Use this value as `tree_id` through generation, edge extraction, replay, tensorization, and manifest logging.
- Strict main runs must raise if a unique tree ID is missing; do not fall back to `(snapshot, question)` alone.
- Derive `edge_id` from the unique tree ID and child identity.
- `ReplayBuffer.add` must raise on duplicate `edge_id`; never overwrite silently.
- Replay insertion must be transactional: validate all incoming edge IDs before inserting any edge from the tree/batch.

**Acceptance tests**

- Two trees for the same question and snapshot have different IDs.
- Both trees coexist in replay.
- IDs survive JSON checkpoint save/load unchanged.
- A duplicate in the middle of an insertion batch leaves the replay buffer unchanged.

---

## P0.4 — Implement the configurable global segment-average loss

**Targets**

```text
verl/recipe/gear_tree/tree_data.py
verl/recipe/gear_tree/policy_loss.py
verl/verl/workers/actor/dp_actor.py
```

**Required changes**

For each retained segment row `s`:

1. Compute the PPO-clipped token surrogate using stored old log-probabilities.
2. Apply response and probability masks.
3. Reduce active token losses according to `segment_token_reduction`:

```text
mean: L_s = sum_t M_st * ell_st / sum_t M_st
sum:  L_s = sum_t M_st * ell_st
```

4. When a segment has zero active tokens after masking, set its numerator contribution to zero without changing `N_seg(T)`.
5. Divide the retained segment numerator by the original pre-filter `tree_total_segment_count`.
6. Average the resulting tree losses across trees/prompts in the update.

The `sum` option corresponds to the score-function convention:

\[
H_s^{\mathrm{sum}}
=
\sum_t M_{s,t}\nabla_\theta\log\pi_\theta(a_{s,t}\mid\cdot).
\]

The `mean` option uses the length-normalized segment score:

\[
H_s^{\mathrm{mean}}
=
\frac{1}{Z_s}
\sum_t M_{s,t}\nabla_\theta\log\pi_\theta(a_{s,t}\mid\cdot),
\qquad Z_s>0.
\]

Neither option may change the weight of a segment based on its parent branch factor.

Implementation requirements:

- Store counts as integer metadata/tensors (`int32` or `int64`).
- Do not add a stored float `objective_weights` tensor on the main path.
- Convert integer denominators to the loss accumulation dtype inside the actor.
- Mixed-precision forward/backward may remain BF16/FP16.
- Loss reduction may accumulate in FP32 for numerical stability.
- `ratio_threshold` must not independently skip arbitrary microbatches in the canonical path.
- The selected token-reduction mode must be recorded in the run manifest.

**Acceptance tests**

Shared tests for both modes:

- parent regrouping does not alter the loss;
- queue regrouping reproduces the direct global segment objective;
- zero-advantage sparse filtering preserves the exact loss;
- variable segment length does not alter the segment denominator;
- row permutation does not alter the loss.

Mode-specific tests:

```text
token_mean:
- duplicating every active token with identical token loss leaves L_s unchanged;
- longer segments do not receive larger weight solely because of length.

token_sum:
- duplicating every active token with identical token loss doubles L_s;
- token contribution matches the summed score-function convention.
```

A non-uniform-length fixture must demonstrate that `mean` and `sum` are genuinely different options rather than aliases.

---

## P0.5 — Preserve the selected objective through replay, minibatching, and distributed training

**Targets**

```text
verl/recipe/gear_tree/replay_buffer.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/verl/workers/actor/dp_actor.py
```

**Required changes**

- Replay must reserve complete selected trees; it must not drop nonzero retained segments from a selected tree.
- `target_edges_per_update` is a soft packing target.
- Preserve on every row:

```text
tree_id
tree_total_segment_count
queue_flush_id
stored old_log_probs
```

- Preserve the update-level `segment_token_reduction` setting through trainer → actor.
- Compute full-update tree counts before mini/microbatch splitting.
- A microbatch must contribute a partial numerator using the original full-tree denominator.
- Do not recompute a local segment mean across rows inside each microbatch.
- Within-row token reduction may be `mean` or `sum`; between-row aggregation must remain the same global segment mean.
- Do not divide again by `gradient_accumulation` when the partial loss already uses the full-update normalization.
- One logical replay update must have explicitly defined optimizer-step semantics. The reference path should accumulate all partial numerators and execute one optimizer step for that logical update.
- Handle dynamic batching and row permutation without changing metadata alignment.
- Account for data-parallel gradient averaging so the multi-rank gradient equals the single-rank reference.

**Acceptance tests**

For each `segment_token_reduction ∈ {mean, sum}`, compare loss and parameter gradients for the same examples under:

```text
full batch
multiple mini-batches with one logical optimizer step
multiple microbatches
permuted row order
dynamic batching
simulated two-rank sharding + explicit gradient averaging
```

All results must match the explicit direct reference within tolerance.

Tests must model the production control flow, including optimizer-step placement; summing two losses and calling `backward()` once is not sufficient evidence for a path that performs multiple optimizer steps.

---

## P0.6 — Verify the remaining runtime contracts

**Targets**

```text
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/gear_core/gear/vllm_scorer.py
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/run_manifest.py
verl/recipe/gear_tree/manifest_lifecycle.py
```

**Required changes**

Main config must use:

```text
pilot_execution_mode = fresh_iid
bound_form = linear
tail_mode = none
eps_tail = 0
allocation_runtime = online_timeout
allocation_scope = per_queue_flush_within_tree
policy_aggregation = global_segment_mean
segment_token_reduction = mean
```

`segment_token_reduction=sum` must remain a supported explicit override for ablation/sensitivity runs.

Runtime checks must verify:

```text
stored generation-time old log-probs are used
no silent prompt/response truncation
actual request sampling parameters are honored
mixed-depth queue flushes remain legal
allocated_k respects lower/upper bounds and feasible budget slack
pilots are discarded before fresh_iid final generation
rollout and scorer use verified matching weights
tree_total_segment_count is computed before filtering
sum_q queue_released_segment_count[q] == tree_total_segment_count
selected token-reduction mode is honored by the actor
```

For scorer/rollout versions:

- require an explicit same-server mode or two explicit endpoints;
- fetch server-reported versions;
- fail strict mode on missing or mismatched versions.

Manifest fields must be set from observed runtime checks, not inferred from config. A main manifest remains invalid until one successful actor update passes all checks.

The manifest must record at least:

```text
policy_aggregation = global_segment_mean
segment_token_reduction = mean | sum
complete_tree_replay
segment_count_invariants_passed
stored_old_log_probs_used
rollout_scorer_weights_verified
no_truncation
```

It must not require parent-balanced normalization for a valid main run.

**Acceptance tests**

- clean synthetic update produces a valid manifest in `mean` mode;
- clean synthetic update also produces a valid explicitly labeled `sum`-mode run;
- each failed invariant makes the manifest invalid;
- manifest save/load preserves the token-reduction setting and all observed fields;
- stale/missing scorer version fails strict mode;
- the repository's default strict config selects a valid explicit scorer topology or fails during config composition with a clear actionable message.

---

## P0.7 — Put all correctness tests behind one pre-GPU gate

**Targets**

```text
.github/workflows/cpu-ci.yml
verl/recipe/gear_tree/tests/
tests/
scripts/pre_gpu_check.sh
```

**Required changes**

CPU CI and `scripts/pre_gpu_check.sh` must run:

```text
python -m compileall vdra_core verl/recipe/gear_tree
global segment-average reference tests for token_mean
global segment-average reference tests for token_sum
mean-vs-sum non-alias test
zero-advantage sparse-filter tests
queue regrouping/Jensen-identity tests
unique tree/edge ID tests
transactional replay insertion tests
complete-tree replay tests
full-vs-split gradient parity tests for both token reductions
manifest lifecycle tests
bounded-allocation slack tests
Hydra composition for main and Smoke A-D configs
```

Main pre-GPU config assertions:

```text
fresh_iid
linear bound
global_segment_mean
segment_token_reduction = mean
no parent-balanced main loss
no main-path objective_weights
strict runtime checks enabled
```

A separate config-composition test must verify that overriding:

```text
segment_token_reduction = sum
```

is accepted without changing allocation, replay, segment counting, or outer aggregation settings.

Do not skip or xfail a known correctness blocker. Python 3.10 and 3.12 CI jobs must pass.

Print only after all checks succeed:

```text
PRE_GPU_CHECK=PASS
```

---

# Definition of done

The repository is ready for a GPU smoke run only when:

```text
[ ] P0.1-P0.7 are complete
[ ] scripts/pre_gpu_check.sh prints PRE_GPU_CHECK=PASS
[ ] CPU CI is green on Python 3.10 and 3.12
[ ] no known correctness blocker is skipped or xfailed
[ ] main config uses global segment mean, not node balancing
[ ] main config defaults to segment_token_reduction=mean
[ ] segment_token_reduction=sum is a tested supported override
[ ] direct and queue-regrouped updates are numerically identical in both modes
[ ] full-batch and split/distributed gradients are numerically identical in both modes
[ ] Smoke D config composes without fallback or deprecated main modes
```

Then run Smoke D for at least five successful actor updates with the default `mean` reduction. Do not start long experiments until Smoke D confirms finite loss/gradients, valid manifest, no replay/count mismatch, and verified scorer/rollout versions.

After the default smoke passes, run a short `sum`-reduction smoke as an objective ablation. It need not be the main paper configuration unless experiments show a clear benefit and the paper is updated consistently.

The synthetic RQ3/RQ4 scripts are not pre-GPU blockers and are not paper evidence.
