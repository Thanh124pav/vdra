# PLAN.md — Pre-GPU Correctness Checklist

## Goal

Complete every task below before launching a GPU smoke run. After this checklist passes, no known algorithmic, batching, replay, metadata, config, or CPU-integration blocker should remain. GPU smoke may still expose CUDA, Ray, vLLM, FSDP, memory, or distributed-runtime issues.

Canonical main path:

```text
fixed-length SPO-style segments
+ VDRA online rollout allocation
+ fresh_iid final children
+ SPO local segment advantage
+ GRPO-style global segment-average PPO update
```

Canonical tree update:

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
\right),
\]

where:

```text
S(T)       = all realized, non-placeholder training segments in tree T
N_seg(T)   = |S(T)|
S_q        = segments released from queue flush q
n_q        = |S_q|
H_s        = sum of active token log-probability gradients in segment s
```

The queue expression is only a regrouping of the same global segment average. It must not introduce parent-balanced weights, queue-specific optimization weights, or a new optimizer.

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

- Main VDRA must use a segment-average loss, for example:

```text
loss_mode = vdra_segment_mean_ppo
policy_aggregation = global_segment_mean
```

- `vdra_node_balanced_ppo` must not be selected by the main config.
- It may be removed or retained only as a clearly labeled ablation.
- Remove `objective_weights` and all parent-balanced normalization requirements from the main path.
- `parent_group_id`, `allocated_k`, and `sample_multiplicity` may remain as rollout/integrity metadata, but must not determine main-policy importance.
- Queue ratios must not be passed as optimization weights.
- Keep `treetune_ppo` unchanged as the legacy SPO baseline.

**Acceptance tests**

- Main config selects `global_segment_mean`.
- Main loss is unchanged when the same segments are regrouped into different parent groups.
- Main loss is unchanged when queue labels are permuted.
- Main loss differs from the parent-balanced objective on a non-uniform tree.

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
- A realized segment with `A_s = 0` may be omitted from `DataProto`, because its contribution is zero.
- The denominator must still count that segment.
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

For one tree with four realized segments and contributions `[2, 0, 0, 0]`:

```text
retained rows may contain only the first segment
tree_total_segment_count == 4
final tree contribution == 2 / 4
```

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
- Derive `edge_id` from the unique tree ID and child identity.
- `ReplayBuffer.add` must raise on duplicate `edge_id`; never overwrite silently.

**Acceptance tests**

- Two trees for the same question and snapshot have different IDs.
- Both trees coexist in replay.
- IDs survive JSON checkpoint save/load unchanged.

---

## P0.4 — Implement the exact segment-average loss

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
3. Sum active token losses inside the segment:

\[
L_s = \sum_t M_{s,t}\,\ell_{s,t}.
\]

This matches the paper definition:

\[
H_s = \sum_t \nabla_\theta \log \pi_\theta(a_{s,t}\mid\cdot).
\]

For one tree:

\[
L_T
=
\frac{1}{N_{\mathrm{seg}}(T)}
\sum_{s\in\text{retained}(T)} L_s.
\]

Zero-advantage segments may be absent from the numerator but remain in `N_seg(T)`.

For an actor update containing multiple trees, preserve the repository's intended outer prompt/tree averaging explicitly. Do not replace it with parent averaging or token averaging.

Implementation requirements:

- Store counts as integer metadata/tensors (`int32` or `int64`).
- Do not add a stored float `objective_weights` tensor.
- Convert integer denominators to the loss accumulation dtype inside the actor.
- Mixed-precision forward/backward may remain BF16/FP16.
- Loss reduction may accumulate in FP32 for numerical stability.
- `ratio_threshold` must not independently skip arbitrary microbatches in the canonical path.

**Acceptance tests**

- Variable segment length does not alter the segment denominator.
- Token duplication changes the segment contribution consistently with token-sum `H_s`.
- Zero-advantage sparse filtering preserves the exact loss.
- Queue regrouping reproduces the direct global segment sum.

---

## P0.5 — Preserve the exact objective through replay, minibatching, and distributed training

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

- Compute full-update tree counts before mini/microbatch splitting.
- A microbatch must contribute a partial numerator using the original full-tree denominator.
- Do not recompute a local segment mean inside each microbatch.
- Do not divide again by `gradient_accumulation` when the partial loss already uses the full-update normalization.
- Handle dynamic batching and row permutation without changing metadata alignment.
- Account for data-parallel gradient averaging so the multi-rank gradient equals the single-rank reference.

**Acceptance tests**

Compare loss and parameter gradients for the same examples under:

```text
full batch
multiple mini-batches
multiple microbatches
permuted row order
dynamic batching
simulated two-rank sharding + gradient averaging
```

All results must match the explicit direct reference within tolerance.

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
```

Runtime checks must verify:

```text
stored generation-time old log-probs are used
no silent prompt/response truncation
actual request sampling parameters are honored
mixed-depth queue flushes remain legal
allocated_k respects lower/upper bounds and feasible budget slack
pilots are discarded before fresh_iid final generation
rollout and scorer use verified matching weights
```

For scorer/rollout versions:

- require an explicit same-server mode or two explicit endpoints;
- fetch server-reported versions;
- fail strict mode on missing or mismatched versions.

Manifest fields must be set from observed runtime checks, not inferred from config. A main manifest remains invalid until one successful actor update passes all checks.

**Acceptance tests**

- clean synthetic update produces a valid manifest;
- each failed invariant makes it invalid;
- manifest save/load preserves fields;
- stale/missing scorer version fails strict mode.

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
segment-average loss reference tests
zero-advantage sparse-filter tests
queue regrouping/Jensen-identity tests
unique tree/edge ID tests
complete-tree replay tests
full-vs-split gradient parity tests
manifest lifecycle tests
bounded-allocation slack tests
Hydra composition for main and Smoke A-D configs
```

Main pre-GPU config assertions:

```text
fresh_iid
linear bound
global_segment_mean
no parent-balanced main loss
strict runtime checks enabled
```

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
[ ] direct and queue-regrouped updates are numerically identical
[ ] full-batch and split/distributed gradients are numerically identical
[ ] Smoke D config composes without fallback or deprecated main modes
```

Then run Smoke D for at least five successful actor updates. Do not start long experiments until Smoke D confirms finite loss/gradients, valid manifest, no replay/count mismatch, and verified scorer/rollout versions.

The synthetic RQ3/RQ4 scripts are not pre-GPU blockers and are not paper evidence.