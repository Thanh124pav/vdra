# PLAN.md — Remaining VERL Migration Fixes for VDRA

## 0. Goal

Complete the `verl/recipe/gear_tree` implementation so that:

1. SPO-tree and VDRA use the same trainer-side sampling protocol.
2. Replay-buffered edges preserve the exact behavior-policy log-probabilities produced at generation time.
3. Pruning and expansion are unified in one exact bounded integer allocation problem.
4. The async likelihood scorer uses the same policy snapshot as the rollout generator.
5. The custom VERL trainer follows the required VERL metadata, stopping, batching, checkpoint, and distributed-training contracts.
6. The direct short-horizon VDRA variant with `eps_tail = 0` can run in strict mode without requiring a tail-calibration artifact.
7. All fixes are covered by deterministic unit tests and a short Ray + FSDP + async-vLLM smoke test.

Do not silently fall back to another algorithm or silently change the requested experimental protocol.

---

# 1. Scope and priorities

## P0 — Must be completed before any training run

1. Replace the old donor-reserve-receiver allocation path with the unified integer problem:

   $$
   \min_{\{k_s\in\mathbb Z_+\}}
   \sum_{s\in\mathcal Q}\frac{C_s}{k_s}
   $$

   subject to:

   $$
   \sum_{s\in\mathcal Q}k_s=B_{\mathrm{target}},
   \qquad
   \ell_s\leq k_s\leq u_s.
   $$

2. Implement the exact greedy marginal integer solver using:

   $$
   \Delta_s(k)
   =
   \frac{C_s}{k(k+1)}.
   $$

3. Use `predicted_k` as a hard upper bound only for nodes satisfying:

   $$
   k_s^{\mathrm{pred}}<n_s.
   $$

4. Remove relax-and-round from the default allocation path.
5. Enforce exact allocation feasibility and exact target-budget preservation.
6. Enforce:

   $$
   \#\text{final children}(s)\leq k_s^\star.
   $$

7. Remove per-iteration recomputation of behavior `old_log_probs`.
8. Add a trainer-level replay buffer.
9. Add the SPO-compatible edge sampling protocol:
   - at most 32 edges per question;
   - at most 512 edges per optimizer update;
   - remove only edges actually sent to `update_actor`;
   - retain unused edges;
   - expire edges with age at least 8 optimizer steps.
10. Set all required VERL metadata before `update_actor`.
11. Respect `trainer.total_training_steps`.
12. Ensure the likelihood scorer and rollout generator use the same current policy snapshot.
13. Add a strict direct-tail mode that permits `eps_tail = 0`.
14. Run a two-to-five-step end-to-end smoke test.

## P1 — Required before main paper experiments

1. Add replay-buffer checkpoint save/load.
2. Add complete allocation, buffer, policy-snapshot, and branch-accounting logging.
3. Add allocation solver microbenchmarks.
4. Add controlled config presets matching the legacy SPO experiment.
5. Validate rollout log-probabilities against actor log-probabilities offline.
6. Verify queue batching with runtime histograms.
7. Clearly separate exact baselines from style variants.
8. Keep the continuous relaxation only for theory, diagnostics, and optional legacy ablations.

# 2. Do not modify the already-fixed queue semantics unless tests fail

The latest implementation already expands eligible siblings concurrently and schedules flushed subtree expansion as independent tasks.

Preserve these properties:

```python
await asyncio.gather(
    *(_process_expandable(child, depth + 1) for child in expandable)
)
```

and:

```python
task = asyncio.create_task(_expand_flushed_item(result, item))
```

The timeout worker must not block while recursively expanding a full subtree.

Keep the following runtime invariants:

```text
all queued futures are eventually resolved
all queues are empty before tree return
all finalized nodes have a reward
every flushed allocation is feasible
sum(final_allocations) == target_budget
```

Add or retain runtime metrics:

```text
queue_size_at_flush
flush_reason
queue_wait_seconds
capacity_flush_count
timeout_flush_count
final_drain_count
```

The main queue test must verify that a sibling frontier can produce at least one flush with:

```text
queue_size_at_flush > 1
```

under a deterministic fake async generator.

---

# 3. Behavior-policy old log-probabilities

## 3.1 Required semantics

Suppose an edge is generated at optimizer step $t$ by behavior policy $\pi_{\theta_t}$.

For generated response token $a_j$, store:

$$
\ell_{t,j}
=
\log \pi_{\theta_t}(a_j \mid s_j).
$$

When the edge is trained at a later optimizer step $t+d$, PPO must compare the current-policy log-probability against the stored behavior-policy log-probability:

$$
r_j(\theta)
=
\exp\left(
\log \pi_{\theta_{t+d}}(a_j \mid s_j)
-
\ell_{t,j}
\right).
$$

The stored value $\ell_{t,j}$ must never be refreshed or overwritten.

A stale log-probability is not a bug. It is the denominator of the PPO importance ratio. Staleness must be controlled through edge age and clipping, not by recomputing the denominator.

## 3.2 Remove trainer-side old-log-prob recomputation

Current incorrect flow:

```python
edge_batch = self._generate_edge_batch(gen_batch)
old_log_prob = self.actor_rollout_wg.compute_log_prob(edge_batch)
edge_batch = edge_batch.union(old_log_prob)
actor_output = self.actor_rollout_wg.update_actor(edge_batch)
```

Required flow:

```python
new_edges = self._generate_tree_edges(gen_batch)

assert_every_edge_has_generation_logprobs(new_edges)

self.replay_buffer.add(
    new_edges,
    generation_step=self.global_steps,
    policy_snapshot_id=current_policy_snapshot_id,
)

sampled_edges, sample_stats = self.replay_buffer.sample_for_update(
    current_step=self.global_steps,
)

edge_batch = edges_to_dataproto(
    sampled_edges,
    self.tokenizer,
    max_prompt_length=self.config.data.max_prompt_length,
    max_response_length=self.config.data.max_response_length,
    include_old_log_probs=True,
)

assert "old_log_probs" in edge_batch.batch
actor_output = self.actor_rollout_wg.update_actor(edge_batch)
```

Do not call `actor_rollout_wg.compute_log_prob()` to create `old_log_probs` during normal training.

The actor will still perform its normal forward pass inside `update_policy()` to compute current log-probabilities and gradients. Only the extra pre-update behavior-log-prob forward pass must be removed.

## 3.3 Validate generation log-probabilities once, offline

Add a standalone parity test or diagnostic script.

For a frozen checkpoint and fixed token sequences, compare:

$$
\Delta_{\max}
=
\max_j
\left|
\ell_j^{\text{vLLM}}
-
\ell_j^{\text{actor}}
\right|
$$

and:

$$
\Delta_{\text{mean}}
=
\frac{1}{L}
\sum_{j=1}^{L}
\left|
\ell_j^{\text{vLLM}}
-
\ell_j^{\text{actor}}
\right|.
$$

This is a validation test only. Do not run this extra actor forward pass in every optimizer iteration.

The diagnostic must verify:

- identical model weights;
- identical tokenizer;
- identical prompt and response token IDs;
- identical temperature convention;
- one log-probability per valid response token;
- correct BOS, EOS, and shift alignment.

Add configurable tolerances appropriate for the selected precision. Log the actual maximum and mean error rather than hiding failures with a large tolerance.

## 3.4 Prevent duplicate-key union bugs

`edges_to_dataproto(..., include_old_log_probs=True)` already creates `batch["old_log_probs"]`.

Do not union another `DataProto` containing the same key.

Add an explicit assertion before update:

```python
assert "old_log_probs" in edge_batch.batch
assert edge_batch.batch["old_log_probs"].shape == edge_batch.batch["responses"].shape
```

---

# 4. Trainer-level replay buffer

## 4.1 Location and ownership

Create a trainer-level replay buffer under:

```text
verl/recipe/gear_tree/replay_buffer.py
```

The buffer must be owned by `RayGearTreeTrainer`, not by individual tree agents, rollout workers, queue managers, or prompts.

It must be shared by:

- SPO-tree;
- VDRA-SPO-tree;
- TreePO-style;
- VDRA-TreePO-style;
- TreeRL-style;
- VDRA-TreeRL-style;
- any other tree-family method using the same edge trainer.

The replay implementation and sampling protocol must be identical across these methods.

## 4.2 Edge record

Each stored edge must preserve at least:

```python
{
    "edge_id": str,
    "question_id": str,
    "generation_step": int,
    "policy_snapshot_id": str,

    "query_token_ids": list[int],
    "response_token_ids": list[int],
    "actor_shifted_log_probs": list[float],

    "advantage": float,
    "value": float,
    "reward": float,

    "depth": int,
    "leaf": bool,
    "pruned": bool,
    "tree_update_mode": str,

    "tree_update_local_advantage": float,
    "tree_update_global_advantage": float,
    "tree_update_parent_reward": float,
    "tree_update_child_reward": float,
    "tree_update_root_reward": float,
}
```

Do not recompute the following fields when an edge is replayed:

```text
old log-probabilities
advantage
value
reward
parent reward
child reward
root reward
tree-update metadata
```

## 4.3 Sampling contract

Add configuration:

```yaml
gear_tree:
  replay_buffer:
    enabled: true
    target_edges_per_update: 512
    max_edges_per_question: 32
    max_edge_age: 8
    underfill_policy: use_available
    sampling_seed: 0
    checkpoint: true
```

At optimizer step $t$:

1. Expire all edges satisfying:

   $$
   t - t_{\text{generation}} \geq 8.
   $$

2. Group remaining edges by `question_id`.
3. Select at most 32 candidate edges from each question.
4. From the candidate pool, select at most 512 edges globally.
5. Remove only the selected edge IDs.
6. Keep every unselected edge in the buffer.
7. Never retrieve more than 512 and truncate afterward.
8. Never duplicate edges to fill an underfull update.
9. If fewer than 512 eligible edges exist, train on all available edges and log the underfill.

The deterministic RNG seed should depend on the global seed and optimizer step:

```python
step_seed = base_seed + current_step
```

Sort stable identifiers before random selection so results do not depend on Python dictionary ordering.

## 4.4 Sampling fairness

The per-question cap must be applied before the global cap.

Incorrect:

```text
randomly select 512 globally
then remove questions with more than 32
```

Correct:

```text
cap each question at 32
then select globally from the capped candidate pool
```

This prevents one large tree from dominating the actor update.

## 4.5 Removal semantics

This invariant is mandatory:

```text
removed_edge_ids == edge_ids_passed_to_update_actor
```

Do not remove all candidate edges and later truncate the training batch.

Add a unit test with:

```text
1000 eligible edges
512 sampled edges
488 unselected edges
```

Expected result:

```text
exactly 512 removed
exactly 488 retained
```

## 4.6 Checkpoint behavior

Preferred behavior:

- save replay-buffer state with trainer checkpoint;
- restore exact edge records, RNG state, and buffer metrics.

Acceptable temporary behavior:

- clear the buffer explicitly after resume;
- log `buffer_reset_on_resume = true`;
- never silently pretend that the buffer was restored.

Use an atomic file such as:

```text
<checkpoint_dir>/gear_tree_replay_buffer.jsonl
<checkpoint_dir>/gear_tree_replay_buffer_meta.json
```

or a safe binary serialization with a versioned schema.

Do not store tensors on GPU in the replay buffer. Store CPU-native token IDs and floats.

---

# 5. Required VERL trainer fixes

Modify:

```text
verl/recipe/gear_tree/gear_ray_trainer.py
```

## 5.1 Respect total training steps

The custom loop must stop at:

```python
self.total_training_steps
```

Add:

```python
is_last_step = self.global_steps >= self.total_training_steps
```

and ensure the final step:

- optionally validates;
- saves a checkpoint when configured;
- saves or explicitly resets the replay buffer;
- exits immediately.

Do not rely only on:

```python
for epoch in range(total_epochs):
    for batch in train_dataloader:
```

## 5.2 Required actor metadata

Before `update_actor`, set:

```python
edge_batch.meta_info["global_token_num"] = (
    edge_batch.batch["attention_mask"].sum(dim=-1).tolist()
)
edge_batch.meta_info["multi_turn"] = False
```

Retain any metadata required by the actor worker, including temperature if the worker expects it.

## 5.3 Balance variable-length edge batches

If:

```yaml
trainer.balance_batch: true
```

call:

```python
self._balance_batch(edge_batch, metrics=metrics)
```

before computing `global_token_num`.

The sampled edge count should normally be fixed at 512 when the buffer is full, but response lengths remain variable.

## 5.4 PPO mini-batch divisibility

VERL splits actor data using `ppo_mini_batch_size`.

Add startup validation:

```python
target_edges_per_update % ppo_mini_batch_size == 0
```

unless dynamic batching explicitly supports the selected configuration.

For legacy SPO parity, add a preset using:

```yaml
target_edges_per_update: 512
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 128
    ppo_epochs: 1
```

If an underfilled update is not divisible by the PPO mini-batch size, use one explicit policy:

- postpone the update until enough edges exist; or
- use a smaller valid mini-batch size for that update; or
- use dynamic batching that supports the underfilled size.

Do not pad by duplicating real training edges unless padding is masked out of the loss.

## 5.5 Logging

Every optimizer step must log:

```text
train/num_edges
train/num_response_tokens
train/unique_questions
train/mean_edge_age
train/max_edge_age

buffer/size_before
buffer/new_edges
buffer/expired_edges
buffer/candidate_edges
buffer/sampled_edges
buffer/size_after
buffer/underfilled
buffer/unique_questions
buffer/mean_edge_age
buffer/max_edge_age
buffer/edges_per_question_mean
buffer/edges_per_question_max

buffer/depth_0_edges
buffer/depth_1_edges
buffer/depth_2_edges
```

Also log PPO staleness indicators:

```text
actor/clip_fraction
actor/approx_kl
actor/ratio_mean
actor/ratio_min
actor/ratio_max
```

If these metrics are already generated by the policy loss, forward them to the trainer logger.

---

# 6. Policy-snapshot consistency for VDRA scoring

## 6.1 Required invariant

The policy used for:

1. pilot generation;
2. short-horizon continuation generation;
3. cross-likelihood scoring;
4. stored behavior log-probabilities;

must correspond to the same frozen rollout snapshot for the current optimizer step.

Formally:

$$
\pi_{\text{pilot}}
=
\pi_{\text{support}}
=
\pi_{\text{scorer}}
=
\pi_{\text{behavior}}.
$$

## 6.2 Avoid an unsynchronized external scorer

Current config can point `scorer_api_base` to an independently launched HTTP vLLM server. This is unsafe if the external server is not updated after each actor optimizer step.

Preferred solution:

- expose prompt-log-prob scoring through the same VERL async rollout server manager;
- use the same loaded weights and snapshot lifecycle as segment generation;
- do not maintain a second independent model server.

Alternative solution, only if integration with the same manager is impossible:

- explicitly synchronize scorer weights after every actor update;
- block until synchronization completes;
- expose a scorer snapshot identifier;
- assert equality with the rollout snapshot identifier before generating any tree;
- log both identifiers.

## 6.3 Real policy snapshot identifier

Do not use a question index as `policy_snapshot_id`.

Use a real actor version such as:

```text
global_step
checkpoint hash
monotonic rollout-policy version
```

Propagate it from the trainer through `gen_batch.meta_info` to:

- the agent loop;
- tree builder;
- queue manager;
- replay-buffer edges;
- artifacts and manifests.

Add an invariant:

```python
assert edge["policy_snapshot_id"] == current_rollout_snapshot_id
```

for newly generated edges.

Replayed edges are allowed to have older snapshot IDs.

## 6.4 Scorer model identifier

Do not send an empty `model` field to the OpenAI-compatible scorer API.

Resolve `scorer_model` from the actual served model configuration or query the server model list at startup.

Fail early with a clear message if no valid served-model identifier can be resolved.

---

# 7. Unified integer allocation for pruning and reallocation

## 7.1 Replace the old donor-reserve-receiver formulation

Remove the old two-stage logic:

```text
predicted_k decides donor/receiver
-> donor savings enter a reserve pool
-> dispersion only distributes reserve among receivers
```

The new method must solve one unified bounded integer allocation problem over all nodes in the current allocation queue.

For queue $\mathcal Q$, solve:

$$
\begin{aligned}
\min_{\{k_s\in\mathbb Z_+\}}
&\quad
\sum_{s\in\mathcal Q}\frac{C_s}{k_s}\\
\text{s.t.}
&\quad
\sum_{s\in\mathcal Q}k_s=B_{\mathrm{target}},\\
&\quad
\ell_s\leq k_s\leq u_s,
\qquad \forall s\in\mathcal Q.
\end{aligned}
$$

Here:

```text
C_s
```

is the VDRA value-dispersion proxy for node $s$,

```text
k_s
```

is the final integer branch allocation,

```text
ell_s
```

is the minimum branch allocation, usually `n_min`,

and:

```text
u_s
```

is the node-specific upper bound.

The allocation result itself decides both operations:

$$
k_s<n_s
\quad\Longrightarrow\quad
\text{pruning},
$$

$$
k_s>n_s
\quad\Longrightarrow\quad
\text{expansion},
$$

$$
k_s=n_s
\quad\Longrightarrow\quad
\text{unchanged}.
$$

Do not create donor and receiver sets before solving the optimization problem.

Do not compute a reserve pool as an input to the solver.

After solving, the transferred budget may be reported only as an accounting identity:

$$
\sum_s(n_s-k_s)_+
=
\sum_s(k_s-n_s)_+,
$$

provided the target budget equals the original queue budget.

## 7.2 Bounds and the role of `predicted_k`

Set:

$$
\ell_s=n_{\min}\geq 1.
$$

Use `predicted_k` as a hard redundancy cap only when the predictor finds fewer useful branches than the default branch factor:

$$
u_s=
\begin{cases}
\max(n_{\min},k_s^{\mathrm{pred}}),
&
k_s^{\mathrm{pred}}<n_s,\\
k_{\max,s},
&
k_s^{\mathrm{pred}}\geq n_s.
\end{cases}
$$

Therefore, when:

$$
k_s^{\mathrm{pred}}<n_s,
$$

the unified solver may choose any value in:

$$
n_{\min}
\leq
k_s
\leq
k_s^{\mathrm{pred}}.
$$

It is no longer forced to choose:

$$
k_s=k_s^{\mathrm{pred}}.
$$

For nodes satisfying:

$$
k_s^{\mathrm{pred}}\geq n_s,
$$

do not use `predicted_k` as a mandatory lower bound.

These nodes must remain eligible for both reduction and expansion unless another explicit experimental constraint forbids it.

Recommended default:

```yaml
gear_tree:
  allocation:
    lower_bound: 1
    receiver_upper_bound_mode: configured_max
    max_k_per_node: 12
    predicted_k_cap_mode: below_default_only
```

Alternative upper-bound modes may be implemented as ablations, but they must be explicitly named:

```text
below_default_only
predicted_k_for_all_nodes
configured_max_for_all_nodes
```

The main unified method should use:

```text
below_default_only
```

unless experiments show a clear reason otherwise.

## 7.3 Feasibility checks

Before solving, assert:

$$
\sum_{s\in\mathcal Q}\ell_s
\leq
B_{\mathrm{target}}
\leq
\sum_{s\in\mathcal Q}u_s.
$$

If the lower-bound sum exceeds the requested target, fail with a clear error.

If the upper-bound sum is below the requested target, do not silently drop budget.

Use one explicit configured policy:

```yaml
gear_tree:
  allocation:
    infeasible_upper_policy: expand_nonredundant_caps
```

Recommended behavior:

1. Keep hard caps for nodes with `predicted_k < n_s`.
2. Increase the caps of nonredundant nodes up to a configured global safety maximum.
3. Recheck feasibility.
4. Fail if the queue is still infeasible.

Do not violate a redundancy cap silently.

Do not convert an equality budget into an inequality budget without logging and changing the method definition.

## 7.4 Continuous relaxation for theory and diagnostics

The continuous relaxation remains useful for theory and validation:

$$
\min_{\{k_s\in\mathbb R_+\}}
\sum_s\frac{C_s}{k_s}
$$

subject to the same equality and box constraints.

For an interior node:

$$
-\frac{C_s}{k_s^2}+\lambda=0,
$$

hence:

$$
k_s^\star
=
\sqrt{\frac{C_s}{\lambda}}.
$$

With bounds:

$$
k_s^\star
=
\min\left\{
u_s,
\max\left\{
\ell_s,
\sqrt{\frac{C_s}{\lambda}}
\right\}
\right\}.
$$

The continuous solution may be retained as:

- a theoretical derivation;
- an allocation-shape diagnostic;
- a unit-test oracle for small approximation checks.

Do not use relax-and-round as the main implementation.

Remove the main-path dependency on:

```text
continuous binary search
largest-remainder rounding
stochastic rounding
rounding repair
```

These may remain only behind a clearly named ablation or legacy compatibility mode.

## 7.5 Exact greedy marginal integer solver

Implement an exact integer solver using diminishing marginal objective reduction.

If node $s$ currently has $k$ branches, the decrease in objective from assigning one additional branch is:

$$
\Delta_s(k)
=
\frac{C_s}{k}
-
\frac{C_s}{k+1}
=
\frac{C_s}{k(k+1)}.
$$

Because:

$$
\Delta_s(k+1)\leq\Delta_s(k),
$$

the problem is a bounded separable discrete-convex resource-allocation problem.

Assigning each remaining unit to the currently largest feasible marginal decrease yields a globally optimal integer allocation for this objective.

Create or replace the allocation function with behavior equivalent to:

```python
import heapq

def allocate_branch_factors_integer(
    nodes,
    *,
    total_budget,
    n_min=1,
    max_k_per_node,
    predicted_k_cap_mode="below_default_only",
    strict=True,
):
    lower = {}
    upper = {}
    dispersion = {}
    default = {}

    for node in nodes:
        key = resolve_node_id(node)
        n_s = resolve_default_k(node)
        predicted_k = resolve_predicted_k(node)
        c_s = resolve_nonnegative_dispersion(node)

        lower[key] = n_min
        default[key] = n_s
        dispersion[key] = c_s

        if predicted_k_cap_mode == "below_default_only" and predicted_k < n_s:
            upper[key] = max(n_min, predicted_k)
        elif predicted_k_cap_mode == "predicted_k_for_all_nodes":
            upper[key] = max(n_min, predicted_k)
        else:
            upper[key] = max(n_min, max_k_per_node)

    validate_or_repair_feasibility(
        lower=lower,
        upper=upper,
        default=default,
        total_budget=total_budget,
    )

    allocation = dict(lower)
    remaining = total_budget - sum(allocation.values())

    heap = []
    for key in sorted(allocation):
        if allocation[key] < upper[key]:
            k = allocation[key]
            gain = dispersion[key] / (k * (k + 1))
            heapq.heappush(heap, (-gain, key))

    for _ in range(remaining):
        if not heap:
            raise RuntimeError("No feasible capacity for remaining budget")

        neg_gain, key = heapq.heappop(heap)
        allocation[key] += 1

        if allocation[key] < upper[key]:
            k = allocation[key]
            gain = dispersion[key] / (k * (k + 1))
            heapq.heappush(heap, (-gain, key))

    return allocation
```

Use stable node IDs as deterministic tie-breakers.

No RNG is required for the integer solver.

The solver must return exactly:

$$
\sum_s k_s=B_{\mathrm{target}}.
$$

## 7.6 Solver complexity and latency requirement

Let:

$$
R
=
B_{\mathrm{target}}
-
\sum_s\ell_s.
$$

With a binary heap, the complexity is:

$$
O\left(R\log|\mathcal Q|\right).
$$

This should be negligible for queue sizes and branch budgets in the current experiments.

Add microbenchmark logging during development:

```text
allocation/solver_time_ms
allocation/queue_size
allocation/target_budget
allocation/increment_steps
```

Acceptance targets on a normal CPU process:

```text
queue size <= 32
target budget <= 512
median solver time < 1 ms
p99 solver time < 5 ms
```

These are engineering acceptance targets, not paper claims.

The solver must run locally without model calls, Ray round trips, GPU synchronization, or tensor transfers.

## 7.7 Replace old allocation summary fields

Update `AllocationSummary` and manifests.

Required fields:

```python
{
    "allocations": dict[str, int],
    "dispersion": dict[str, float],
    "default_allocations": dict[str, int],
    "predicted_allocations": dict[str, int],
    "lower_bounds": dict[str, int],
    "upper_bounds": dict[str, int],
    "pruned_allocations": dict[str, int],
    "expanded_allocations": dict[str, int],
    "transferred_budget": int,
    "requested_budget": int,
    "allocated_budget": int,
    "objective_before": float,
    "objective_after": float,
    "solver_time_ms": float,
    "solver_name": str,
}
```

Remove or deprecate fields whose semantics belong only to the old two-stage method:

```text
saved_allocations
unmet_demands
additional_allocations
reserve_contributed
reserve_consumed
reserve_remaining
dual_lambda
rounding_strategy
rounding_seed
raw_allocations
```

Legacy fields may be read for old checkpoint compatibility, but new runs must not emit them as primary method outputs.

Recommended new solver label:

```text
bounded_marginal_integer
```

## 7.8 Objective accounting

Log the objective before and after allocation.

Use the default allocation clipped to feasibility as the comparison point:

$$
J_{\mathrm{default}}
=
\sum_s\frac{C_s}{\bar n_s},
$$

where $\bar n_s$ is the feasible reference allocation used for comparison.

The optimized objective is:

$$
J_{\mathrm{optimized}}
=
\sum_s\frac{C_s}{k_s^\star}.
$$

Log:

```text
allocation/objective_default
allocation/objective_optimized
allocation/objective_reduction
allocation/objective_reduction_ratio
allocation/num_pruned_nodes
allocation/num_expanded_nodes
allocation/num_unchanged_nodes
allocation/transferred_budget
allocation/cap_active_count
allocation/floor_active_count
allocation/feasibility_repair_count
```

Numerically assert:

$$
J_{\mathrm{optimized}}
\leq
J_{\mathrm{default}}+\varepsilon
$$

whenever the default reference allocation is feasible under the same bounds and budget.

## 7.9 Pilot reuse after unified allocation

`allocated_k` is now the final output of the unified integer solver.

All pilot types consume slots from this final allocation:

- terminal shortcut pilots;
- reusable nonterminal pilots;
- newly generated main-expansion children.

Required selection flow:

```python
selected_shortcuts = deterministic_uniform_select(
    shortcut_pilots,
    at_most=allocated_k,
)

remaining = allocated_k - len(selected_shortcuts)

selected_reusable = deterministic_uniform_select(
    reusable_nonterminal_pilots,
    at_most=remaining,
)

remaining -= len(selected_reusable)

additional = generate_fresh_children(
    count=remaining,
)

children = selected_shortcuts + selected_reusable + additional
```

Do not return all shortcuts when:

```text
len(shortcuts) > allocated_k
```

When pilot count exceeds the final allocation, pilot retention must not depend on reward.

Use deterministic seeded uniform selection or deterministic generation-order selection after a seeded shuffle.

## 7.10 Branch and compute accounting invariants

Add strict assertions:

```python
len(children) <= allocated_k
num_selected_shortcuts + num_selected_reusable + num_additional == len(children)
num_selected_shortcuts <= num_generated_shortcuts
num_selected_reusable <= num_reusable_nonterminal_pilots
sum(final_allocations.values()) == target_budget
```

Usually, when generation succeeds:

$$
\#\text{final children}(s)=k_s^\star.
$$

Generated pilot cost must count every pilot, including discarded pilots:

```text
pilot_generated = all pilots generated
pilot_selected = pilots retained as children
pilot_discarded = pilot_generated - pilot_selected
```

Do not confuse generated compute with final branch allocation.

## 7.11 Code locations

Modify at least:

```text
vdra_core/core.py
verl/recipe/gear_tree/
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/config/
```

In `vdra_core/core.py`:

1. Replace the main use of `_continuous_capped_allocation`.
2. Replace `round_bounded` in the default allocation path.
3. Add the exact marginal integer solver.
4. Add unified bounds and feasibility validation.
5. Update `AllocationSummary`.
6. Preserve a legacy continuous solver only under an explicit compatibility mode if required by old tests.

Do not leave the old solver as the silent default.

# 8. Direct short-horizon VDRA mode

## 8.1 Separate strictness from tail calibration

`strict_vdra = true` should mean:

- no silent fallback;
- scorer is available;
- queue invariants hold;
- support requirements are satisfied;
- accounting invariants hold;
- configuration is valid.

It must not automatically mean that a tail-calibration artifact is required.

Add:

```yaml
gear_tree:
  gear:
    tail_mode: none
```

Supported values:

```text
none
calibrated
fixed
```

Semantics:

### `tail_mode: none`

```yaml
eps_tail: 0.0
eps_tail_calibration_path: null
```

Use the observed short-horizon divergence directly as a relative allocation proxy.

The manifest must state:

```text
tail_mode: none
eps_tail: 0.0
score_interpretation: relative short-horizon proxy
certified_full_horizon_bound: false
```

### `tail_mode: calibrated`

Require a compatible calibration artifact and load depth-specific or global tail values.

### `tail_mode: fixed`

Use the configured numeric `eps_tail` without a calibration artifact. This should normally be an ablation, not the default main method.

## 8.2 Launcher changes

Do not require `EPS_TAIL_CALIBRATION_PATH` for every VDRA run.

Required behavior:

```bash
TAIL_MODE=none
```

runs without a calibration file.

Only:

```bash
TAIL_MODE=calibrated
```

must require:

```bash
EPS_TAIL_CALIBRATION_PATH=...
```

## 8.3 Documentation

The main method documentation should describe short-horizon divergence as a lightweight relative proxy for value dispersion.

Do not claim that, under `tail_mode: none`, it is a certified estimate or upper bound of full-trajectory value divergence.

---

# 9. Experimental protocol presets

Add explicit presets rather than relying on implicit defaults.

## 9.1 Legacy SPO-parity preset

Create a config overlay such as:

```text
verl/recipe/gear_tree/config/legacy_spo_parity.yaml
```

Set:

```yaml
gear_tree:
  tree_shape: [6, 6, 6]
  segment_length: 600
  replay_buffer:
    target_edges_per_update: 512
    max_edges_per_question: 32
    max_edge_age: 8

actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 128
    ppo_epochs: 1
```

Also match:

- model;
- tokenizer;
- prompt format;
- response maximum length;
- sampling temperature;
- top-p;
- dataset order;
- number of questions generated per optimizer iteration;
- optimizer;
- learning rate;
- evaluation protocol.

## 9.2 Reduced VERL preset

A faster controlled setting may use:

```yaml
gear_tree:
  segment_length: 100
```

but it must be named and reported as a reduced-segment setting.

Do not mix its result with the legacy $M=600$ setting as if they were the same protocol.

## 9.3 Fixed iteration budget

The launcher must support an explicit optimizer-step budget:

```bash
trainer.total_training_steps=150
```

and the custom trainer must stop exactly after that many optimizer updates.

---

# 10. `fixed_total_generated` naming and accounting

The current analytical cap assumes a complete uniform tree and full token cap at every branch.

This is a maximum-style reference budget, not the realized token cost of an actual SPO run with early EOS.

Rename or document it clearly as:

```text
fixed_uniform_max_generated_tokens
```

or:

```text
uniform_full_tree_token_cap
```

Do not claim equality with actual SPO generated tokens unless the cap is obtained from a paired SPO rollout log.

Continue reporting separately:

```text
pilot decode tokens
pilot-support decode tokens
main-expansion decode tokens
proxy-rollout tokens
likelihood-scored prompt tokens
likelihood-scored continuation tokens
```

---

# 11. Baseline naming

Until official parity is established, use internal names such as:

```text
spo
treepo_style
treerl_style
vdra_spo
vdra_treepo_style
vdra_treerl_style
```

Do not present the current objective-only variants as exact official TreePO or TreeRL reproductions.

Exact naming requires parity in:

- tree construction;
- branching policy;
- rollout scheduling;
- node-value definition;
- advantage and return computation;
- policy loss;
- data reuse;
- optimizer protocol;
- official hyperparameters.

This naming change is documentation and experiment-label work. It must not change the numerical objective unless an official implementation is later ported.

---

# 12. Tests

## 12.1 Replay-buffer unit tests

Create:

```text
verl/recipe/gear_tree/tests/test_replay_buffer.py
```

Required tests:

1. Per-question cap:
   - 100 edges from one question;
   - selected count is at most 32.

2. Global cap:
   - more than 512 eligible candidates;
   - selected count is exactly 512.

3. Underfill:
   - only 300 eligible edges;
   - select exactly 300;
   - no duplication.

4. Unused-edge retention:
   - 1000 eligible edges;
   - sample 512;
   - retain exactly 488.

5. Removal correctness:
   - removed IDs exactly equal trained IDs.

6. Age expiration:
   - edge age 7 survives;
   - edge age 8 expires.

7. Determinism:
   - same seed and step produce identical selected IDs.

8. Different steps:
   - a different step seed may produce a different valid sample.

9. Stored old log-probs:
   - serialization and sampling preserve values exactly.

10. Stored advantages:
    - replay does not recompute or alter advantages.

11. Checkpoint round trip:
    - save and restore produce identical buffer contents and RNG state.

12. Method independence:
    - identical synthetic edge sets from SPO and VDRA pass through the same sampler implementation.

## 12.2 Trainer tests

Add tests for:

- `global_token_num` is set before actor update;
- `multi_turn` is set;
- `total_training_steps` stops the loop exactly;
- no call to `compute_log_prob()` occurs in normal replay training;
- `old_log_probs` exists before update;
- no duplicate-key union is attempted;
- `_balance_batch()` is called when enabled;
- replay buffer is saved or explicitly reset on resume.

Use fake actor and rollout worker groups to count method calls.

## 12.3 Unified allocation solver tests

Create or update:

```text
vdra_core/tests/test_integer_allocation.py
```

Required tests:

1. Exact budget:
   - verify:

     $$
     \sum_s k_s=B_{\mathrm{target}}.
     $$

2. Lower bounds:
   - every allocation satisfies:

     $$
     k_s\geq\ell_s.
     $$

3. Upper bounds:
   - every allocation satisfies:

     $$
     k_s\leq u_s.
     $$

4. Redundancy cap:
   - for `default_k = 6`, `predicted_k = 3`;
   - verify `allocated_k <= 3`.

5. Unified pruning:
   - a capped low-dispersion node may receive fewer than `predicted_k`.

6. Unified expansion:
   - an uncapped high-dispersion node may receive more than `default_k`.

7. No preassigned donor/receiver sets:
   - an uncapped node is allowed to end below `default_k`;
   - another uncapped node is allowed to end above `default_k`.

8. Zero dispersion:
   - zero-score nodes receive remaining slots only after all positive marginal gains are exhausted or capped.

9. Tie determinism:
   - identical gains are resolved by stable node ID.

10. Infeasible lower sum:
    - hard error when:

      $$
      B_{\mathrm{target}}<\sum_s\ell_s.
      $$

11. Infeasible upper sum:
    - configured repair or hard error when:

      $$
      B_{\mathrm{target}}>\sum_su_s.
      $$

12. Brute-force optimality:
    - for small random problems, enumerate every feasible integer allocation;
    - verify greedy objective equals the global minimum.

13. Marginal monotonicity:
    - verify:

      $$
      \Delta_s(k+1)\leq\Delta_s(k).
      $$

14. Objective non-increase:
    - whenever the reference default allocation is feasible, verify:

      $$
      J_{\mathrm{optimized}}
      \leq
      J_{\mathrm{default}}+\varepsilon.
      $$

15. Continuous-relaxation sanity:
    - for uncapped interior cases, integer allocations should follow the ranking implied by:

      $$
      k_s\propto\sqrt{C_s}.
      $$

16. Latency:
    - benchmark queue size 32 and budget 512;
    - median below 1 ms and p99 below 5 ms on a normal CPU test environment;
    - keep the timing test non-flaky by reporting results and using a conservative CI threshold.

## 12.4 Pilot-budget tests

Add cases:

1. `allocated_k = 4`, shortcuts = 5:
   - final shortcuts selected = 4;
   - final children = 4.

2. `allocated_k = 5`, shortcuts = 2, reusable = 2:
   - generate 1 additional child;
   - final children = 5.

3. `allocated_k = 3`, shortcuts = 1, reusable = 8:
   - select 2 reusable;
   - generate 0 additional;
   - final children = 3.

4. Selection is deterministic under fixed seed.

5. Selection does not depend on shortcut reward.

6. All generated pilots remain counted in compute metrics even when discarded.

7. Final child counts match the unified solver output.

8. Sum of per-node final allocations equals the queue target budget.

## 12.5 Tail-mode tests

Test:

```text
strict_vdra=true, tail_mode=none, eps_tail=0
```

must start without a calibration artifact.

Test:

```text
strict_vdra=true, tail_mode=calibrated
```

must reject a missing artifact.

Test:

```text
tail_mode=calibrated
```

must reject incompatible metadata.

Test manifest fields for all modes.

## 12.6 Policy-snapshot tests

Use fake snapshot IDs to verify:

- pilot generator and scorer receive the same current snapshot ID;
- newly generated edges store that snapshot ID;
- replayed older edges preserve their old snapshot ID;
- a mismatch between rollout and scorer snapshot causes a hard error.

---

# 13. End-to-end smoke test

After all CPU tests pass, run a real two-to-five-step training job with:

```text
Ray
FSDP
async vLLM rollout
tree agent
current-policy scorer
replay buffer
actor update
checkpoint save
```

Use a small model and small prompt set.

The smoke test passes only if:

1. At least two actor optimizer updates complete.
2. Actor parameters change after each successful update.
3. No NaN or Inf appears in:
   - loss;
   - gradient norm;
   - PPO ratio;
   - KL;
   - advantages;
   - old log-probabilities.

4. Replay buffer:
   - receives new edges;
   - samples edges;
   - retains unused edges;
   - expires old edges when configured.

5. Checkpoint save and resume work.
6. The replay buffer is restored or explicitly reset and logged.
7. `queue_size_at_flush` is logged.
8. At least one multi-node flush occurs under a suitable test config.
9. The scorer snapshot equals the rollout snapshot.
10. Validation can run at least once.
11. `trainer.total_training_steps` stops at the exact requested step.

Save a compact smoke-test report containing:

```text
commit SHA
resolved Hydra config
model
dataset subset
GPU count
VERL version
vLLM version
number of optimizer steps
number of generated edges
number of trained edges
buffer statistics
queue statistics
policy snapshot IDs
final loss and gradient norm
```

---

# 14. Acceptance criteria

The task is complete only when all of the following are true:

## Algorithmic invariants

```text
the default allocation path uses the exact bounded marginal integer solver
no donor/receiver partition is created before optimization
predicted_k is a hard cap for below-default redundancy predictions
the integer solver preserves the exact feasible target budget
all allocations satisfy their lower and upper bounds
greedy allocations match brute-force optima on small test problems
old log-probs are captured at generation and never refreshed
advantages are captured at tree construction and never refreshed
final children <= allocated_k
all pilot generation cost is accounted for
only trained replay edges are removed
edge age >= 8 expires
per-question selected edges <= 32
global selected edges <= 512
```

## Runtime invariants

```text
allocation requires no model call, GPU synchronization, or Ray round trip
allocation median latency is below 1 ms for queue size <= 32 and budget <= 512
allocation p99 latency is below 5 ms for queue size <= 32 and budget <= 512
global_token_num is present before update_actor
total_training_steps is respected
no duplicate old_log_probs union occurs
scorer and rollout use the same policy snapshot
queues and futures are empty/completed before tree return
```

## Experimental invariants

```text
pruning and expansion are outputs of the same optimization problem
relax-and-round is not the default solver
old reserve-pool fields are not emitted as primary new-run outputs
SPO and VDRA use the same replay-buffer implementation
SPO and VDRA use the same edge caps
SPO and VDRA use the same actor mini-batch protocol
direct eps_tail=0 runs without a fake calibration artifact
M=100 and M=600 results are labeled as different settings
TreePO-style and TreeRL-style are not mislabeled as exact official reproductions
```

## Verification

```text
all unit tests pass
small-instance brute-force allocation checks pass
allocation microbenchmark passes
offline rollout-vs-actor log-prob diagnostic passes
two-to-five-step Ray+FSDP+async-vLLM smoke test passes
checkpoint resume behavior is verified
```

# 15. Suggested implementation order

Implement in this exact order:

1. Add the unified bound construction for $\ell_s$ and $u_s$.
2. Implement the exact marginal integer solver.
3. Add brute-force optimality tests for small instances.
4. Replace the old continuous-plus-rounding default path in `vdra_core/core.py`.
5. Update `AllocationSummary`, manifests, and allocation logging.
6. Add feasibility repair for insufficient nonredundant upper capacity.
7. Enforce `final children <= allocated_k`, including shortcut pilots.
8. Add solver latency microbenchmarks.
9. Add `tail_mode` and remove the unconditional calibration requirement.
10. Add replay-buffer data structures and unit tests.
11. Refactor tree generation to return raw edge records before `DataProto` conversion.
12. Store generation-time log-probabilities in replay edges.
13. Remove trainer-side behavior-log-prob recomputation.
14. Integrate replay sampling into `RayGearTreeTrainer`.
15. Add `global_token_num`, `multi_turn`, balancing, and exact step stopping.
16. Add replay checkpointing and logging.
17. Bind the scorer to the current rollout policy snapshot.
18. Add parity diagnostics and policy-snapshot tests.
19. Add protocol presets.
20. Run all CPU tests.
21. Run the end-to-end smoke test.
22. Only after the smoke test passes, start SPO-versus-VDRA experiments.

Do not start long training runs before steps 1–21 are complete.
