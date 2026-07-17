# PLAN.md — VDRA Pipeline and Engineering Alignment

## 0. Objective and audited baseline

This plan is based on repository commit:

```text
d137bd94bd700c2358a4394e930049af99bad02d
```

The goal is not only to make the code run. Every reported result must correspond to a clearly defined:

- behavior-policy snapshot;
- sampling and scoring distribution;
- node-value estimator;
- allocation scope and budget;
- PPO denominator;
- optimizer-step count;
- compute accounting protocol.

Status labels:

- **DONE**: implemented and supported by direct tests.
- **PARTIAL**: implementation exists, but the end-to-end invariant is not guaranteed.
- **OPEN**: still inconsistent, ambiguous, or untested in the deployed Ray/FSDP/vLLM stack.

**No long main-paper training run is allowed until every P0 item is complete and the real async smoke test passes.**

---

# 1. Current implementation status

## 1.1 DONE

The following parts are now substantially aligned:

1. Exact bounded integer allocation is the production solver.
2. The solver preserves the exact budget within each allocation flush.
3. True lower and upper solver bounds are written into node accounting.
4. `fresh_iid` exists and discards pilots before final-child generation.
5. Pilot clusters and multiplicity metadata exist for `weighted_reuse`.
6. Exact token-ID likelihood scoring exists.
7. Silent query/response truncation is rejected.
8. Replay uses reservation, commit, and rollback.
9. Replay checkpoint restore/reset is explicit.
10. `global_steps` increases only after a successful actor update.
11. PPO masked means return a finite differentiable zero for empty masks.
12. Edge-weighted PPO loss is implemented at the loss-function level.
13. `tail_mode=none` with `eps_tail=0` is supported.
14. Main configuration uses `bound_form=linear`, not the legacy simulation-lemma mode.
15. TreePO/TreeRL modes are documented as style/parity ablations rather than exact reproductions.

These fixes are necessary but not sufficient. Several pipeline connections remain broken or unverifiable.

---

# 2. P0 blockers before any long run

## P0.1 Propagate the dynamic policy snapshot into the agent loop

### Current risk

The trainer writes the current snapshot to `gen_batch.meta_info`, while the upstream agent-loop worker passes per-sample fields from `non_tensor_batch` into `TreeAgentLoop.run()`.

Consequently, the agent loop may receive no dynamic snapshot and fall back to:

```text
rollout_step:unknown
```

The trainer then expects `global_step:t`, causing either a mismatch error or incorrect provenance.

### Required implementation

Before `generate_sequences`, add per-row fields:

```python
gen_batch.non_tensor_batch["policy_snapshot_id"] = np.array(
    [snapshot_id] * len(gen_batch), dtype=object
)
gen_batch.non_tensor_batch["current_rollout_snapshot_id"] = np.array(
    [snapshot_id] * len(gen_batch), dtype=object
)
```

Keep the `meta_info` copies for logging, but do not rely on them for agent-loop kwargs.

### Acceptance test

Run two real rollout iterations and assert for every edge:

```text
edge.policy_snapshot_id == trainer snapshot at generation
edge.policy_snapshot_id != rollout_step:unknown
```

---

## P0.2 Prove scorer weights equal rollout weights

### Current risk

Generation uses the VERL async rollout server manager, while likelihood scoring uses a separately constructed HTTP client. Equal string snapshot IDs do not prove equal model parameters:

\[
\theta_{\mathrm{scorer}}=\theta_{\mathrm{rollout}}.
\]

The scorer is also built inside each agent-loop instance, which can create one `/models` request, HTTP client, semaphore, and connection pool per prompt.

### Preferred implementation

Add token-logprob scoring to the same async rollout server manager used for generation:

```python
await server_manager.prompt_token_logprobs(
    request_id=...,
    prompt_ids=full_token_ids,
)
```

Generation and scoring must share:

- server pool;
- loaded model replicas;
- weight-update lifecycle;
- tokenizer;
- model revision/version.

### Temporary acceptable implementation

If an external scorer remains:

1. construct one shared scorer per `AgentLoopWorker`, not per prompt;
2. cache resolved model ID;
3. expose a server-side `weight_version` or weight fingerprint;
4. verify it against the rollout server after every actor update;
5. close the HTTP client at worker shutdown;
6. log scorer endpoint, model ID, weight version, and rollout version.

### Acceptance test

A real GPU diagnostic must score identical token sequences with:

- rollout vLLM;
- scorer path;
- FSDP actor.

Log:

\[
\Delta_{\max}=\max_j|\ell_j^{a}-\ell_j^{b}|,
\qquad
\Delta_{\mathrm{mean}}=\frac1L\sum_j|\ell_j^{a}-\ell_j^{b}|.
\]

The test must fail when weight versions differ.

---

## P0.3 Validate the actual sampling distribution

### Current risk

The strict gate may validate configured defaults `temperature=1` and `top_p=1`, while the actual agent-loop generator uses values from `rollout_config`. Using `setdefault` can preserve stale config values instead of the actual runtime values.

### Required implementation

Overwrite, do not default:

```python
gear_cfg["rollout_temperature"] = float(self.rollout_config.temperature)
gear_cfg["rollout_top_p"] = float(self.rollout_config.top_p)
```

Pass both values into every `GearGate` construction path.

For the main tanh-TV estimator, enforce:

```text
actual rollout temperature == 1.0
actual rollout top_p == 1.0
```

until the scorer explicitly implements the same transformed distribution.

### Acceptance test

A run configured with actual `temperature=0.7` must fail strict startup even if the nested VDRA config says `1.0`.

---

## P0.4 Preserve stored behavior log-probabilities in every actor batch

### Current risk

VERL actor code treats a single PPO minibatch with one epoch as on-policy and replaces stored old log-probabilities with current log-probabilities:

```python
old_log_prob = log_prob.detach()
```

This is invalid for replayed tree edges and forces the PPO ratio toward one.

### Required implementation

Set:

```python
edge_batch.meta_info["force_stored_old_log_probs"] = True
```

Actor logic must use current log-probabilities as the denominator only when this flag is false.

The tree/replay trainer must always use stored generation-time values:

\[
r_j(\theta)
=
\exp\left(
\log\pi_{\theta_{t+d}}(a_j\mid s_j)
-
\log\pi_{\theta_t}(a_j\mid s_j)
\right).
\]

### Acceptance test

Use exactly one PPO minibatch where stored old log-probabilities differ from current values. Assert that the computed ratio is not identically one.

---

## P0.5 Complete the `weighted_reuse` tree-to-loss pipeline

### Current risk

Representative weights exist on tree children and the policy loss accepts `edge_weights`, but edge extraction currently does not guarantee that these fields survive:

```text
tree child -> replay edge -> DataProto -> actor worker -> policy loss
```

### Required implementation

Copy these fields into every extracted edge:

```python
"edge_weight": node.get("edge_weight", node.get("vdra_representative_weight")),
"vdra_cluster_id": node.get("vdra_cluster_id"),
"vdra_cluster_multiplicity": node.get("vdra_cluster_multiplicity"),
"vdra_original_pilot_indices": node.get("vdra_original_pilot_indices"),
```

Replay validation and checkpoint serialization must preserve them.

### Acceptance test

Construct a weighted tree with multiplicities `[2,1]` and verify end-to-end that:

1. extracted edges contain weights;
2. `DataProto.batch["edge_weights"]` exists;
3. actor receives it;
4. weighted loss equals explicit edge duplication.

---

## P0.6 Implement actual `weighted_reuse_fallback`

### Current risk

If the final allocation contains fewer slots than the number of required clusters, selecting only some representatives removes entire clusters. Renormalizing the remaining weights cannot recover the missing probability mass.

The config contains:

```yaml
weighted_reuse_fallback: fresh_iid
```

but this must be an executed runtime rule, not only stored configuration.

### Required contract

Use weighted reuse only when every required cluster is represented:

```text
allocated_k >= number_of_required_clusters
```

Otherwise:

- `fresh_iid`: generate all final children freshly and mark fallback;
- `error`: abort the run.

Required fields:

```text
vdra_weighted_reuse_fallback_triggered
vdra_weighted_reuse_fallback_reason
vdra_required_cluster_count
vdra_allocated_k
```

### Acceptance test

Create five clusters with allocation three. `fresh_iid` fallback must produce no representative-weighted final children.

---

## P0.7 Replace connected-component clusters with representative-valid clusters

### Current risk

Connected components only enforce paths of pairwise-near nodes. They do not guarantee that every member is close to the chosen representative.

A cluster may satisfy:

\[
TV(A,B)<\epsilon,
\quad
TV(B,C)<\epsilon,
\quad
TV(A,C)\gg\epsilon.
\]

### Required implementation

Use a deterministic clustering rule satisfying:

\[
\max_{x\in G_r}TV(x,r)\le\epsilon.
\]

Recommended options:

1. star clustering around a deterministic representative;
2. complete-linkage clustering;
3. medoid selection followed by a maximum-distance check.

Do not use reward or correctness for representative selection.

### Acceptance test

Use a non-transitive A-B-C example and assert that A and C cannot share a cluster when their direct TV exceeds the threshold.

---

## P0.8 Correct terminal-pilot contribution to dispersion

### Current risk

Terminal-continuation pairs are conservatively assigned TV one, but terminal-terminal pairs currently contribute zero. This is wrong when terminal rewards differ.

For observed terminal rewards:

\[
B_{ij}^{\mathrm{terminal}}=|R_i-R_j|.
\]

### Required implementation

Grade terminal pilots before allocation and compute:

\[
C_s
=
\frac{1}{n^2}
\left(
\sum_{\mathrm{cont-cont}}B_{ij}^2
+
\sum_{\mathrm{terminal-cont}}B_{ij}^2
+
\sum_{\mathrm{terminal-terminal}}(R_i-R_j)^2
\right).
\]

Keep terminal-continuation as a documented conservative bound if no sharper estimate is available.

### Acceptance test

Two terminal pilots with rewards `[0,1]` must yield:

```text
C_terminal > 0
C_total > 0
```

---

## P0.9 Disable the unused critic path

### Current risk

`algorithm.adv_estimator=gae` causes VERL to create a critic worker, while the custom trainer uses precomputed tree advantages and never trains the critic.

This wastes memory and makes `critic_warmup` semantics misleading.

### Required implementation

Set explicitly:

```yaml
critic:
  enable: false
```

Use or introduce an estimator/config path that does not imply a critic. Remove critic warmup from the custom actor-only loop.

### Acceptance test

Startup logs and Ray roles must show no critic worker for SPO-tree and VDRA runs.

---

# 3. Canonical pilot execution modes

## 3.1 `fresh_iid` — main-paper default

Pilots and support blocks estimate redundancy and the allocation coefficient only. After solving for `allocated_k`, generate exactly that many new final children:

\[
x_{s,1},\ldots,x_{s,k_s^*}
\overset{iid}{\sim}
\pi_{\theta_t}(\cdot\mid s).
\]

Parent value:

\[
\hat V_s^{\mathrm{fresh}}
=
\frac1{k_s^*}
\sum_{j=1}^{k_s^*}\hat V(x_{s,j}).
\]

Required invariants:

```text
pilot_children_reused == 0
terminal_shortcuts_reused == 0
final_children == allocated_k, unless an explicit token cap is hit
all final children carry the same generation snapshot
```

This is the only mode that may directly use the iid Monte Carlo interpretation in the main theory.

## 3.2 `weighted_reuse` — approximate efficiency variant

For representative `r` of cluster `G_r`:

\[
m_r=|G_r|.
\]

With fresh extra children `F_s`, use:

\[
\hat V_s^{\mathrm{reuse}}
=
\frac{
\sum_{r\in\mathcal R_s}m_r\hat V(r)
+
\sum_{f\in\mathcal F_s}\hat V(f)
}{
\sum_r m_r+|\mathcal F_s|
}.
\]

Required invariants:

```text
every pilot belongs to exactly one cluster
every retained cluster has exactly one representative
sum(cluster multiplicities) == number of represented pilot draws
all required clusters are covered, otherwise fallback
parent aggregation and PPO loss use the same weights
```

This mode must be reported as a compute-efficient representative/coreset variant. It must not inherit the iid estimator claim.

---

# 4. P1 engineering hardening

## P1.1 Correct allocation-scope terminology

Runtime allocation is solved separately for each queue flush, not for the complete depth frontier or whole tree.

Use:

```text
allocation_scope: per_queue_flush_within_tree
```

The method statement should define `Q` as the set of nodes in one flush. Queue count, capacity, timeout, and flush-size histograms must be reported.

## P1.2 Remove reserve terminology from primary reporting

The exact bounded solver redistributes budget inside a flush. The old reserve pool is no longer the optimization mechanism.

Primary fields:

```text
pruned_k
expanded_k
transferred_budget_within_flush
lower_bound_k
upper_bound_k
objective_before
objective_after
```

Keep `gear_reserve_*` only as deprecated compatibility aliases.

## P1.3 Add context-length startup validation

Strict no-truncation can fail during training when accumulated parent trajectories exceed `max_prompt_length`.

Before training, compute or conservatively bound:

\[
L_{\max}^{\mathrm{edge-query}}
\le
L_{\max}^{\mathrm{original-prompt}}
+(d-1)M.
\]

Validate it against configured prompt length and log dataset statistics.

## P1.4 Make `replay_buffer.enabled` real or remove it

If `enabled=false` is supported, implement a direct non-replay update path. Otherwise remove the field so an ablation cannot silently do nothing.

## P1.5 Use deterministic edge IDs

Replace random UUID edge IDs with a stable hash of:

```text
policy_snapshot_id
stable_question_id
tree_id
parent_path
child_index
```

This is required for reproducible replay sampling.

## P1.6 Require stable global question IDs

Do not fall back to a per-batch index. Require a dataset UID or hash of the normalized problem. The per-question replay cap must never combine different questions that happen to share a batch-local index.

## P1.7 Avoid placeholder reward computation

Tree construction already grades the actual leaves. The placeholder response returned only to satisfy `AgentLoopOutput` should not trigger another reward-manager call.

Add a tree-output flag or explicit sentinel reward to skip this redundant computation.

## P1.8 Preserve sticky sessions for pilot completion

Do not create unrelated random request IDs for every pilot and continuation. Use stable session IDs derived from tree/node/branch identity so continuation requests remain on the same server and can reuse prefix cache.

This also reduces latency-driven variation in queue composition.

## P1.9 Fix or quarantine legacy simulation-lemma mode

The main run must remain:

```yaml
tail_mode: none
bound_form: linear
```

Do not claim a certified tight simulation-lemma bound for this mode. Either correct and separately test the legacy discounted formula or mark it unsupported for paper experiments.

---

# 5. Required tests

## 5.1 CPU/unit tests

1. Exact allocator vs exhaustive brute force on many random small instances.
2. True lower/upper bound persistence.
3. Replay reservation–commit–rollback.
4. Replay resume with exact edge records.
5. One-minibatch stored-old-logprob preservation.
6. Dynamic snapshot propagation through non-tensor kwargs.
7. Exact token-ID scorer boundary alignment.
8. Actual rollout temperature/top-p validation.
9. `fresh_iid` never reuses pilots.
10. Weighted tree-to-edge-to-loss propagation.
11. Weighted fallback when cluster coverage is impossible.
12. Non-transitive clustering counterexample.
13. Terminal rewards `[0,1]` produce positive terminal dispersion.
14. Empty PPO mask produces finite zero and backward succeeds.
15. Stable question and edge IDs are deterministic.

## 5.2 Real integration tests

Run with real Ray, FSDP, and async vLLM:

### Smoke A — SPO-tree

- 2–5 successful actor updates;
- no critic worker;
- finite loss and gradients;
- stored old log-probabilities preserved.

### Smoke B — VDRA `fresh_iid`

- 2–5 successful actor updates;
- scorer and rollout report the same weight version;
- exact token scorer is used;
- no pilot is reused;
- final-child count matches allocation;
- queue flushes contain more than one node at least once.

### Smoke C — VDRA `weighted_reuse`

- 2–5 successful actor updates;
- all cluster coverage invariants pass;
- edge weights reach actor loss;
- fallback is exercised at least once in a controlled test;
- no NaN/Inf in weighted value, advantage, ratio, loss, or gradient.

---

# 6. Required runtime logging

Every run manifest must contain:

```text
commit_sha
algorithm_requested
algorithm_executed
pilot_execution_mode
weighted_reuse_fallback
allocation_scope
allocation_proxy
bound_form
tail_mode
eps_tail
tree_shape
segment_length
pilot_branch_factor
likelihood_samples_per_distribution
queue_count
queue_capacity
queue_timeout_seconds
actual_rollout_temperature
actual_rollout_top_p
policy_snapshot_id
rollout_weight_version
scorer_weight_version
scorer_model
critic_enabled
budget_mode
run_valid_for_main_results
```

Every node/allocation record must contain:

```text
default_k
predicted_k
allocated_k
lower_bound_k
upper_bound_k
pruned_k
expanded_k
transferred_budget_within_flush
C_continuation
C_terminal
C_cross
C_total
objective_before
objective_after
solver_time_ms
queue_id
queue_size_at_flush
flush_reason
queue_wait_seconds
pilot_execution_mode
pilot_generated
pilot_reused
pilot_discarded
cluster_count
fallback_triggered
final_child_count
```

Training logs must separate:

```text
rollout_iteration
optimizer_step
successful_actor_updates
postponed_updates
failed_updates
replay_buffer_size
mean_edge_age
max_edge_age
```

---

# 7. Paper-safe claims

For the current main configuration, use language equivalent to:

> VDRA estimates a short-horizon node-wise dispersion proxy from conditional policy divergence and solves an exact bounded integer allocation problem within each online queue flush.

Do not claim:

- full-frontier global optimality;
- full-horizon certified value bounds when `tail_mode=none`;
- exact TreePO or TreeRL reproduction;
- unbiased Monte Carlo estimation for `weighted_reuse`;
- equal scorer and rollout policies based only on equal string IDs;
- equal compute based only on the full-tree maximum token cap.

`fresh_iid` is the primary theory-aligned method. `weighted_reuse` is a separately named efficiency variant.

---

# 8. Implementation order

Execute in this order:

1. Dynamic snapshot propagation through `non_tensor_batch`.
2. Force stored old log-probabilities in actor updates.
3. Shared scorer lifecycle and verifiable weight-version equality.
4. Actual temperature/top-p propagation and strict validation.
5. Tree-to-edge representative-weight propagation.
6. Weighted-reuse coverage fallback.
7. Representative-valid clustering.
8. Terminal-terminal dispersion from observed rewards.
9. Disable critic and remove critic warmup semantics.
10. Context-length and stable-ID validation.
11. Scope/terminology/logging cleanup.
12. CPU test suite.
13. Real SPO, fresh-iid, and weighted-reuse smoke tests.
14. Only then start long comparison runs.

---

# 9. Go / no-go gate

A long run is **GO** only when all conditions are true:

```text
[ ] Dynamic snapshot reaches every agent-loop sample.
[ ] Scorer and rollout weights are verifiably identical.
[ ] Actual sampling distribution is validated.
[ ] Stored behavior log-probabilities are never overwritten.
[ ] fresh_iid passes exact final-child invariants.
[ ] weighted_reuse weights reach the actor loss.
[ ] weighted_reuse fallback handles missing cluster coverage.
[ ] Cluster membership is representative-valid.
[ ] Terminal reward disagreement contributes to C_s.
[ ] No unused critic worker is created.
[ ] Stable global question IDs and deterministic edge IDs are used.
[ ] Context-length validation passes.
[ ] Real Ray/FSDP/async-vLLM smoke tests pass for all three modes.
[ ] No NaN/Inf appears in value, advantage, PPO ratio, loss, or gradients.
[ ] Manifests and artifacts identify the exact executed estimator and scope.
```

Until this gate passes, results are engineering diagnostics rather than main-paper evidence.
