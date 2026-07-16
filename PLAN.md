# PLAN.md — Theory–Runtime Alignment for VDRA

## 0. Purpose and audited baseline

This plan replaces the previous migration checklist. It is based on repository state `de7d1ee32a2206505c0c1d772dfa1186cc27b2c2` and the follow-up audit performed after that commit.

The objective is not merely to make training run. Every main-paper result must correspond to a precisely defined estimator, sampling distribution, behavior policy, allocation problem, compute budget, and optimizer-update count.

Status labels:

- **DONE**: implemented and directly supported by code/tests.
- **PARTIAL**: some plumbing or tests exist, but the scientific invariant is not yet guaranteed.
- **OPEN**: not implemented or still inconsistent with the theory.

No long main-paper run is allowed until all P0 items are complete and the end-to-end smoke test passes.

---

# 1. Audited status of the current code

## 1.1 Completed fixes

### DONE — Exact unified bounded integer allocation

The production allocator solves

\[
\min_{k_s\in\mathbb Z_+}\sum_{s\in\mathcal Q}\frac{C_s}{k_s}
\]

subject to

\[
\sum_{s\in\mathcal Q}k_s=B_{\mathrm{target}},\qquad
\ell_s\le k_s\le u_s.
\]

The default path uses exact marginal gains

\[
\Delta_s(k)=\frac{C_s}{k(k+1)}
\]

with deterministic stable-ID tie breaking and exact budget preservation. Continuous relax-and-round is not the main runtime path.

### DONE — Generation-time behavior log-probabilities are retained

Normal training no longer recomputes `old_log_probs` immediately before actor update. Tree edges retain rollout-time token log-probabilities and replay them as the PPO denominator.

### DONE — Replay sampling fairness and delayed removal

The trainer-owned replay buffer applies the per-question cap before the global cap. The trainer samples with `remove=False` and removes edges only after `update_actor` succeeds. Actor failure therefore leaves sampled edges available.

### DONE — Replay checkpoint restore/reset is explicit

The trainer restores the replay buffer when its checkpoint exists. A missing replay checkpoint on resume causes an explicit reset metric rather than silently pretending the buffer was restored.

### DONE — Tail modes

`tail_mode=none` permits strict short-horizon operation with `eps_tail=0`. `calibrated` requires an artifact. `fixed` is an explicit numerical ablation.

### DONE — Snapshot ID propagation through trainer, agent loop, tree, and edges

A policy snapshot identifier is now propagated through the rollout metadata and stored on generated edges.

### DONE — Solver microbenchmark scaffolding

A CPU benchmark reports median and p99 allocation latency for normal queue sizes and budgets.

### DONE — TreePO/TreeRL naming disclaimer

The documentation now treats the corresponding objectives as style/parity ablations rather than verified official reproductions.

## 1.2 Partially fixed items

### PARTIAL — Policy snapshot consistency

The code checks equality between `rollout_snapshot_id` and `scorer_snapshot_id`, but both can be labels attached to independently served models. Equal strings do not prove equal weights.

Required invariant:

\[
\theta_{\mathrm{pilot}}
=
\theta_{\mathrm{support}}
=
\theta_{\mathrm{scorer}}
=
\theta_{\mathrm{behavior}}.
\]

The scorer must either use the same VERL async server manager/model instance as rollout generation or expose a verifiable model version/weight fingerprint synchronized after each actor update.

### PARTIAL — Rollout-vs-actor log-prob parity

Comparison helpers and unit tests exist, but the repository still needs a real diagnostic that produces records from the actual vLLM rollout server and FSDP actor for identical token sequences.

### PARTIAL — Underfilled replay updates

Indivisible underfilled batches are postponed, but `global_steps` still advances on postponed/no-edge iterations. Thus `trainer.total_training_steps` still counts loop iterations rather than successful optimizer updates.

### PARTIAL — Replay transaction semantics

Removal now happens after successful actor update, which is correct. However, sampled IDs should be represented by an explicit reservation/commit/rollback API to prevent concurrent or future callers from training the same reserved edges twice.

## 1.3 Remaining open mismatches

### OPEN — Representative reuse biases the node-value estimator

Current runtime prunes pilot prefixes using TV-based duplicate structure, reuses surviving representatives, and computes the parent value as an unweighted mean of final children. The retained sample is not iid from the original rollout policy, so the estimator can be biased.

### OPEN — Two required pilot execution modes

The runtime must implement both:

1. `fresh_iid`: theory-safe main mode;
2. `weighted_reuse`: pilot-efficient mode with explicit multiplicity/weight correction.

They must never silently fall back into one another.

### OPEN — Terminal shortcuts are excluded from `C_s`

Terminal phase-one pilots consume final branch slots but are excluded from the continuation TV matrix. Therefore the current `C_s` does not represent the dispersion of the actual final-child estimator.

### OPEN — Scorer uses text concatenation instead of exact token IDs

Retokenizing `prefix + continuation` can move the BPE boundary. The scorer must condition and slice using exact rollout token IDs.

### OPEN — Sampling distribution and scoring distribution can differ

Support continuations may be sampled with rollout temperature/top-p while scorer likelihoods are computed under a different distribution. The tanh likelihood-ratio estimator is meaningful only when samples and likelihoods refer to the same distributions.

### OPEN — Query/response truncation can invalidate PPO ratios

`edges_to_dataproto` can left-truncate queries and right-truncate responses after generation. Then current and stored behavior log-probabilities need not condition on the same token sequence.

### OPEN — `global_steps` is not the optimizer-step counter

No-edge, postponed, and warmup iterations currently consume global steps. Main curves by iteration and max-edge-age semantics are therefore ambiguous.

### OPEN — PPO all-masked-token NaN

The custom masked mean divides by `mask.sum()` without a zero-denominator guard.

### OPEN — Allocation accounting overwrites true solver bounds

The queue manager rewrites node accounting without forwarding `summary.lower_bounds` and `summary.upper_bounds`, so persisted caps may differ from the optimization problem actually solved.

### OPEN — Queue-local versus frontier-global claim

Online timeout allocation optimizes within each queue flush, not over the entire depth frontier. The paper and manifest must define `Q` as the flush set unless a depth-global allocator is used.

### OPEN — Legacy reserve terminology remains primary

Reserve contribution/consumption fields remain in logs although the reserve pool is no longer an allocation input. Primary reporting must use pruning, expansion, transfer, bounds, and objective fields.

### OPEN — `fixed_total_generated` naming and claim

The cap corresponds to a uniform full-tree maximum-style token cap, not expected or realized SPO compute under early stopping.

### OPEN — No real end-to-end Ray + FSDP + async-vLLM smoke evidence

Unit tests alone do not establish policy synchronization, log-prob alignment, optimizer-step counting, or absence of NaNs in the deployed stack.

---

# 2. Canonical mathematical contract

For every allocation flush `Q`, define one scalar nonnegative allocation coefficient `C_s` for each node `s` and solve the exact bounded integer problem above.

The allocation solver is valid conditional on its input coefficients. Claims about variance reduction additionally require the runtime estimator associated with `C_s` to match the theory.

The main paper must use the following careful language:

> VDRA uses short-horizon policy-divergence information to construct a node-wise dispersion proxy `C_s`, then solves an exact bounded integer resource-allocation problem within each online queue flush.

With `tail_mode=none`, do not call `C_s` a certified full-horizon upper bound. It is a relative short-horizon allocation proxy inspired by value-difference bounds.

For `fresh_iid`, the final branch samples are iid conditional on the parent and behavior policy, so the standard Monte Carlo variance interpretation remains defensible.

For `weighted_reuse`, the final estimator is a weighted representative estimator and requires its own derivation and empirical validation. It must not inherit the iid Monte Carlo claim automatically.

---

# 3. Dual pilot execution modes

Add configuration:

```yaml
gear_tree:
  gear:
    pilot_execution_mode: fresh_iid  # fresh_iid | weighted_reuse
    weighted_reuse_fallback: fresh_iid
    representative_weight_mode: cluster_multiplicity
    terminal_pilot_handling: include_in_dispersion
```

The selected mode must be written to every run manifest, tree artifact, node record, and experiment name.

## 3.1 Mode A — `fresh_iid` (main-paper default)

### Semantics

Pilots and support blocks are used only to estimate:

- pairwise short-horizon divergence;
- redundancy structure and `predicted_k`;
- allocation coefficient `C_s`.

After the allocator returns `k_s*`, discard pilot trajectories from the final value/training sample and generate exactly `k_s*` fresh children from the current frozen behavior policy:

\[
x_{s,1},\ldots,x_{s,k_s^*}\overset{iid}{\sim}\pi_{\theta_t}(\cdot\mid s).
\]

The parent estimator is

\[
\hat V_s^{\mathrm{fresh}}
=
\frac{1}{k_s^*}\sum_{j=1}^{k_s^*}\hat V(x_{s,j}).
\]

### Required implementation

1. Add a separate expansion function such as `_expand_fresh_iid`.
2. Do not attach terminal shortcuts or reusable pilots as final children.
3. Generate exactly `allocated_k` final children, subject only to an explicitly selected total-token cap mode.
4. Preserve all pilot/support compute in overhead accounting.
5. Store `pilot_execution_mode=fresh_iid` on nodes and edges.
6. Assert:

```text
final_children == allocated_k
pilot_children_reused == 0
representative_weights_absent_or_all_one
```

7. If token-cap mode prevents exact generation, mark the run as budget-capped and not valid for the fixed-branch theoretical claim.

### Reporting

This is the default for primary accuracy, convergence-by-iteration, and theoretical-alignment experiments. It is more expensive because pilot work is not reused; report that overhead honestly.

## 3.2 Mode B — `weighted_reuse` (efficiency variant)

### Semantics

Use pilot clustering to create representative groups. Every continuable pilot and terminal pilot must belong to exactly one cluster. For representative `r`, define multiplicity

\[
m_r=|G_r|,
\qquad
w_r=\frac{m_r}{\sum_q m_q}.
\]

The cluster assignment must be deterministic for fixed inputs and may depend on pairwise TV/redundancy, but not on observed rewards.

If a representative is completed into a child trajectory, compute the parent estimator as

\[
\hat V_s^{\mathrm{reuse}}
=
\sum_{r\in\mathcal R_s}w_r\hat V(r).
\]

If fresh extra branches are added beyond the representative set, define and implement a single mathematically explicit mixture estimator. Do not combine weighted representatives and fresh samples with an arbitrary unweighted mean.

Recommended estimator:

1. Representatives account for `M_rep` original pilot draws through multiplicities.
2. Each fresh child has multiplicity 1.
3. Normalize all multiplicities:

\[
\hat V_s
=
\frac{
\sum_{r\in\mathcal R_s}m_r\hat V(r)
+
\sum_{f\in\mathcal F_s}\hat V(f)
}{M_{rep}+|\mathcal F_s|}.
\]

This estimator approximates the empirical pilot distribution plus additional iid draws. It is not exactly the same estimator as fresh iid Monte Carlo; treat it as a separate method variant.

### Required implementation

1. Replace survivor-only output from `_unique_prefix_indices` with explicit cluster assignments:

```python
cluster_id_per_pilot: list[int]
representative_index_per_cluster: dict[int, int]
cluster_size: dict[int, int]
```

2. Every pilot, including terminal shortcuts, must be assigned to a cluster.
3. Store on each representative:

```text
vdra_cluster_id
vdra_cluster_multiplicity
vdra_representative_weight
vdra_original_pilot_indices
```

4. Propagate edge/sample weights through:

- parent reward aggregation;
- parent reward standard deviation or weighted dispersion statistic;
- local child-parent advantage;
- replay edge records;
- token-level actor loss via `rollout_is_weights` or a dedicated `edge_weights` field.

5. The weighted parent value used to compute advantages must be identical to the value logged in artifacts.
6. Representative selection must not use reward or correctness.
7. If valid cluster weights cannot be formed, apply the configured explicit fallback `fresh_iid`, set `weighted_reuse_fallback_triggered=true`, and exclude that tree from pure weighted-reuse analysis.
8. Assert:

```text
sum(cluster_multiplicity) == pilot_children_generated
all cluster_multiplicity >= 1
sum(representative_weight) == 1 within tolerance
no pilot belongs to multiple clusters
no pilot is unassigned
```

### Required theory note

The paper must present weighted reuse as a coreset/representative estimator based on empirical pilot multiplicities, not as iid Monte Carlo. The variance objective used for allocation must either be rederived for this weighted estimator or be described as a heuristic allocation proxy for this variant.

## 3.3 Ablation matrix

At minimum run:

| Variant | Pilot mode | Allocation | Purpose |
|---|---|---|---|
| SPO | none | uniform | baseline |
| VDRA-Fresh | fresh_iid | exact VDRA | theory-safe main |
| VDRA-Weighted | weighted_reuse | exact VDRA | compute-efficient variant |
| Uniform-Fresh | fresh_iid | uniform | isolates allocation signal |
| Uniform-Weighted | weighted_reuse | uniform | isolates reuse/weighting effect |

Primary method claims should be based on `VDRA-Fresh` until the weighted estimator derivation and tests are complete.

---

# 4. Terminal pilots and definition of `C_s`

Terminal phase-one pilots have observed bounded return and consume branch probability mass. They cannot be excluded from dispersion while still consuming final branch slots.

Implement one of the following, with `include_in_dispersion` as the main choice:

1. Include each terminal pilot as a distribution atom with known terminal value/reward.
2. Extend the pairwise bound matrix to terminal–terminal and terminal–continuation pairs.
3. Construct `C_s` over all pilot groups used by the selected estimator.

For `fresh_iid`, terminal pilots affect `C_s` and `predicted_k` but are still discarded before final iid generation.

For `weighted_reuse`, terminal pilots become weighted representatives and their observed terminal values enter the weighted estimator.

Log separately:

```text
C_continuation
C_terminal
C_cross
C_total
```

Assert `C_total` is the coefficient passed to the allocator.

---

# 5. Scorer, token, and policy invariants

## 5.1 Real policy synchronization

Preferred implementation: expose prompt-token log-prob scoring through the same `AsyncLLMServerManager`/served rollout model used for pilots and main generation.

Alternative implementation: maintain an external scorer server only if all of the following are enforced:

- actor update synchronizes both rollout and scorer weights;
- both expose a monotonic model version;
- a weight fingerprint or version handshake is verified before every rollout iteration;
- scorer requests carry that version;
- mismatch aborts strict VDRA.

Snapshot strings supplied by configuration are not sufficient evidence.

## 5.2 Exact token-ID scorer

Replace text-only scorer calls with:

```python
score_one(
    prefix_token_ids: list[int],
    continuation_token_ids: list[int],
    policy_snapshot_id: str,
) -> float
```

The scorer must compute the continuation log-likelihood on exact concatenated token IDs. No decode–concatenate–retokenize path is allowed in strict mode.

Required assertions:

```text
scored_input_ids == prefix_token_ids + continuation_token_ids
returned continuation logprobs count == len(continuation_token_ids)
all scored values finite or handled by explicit invalid-support policy
```

## 5.3 Same sampling and likelihood distributions

For the main configuration, choose the simplest auditable setting:

```yaml
actor_rollout_ref:
  rollout:
    temperature: 1.0
    top_p: 1.0
```

If nontrivial temperature/top-p is used, scorer likelihoods must apply the same transformed and renormalized distribution. Store these parameters in the manifest and parity records.

## 5.4 Real log-prob parity diagnostic

Add a script that:

1. freezes one actor/rollout snapshot;
2. generates or loads fixed exact prompt/response token IDs;
3. gets chosen-token log-probs from vLLM;
4. gets aligned log-probs from the actor forward pass;
5. records model version, tokenizer ID, temperature, top-p, precision, BOS/EOS convention;
6. checks max and mean absolute deltas.

The helper-only JSON test is not sufficient. A real diagnostic must pass before long runs.

---

# 6. PPO conditioning and truncation

PPO numerator and denominator must condition on identical token sequences.

In strict mode, `edges_to_dataproto` must never silently truncate an edge generated under a longer context.

Before replay insertion or batch conversion, enforce:

```text
len(query_token_ids) <= max_prompt_length
len(response_token_ids) <= max_response_length
```

Recommended behavior: reject the edge and fail the smoke test. Do not crop only at training time.

If context cropping is necessary, crop before generation and store the exact cropped behavior query tokens. Add fields:

```text
behavior_query_token_ids
behavior_response_token_ids
behavior_context_truncated
```

Then assert actor input exactly equals the stored behavior sequence.

Do not mutate `actor_shifted_log_probs` inside `edges_to_dataproto`.

---

# 7. Trainer counters and replay semantics

Maintain distinct counters:

```text
rollout_iteration
optimizer_step
successful_actor_updates
postponed_updates
failed_updates
```

Use `optimizer_step` for:

- `trainer.total_training_steps`;
- replay edge age;
- policy snapshot version;
- validation/save frequency when reporting performance by training iteration.

A no-edge, postponed, critic-warmup, or failed actor call must not increment `optimizer_step`.

Refactor replay into an explicit transaction:

```python
reservation = replay_buffer.reserve_for_update(current_optimizer_step)
try:
    actor_output = update_actor(batch)
except Exception:
    replay_buffer.rollback(reservation)
    raise
else:
    replay_buffer.commit(reservation)
    optimizer_step += 1
```

Reserved edges must not be sampled by another reservation.

Underfill policy for main runs:

- postpone until at least one valid mini-batch exists;
- prefer exactly `target_edges_per_update` for baseline parity;
- log every underfill/postponement;
- never duplicate edges.

---

# 8. PPO numerical safety and weighted loss

Fix `_masked_mean`:

```python
mask_sum = mask.sum()
if mask_sum == 0:
    return a differentiable zero and zero-valued metrics
```

Log:

```text
actor/all_tokens_prob_masked
actor/valid_action_tokens
```

For `weighted_reuse`, propagate representative/example weights into the actor loss. Define whether weights normalize per edge, per question, or globally. Recommended:

\[
L
=
\frac{\sum_e w_e\sum_t m_{e,t}\ell_{e,t}}
{\sum_e w_e\sum_t m_{e,t}}.
\]

Do not multiply the loss by weights and then divide by an unweighted token count.

Add tests for:

- all tokens masked;
- one weighted representative with multiplicity greater than one;
- weighted loss equivalence to explicitly duplicated identical edges;
- zero/negative/nonfinite weights rejected.

---

# 9. Allocation bounds and accounting

The queue manager must pass the actual solver outputs back into node accounting:

```python
lower_bound=summary.lower_bounds[node_id]
upper_bound=summary.upper_bounds[node_id]
```

Persist:

```text
requested_budget
allocated_budget
lower_bound
upper_bound
predicted_k
default_k
allocated_k
pruned_k
expanded_k
transferred_budget
objective_before
objective_after
solver_name
solver_time_ms
feasibility_repair_count
```

Compute `objective_before` on a feasible reference allocation with the same total budget and bounds. Never compare the optimized objective to an infeasible raw default vector.

Legacy reserve fields may remain aliases temporarily, but must not be the primary paper metrics.

---

# 10. Allocation scope and reproducibility

For `allocation_runtime=online_timeout`, define:

\[
\mathcal Q=\text{the set of nodes in one queue flush}.
\]

Do not claim a depth-global optimum.

Because queue membership can depend on wall-clock scheduling, log:

```text
queue_id
node_ids_in_flush
queue_size_at_flush
flush_reason
queue_wait_seconds
arrival_order
policy_snapshot_id
```

Add a deterministic `depth_batch` comparison that allocates over the complete depth frontier. Use it as an ablation to quantify the effect of online queue partitioning.

Run repeated fixed-seed online rollouts and report allocation stability across scheduling runs.

---

# 11. Compute-budget terminology

Rename `fixed_total_generated` to a name that matches its semantics, for example:

```text
uniform_full_tree_token_cap
```

Describe it as a maximum-style cap derived from a full uniform tree with configured segment limits, not expected or realized SPO compute.

Report separately:

```text
pilot_decode_tokens
support_decode_tokens
final_expansion_decode_tokens
proxy_rollout_tokens
scorer_prefill_tokens
scorer_continuation_tokens
wall_clock_seconds
```

For fair main comparisons, always report both:

1. performance versus successful optimizer updates;
2. performance versus wall-clock time / total token-equivalent compute.

---

# 12. Required tests

## 12.1 Unit tests

### Pilot modes

- `fresh_iid` never reuses pilots as final children.
- `fresh_iid` produces exactly `allocated_k` final children when not token-capped.
- `weighted_reuse` assigns every pilot to exactly one cluster.
- multiplicities sum to number of pilots.
- weighted parent value matches explicit pilot duplication.
- terminal pilots enter `C_total`.
- representative selection is reward-independent.

### Scorer and policy

- exact token-ID boundary test with a tokenizer where `tok(x+y) != tok(x)+tok(y)`.
- real snapshot/version mismatch aborts.
- temperature/top-p mismatch aborts in strict mode.
- actual rollout-vs-actor parity diagnostic fixture.

### PPO and replay

- overlength query/response fails before training.
- batch conversion does not mutate stored edge log-probs.
- all-masked PPO loss is finite zero.
- reserve/commit/rollback semantics.
- postponed iteration does not increment optimizer step.
- failed actor update leaves edges in replay.
- age uses optimizer steps, not rollout attempts.

### Allocation and accounting

- brute-force optimality over randomized small bounded instances.
- exact budget and bounds.
- persisted lower/upper equal solver summary.
- feasible baseline objective.
- median <1 ms and p99 <5 ms for normal target environment, with nonbinding CI fallback thresholds where needed.

## 12.2 End-to-end smoke test

Run 2–5 successful actor optimizer updates with Ray + FSDP + async vLLM for:

1. SPO-tree;
2. VDRA `fresh_iid`;
3. VDRA `weighted_reuse`.

The smoke test must assert:

```text
successful_actor_updates == requested_steps
no NaN or Inf in loss, ratio, KL, advantages, values, or weights
rollout/scorer/edge snapshot versions match
rollout-vs-actor old-logprob parity within configured tolerance
no query or response truncation
replay removal equals successfully trained edge IDs
all queues empty and all futures resolved
all finalized nodes have rewards
sum allocations equals target per flush
final children obey selected pilot-mode contract
all compute counters are nonnegative and internally consistent
```

Persist a machine-readable smoke report. A unit-test-only pass is not sufficient.

---

# 13. Experiment protocol after P0 completion

Main-paper default:

```yaml
pilot_execution_mode: fresh_iid
tail_mode: none
bound_form: linear
allocation_runtime: online_timeout
allocation_proxy: vdra
strict_vdra: true
temperature: 1.0
top_p: 1.0
```

Required comparisons:

- same model checkpoint;
- same tokenizer and prompt template;
- same dataset order and seeds;
- same successful optimizer updates;
- same replay protocol;
- same PPO mini-batch and epochs;
- exact logged tree/segment configuration;
- wall-clock and total compute accounting.

`weighted_reuse` is reported as a separate efficiency variant until its estimator derivation and weighted-loss validation are complete.

TreePO/TreeRL variants remain `*-style` unless official parity is established.

---

# 14. Implementation order

## P0 — Before any long run

1. Implement `fresh_iid` and make it the main default.
2. Implement complete clustering/multiplicity representation for `weighted_reuse`.
3. Include terminal pilots in the coefficient used by allocation.
4. Replace text scorer with exact token-ID scoring.
5. Guarantee real scorer/rollout weight synchronization.
6. Enforce matching sampling/scoring distributions.
7. Remove silent query/response truncation.
8. Separate rollout iteration from optimizer step.
9. Add replay reserve/commit/rollback.
10. Add zero-mask PPO guard and weighted loss.
11. Preserve true allocation bounds/objectives in artifacts.
12. Pass all unit tests and the three-mode end-to-end smoke test.

## P1 — Before main paper experiments

1. Run real log-prob parity diagnostics on the target model/precision.
2. Validate weighted estimator against explicit duplication on sampled trees.
3. Compare online-timeout versus depth-batch allocation.
4. Finalize compute-budget terminology and plots.
5. Freeze controlled SPO and VDRA presets.
6. Run short paired-seed experiments before committing to full runs.

## P2 — Paper/rebuttal strengthening

1. Derive or bound variance for the weighted representative estimator.
2. Calibrate tail correction and compare `none` versus `calibrated`.
3. Study allocation stability under runtime scheduling.
4. Add oracle and empirical-variance proxy diagnostics.
5. Establish official baseline parity where feasible.

---

# 15. Stop conditions

Do not start or report a long main-paper run if any of the following holds:

- final estimator mode is missing from the manifest;
- scorer and rollout weights are not verifiably synchronized;
- PPO context was silently truncated;
- optimizer-step count differs from successful actor updates;
- any sampled edge is removed without successful training;
- `fresh_iid` reuses a pilot;
- `weighted_reuse` has missing/invalid multiplicities or unweighted aggregation;
- terminal pilots are excluded from the actual allocation coefficient;
- allocation artifacts do not match solver bounds;
- smoke test contains NaN/Inf or unresolved queue state.

Partial completion is not sufficient for main claims. Any remaining deviation must be labeled as an ablation or engineering approximation rather than silently folded into VDRA.