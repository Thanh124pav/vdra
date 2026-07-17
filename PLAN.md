# PLAN.md — VDRA Runtime Correctness and Experiment Readiness

## 0. Objective and audited baseline

This plan supersedes the previous checklist and is based on repository commit:

```text
b914a1c1db21551822a93e986f92debe7d96c209
```

Audit date: 2026-07-17.

The objective is not merely to make the trainer execute. Every reported result must correspond to a precisely defined and verifiable:

- behavior-policy snapshot;
- rollout and scoring distribution;
- request/session routing protocol;
- node-value estimator;
- allocation scope and budget;
- PPO behavior denominator;
- representative-reuse weighting objective;
- optimizer-update count;
- compute-accounting protocol.

Status labels:

- **DONE**: implemented and supported by a meaningful direct test.
- **PARTIAL**: plumbing exists, but the deployed end-to-end invariant is not guaranteed.
- **OPEN**: still inconsistent, ambiguous, or untested in the Ray/FSDP/async-vLLM stack.

**No long main-paper training run is allowed until all common P0 items are complete and the real async smoke test passes.**

**`weighted_reuse` must remain an experimental efficiency variant until all weighted-reuse P0 items are complete.**

---

# 1. Current audited status

## 1.1 DONE or substantially aligned

The following parts are now substantially aligned:

1. The production allocator solves the bounded integer problem with exact budget preservation inside each allocation flush.
2. True solver lower and upper bounds are retained in accounting.
3. `fresh_iid` exists and discards pilots before final-child generation.
4. Generation-time token log-probabilities are stored on tree edges.
5. Replay batches set `force_stored_old_log_probs=True`.
6. The actor no longer overwrites stored behavior log-probabilities in the one-minibatch/one-epoch shape when that flag is present.
7. Replay uses reservation, commit, and rollback semantics.
8. `global_steps` advances only after a successful actor update.
9. Query and response truncation is rejected rather than silently changing the PPO sequence.
10. The unused critic is disabled in the shipped configuration.
11. Snapshot IDs are propagated through `gen_batch.non_tensor_batch` and stored on generated edges.
12. Star clustering replaces connected-component clustering for continuable pilots.
13. Representative fields can travel from a correctly annotated tree child through edge extraction into `DataProto.edge_weights`.
14. The policy loss has a finite zero-mask path.
15. The tree artifact records that the runtime allocator acts per queue flush.

These items still require inclusion in the final real-stack smoke test; unit plumbing alone is not final evidence.

## 1.2 PARTIAL or false-positive fixes

### PARTIAL — Sticky routing

A sticky-session patch exists, but its current implementation is not safe:

- `bytes(token_ids)` fails for token IDs above 255;
- concurrent siblings with the same prefix receive the same request ID;
- a continuation has a different prompt hash, so pilot-to-continuation affinity is not preserved.

### PARTIAL — Snapshot propagation

The edge receives the dynamic rollout snapshot ID, but the gate/scorer is constructed in `TreeAgentLoop.__init__` from static worker configuration before `run()` receives the dynamic snapshot.

### PARTIAL — Scorer synchronization

The scorer client is cached, which improves resource usage. However, `scorer.weight_version` is currently assigned from the expected rollout label. It is not a server-reported model fingerprint and therefore does not prove equal weights.

### PARTIAL — Terminal dispersion

The estimator can use observed terminal reward differences when rewards already exist. The real gate construction does not currently pass a terminal grader callback, so runtime terminal pilots normally reach the dispersion calculation without rewards.

### PARTIAL — Weighted edge plumbing

The tree-to-edge-to-batch path can carry weights, but candidate cluster metadata is lost when a candidate is converted to `SegmentSample`. As a result, the runtime often reconstructs all multiplicities as one.

### PARTIAL — Context validation

The trainer contains a context check, but the current `max_prompt * 8` threshold does not guarantee that an accumulated edge query fits `data.max_prompt_length`.

### PARTIAL — Test quality

Several tests inspect source strings rather than exercising the runtime contract. They are useful regression hints but do not establish correctness.

---

# 2. Canonical method contracts

## 2.1 Allocation contract

For each online queue flush `Q`, solve:

\[
\min_{k_s\in\mathbb Z_+}\sum_{s\in Q}\frac{C_s}{k_s}
\]

subject to:

\[
\sum_{s\in Q}k_s=B_Q,
\qquad
\ell_s\leq k_s\leq u_s.
\]

The runtime claim is **per queue flush within one tree**, not whole-tree and not whole-depth-frontier optimization.

Canonical manifest value:

```text
allocation_scope = per_queue_flush_within_tree
```

## 2.2 Main-paper estimator contract: `fresh_iid`

Pilots and support samples estimate redundancy and `C_s` only. After allocation, discard them and generate exactly `allocated_k` new children from the frozen behavior policy:

\[
x_{s,1},\ldots,x_{s,k_s}\overset{iid}{\sim}\pi_{\theta_t}(\cdot\mid s).
\]

Use:

\[
\hat V_s=\frac{1}{k_s}\sum_{j=1}^{k_s}\hat V(x_{s,j}).
\]

Required invariants:

```text
pilot_children_reused == 0
pilot_children_shortcut == 0
pilot_children_discarded == pilot_children_generated
final_children == allocated_k unless an explicitly reported token cap is hit
all final edge weights are absent or equal to one
```

`fresh_iid` is the default main-paper method.

## 2.3 Efficiency estimator contract: `weighted_reuse`

Every pilot belongs to exactly one valid representative cluster. Terminal pilots are singleton clusters unless a separate mathematically valid terminal clustering rule is implemented.

For representative `r` with multiplicity `m_r`, and fresh extra children with multiplicity one:

\[
\hat V_s=
\frac{
\sum_{r\in\mathcal R_s}m_r\hat V(r)
+
\sum_{f\in\mathcal F_s}\hat V(f)
}{
\sum_{r\in\mathcal R_s}m_r+|\mathcal F_s|
}.
\]

This is a representative approximation to an empirical pilot sample plus fresh draws. It is not iid Monte Carlo and must not inherit the `fresh_iid` variance claim.

For the actor objective, use one explicit interpretation:

```text
edge_weight = represented sample multiplicity
representative edge: edge_weight = cluster_multiplicity
fresh edge:          edge_weight = 1
```

The token loss then matches the objective obtained by duplicating each representative edge according to its multiplicity. Do not use a per-parent normalized representative probability as the global actor weight unless a two-stage parent-wise reduction is explicitly implemented.

---

# 3. Common P0 blockers

## P0.1 — Separate unique request identity from sticky server affinity

### Current failure

The current session key is derived using:

```python
bytes(list(prompt_ids)[-256:])
```

This raises for normal vocabulary IDs above 255. The same prefix also creates the same request ID for concurrent siblings, while a pilot continuation creates a different hash and loses affinity.

### Required architecture

Use two separate identifiers:

```text
request_id: unique for every generation call
sticky_key: stable across calls belonging to one branch/session
```

Recommended API:

```python
await server_manager.generate(
    request_id=unique_request_id,
    sticky_key=branch_session_id,
    prompt_ids=prompt_ids,
    sampling_params=sampling_params,
)
```

`AsyncLLMServerManager` must choose the server using `sticky_key` and forward the unique `request_id` to the selected vLLM server.

Recommended branch session identity:

```text
run_id / rollout_iteration / tree_id / parent_segment_id / pilot_or_child_index
```

Do not derive the complete branch identity only from prompt token contents.

### Required implementation targets

- `verl/recipe/gear_tree/async_tree_rollout.py`
- upstream async server manager wrapper used by the repository
- `SegmentNodeExpander` and pilot completion path

### Acceptance tests

1. Token IDs containing values above 255 do not fail.
2. Eight concurrent siblings have eight unique request IDs.
3. A pilot and its continuation share the same sticky key.
4. Different sibling branches do not share a sticky key.
5. A fake two-server manager confirms pilot and continuation reach the same server.

---

## P0.2 — Make gate/scorer snapshot state dynamic per rollout

### Current failure

`TreeAgentLoop` constructs the gate/scorer in `__init__`, but the current snapshot is only available later in `run()` through per-row kwargs. Edge provenance can therefore be current while the scorer object remains static or unstamped.

### Required contract

For optimizer step `t`:

\[
\theta_{\mathrm{pilot}}
=
\theta_{\mathrm{support}}
=
\theta_{\mathrm{scorer}}
=
\theta_{\mathrm{behavior}}
=
\theta_t.
\]

Before building a tree, the runtime must bind the gate/scorer to the dynamic snapshot supplied to that sample.

Acceptable designs:

1. `TreeAgentLoop.run()` creates or retrieves a gate keyed by verified server weight version; or
2. the gate exposes `bind_snapshot(snapshot_id, verified_weight_version)` and rejects stale state; or
3. scoring is routed through the same async server manager/model replicas used for rollout.

Static `trainer_config` values are not sufficient.

### Acceptance tests

1. Step `t` tree edges carry snapshot `t`.
2. The gate used for that tree reports expected snapshot `t`.
3. After an actor update, a step `t+1` tree cannot reuse gate state stamped only for `t`.
4. A stale scorer causes an explicit error before pilot scoring.

---

## P0.3 — Replace self-stamped scorer labels with a real server-side version handshake

### Current failure

Assigning:

```python
scorer.weight_version = rollout_snapshot
```

only records the client's expectation. It does not inspect the scorer model weights.

### Required contract

Each rollout/scorer server must expose or return a version that changes when actor weights are synchronized. Examples:

- actor update number acknowledged by the server;
- loaded checkpoint revision;
- monotonic server-side weight version;
- deterministic lightweight parameter fingerprint.

The run must record:

```text
expected_policy_snapshot_id
rollout_server_weight_version
scorer_server_weight_version
version_verified = true|false
```

Strict mode must fail when:

```text
rollout_server_weight_version != scorer_server_weight_version
```

Best design: use the same server pool and weight-sync mechanism for generation and exact-token likelihood scoring.

### Acceptance tests

1. A mocked server with an old version is rejected even when string snapshot labels match.
2. A successful actor sync advances the server-side version.
3. A two-step real-stack smoke records matching rollout/scorer versions at both steps.

---

## P0.4 — Wire terminal grading into the real dispersion pipeline

### Current failure

The estimator supports `terminal_reward_fn`, but the real `GearGate` estimator construction does not pass it. Missing terminal rewards currently produce zero terminal-terminal contribution, which can underestimate dispersion.

### Required contract

Before computing terminal-terminal dispersion, grade every terminal phase-one pilot using the same reward function and dataset instance used for final leaves.

For terminal pilots `i,j`:

\[
B_{ij}^2=(R_i-R_j)^2.
\]

For a terminal-continuation pair, retain a documented conservative bound until a tighter valid rule is derived.

In strict mode:

- grader failure must abort the estimate;
- missing terminal reward must not silently contribute zero;
- an explicit worst-case fallback may be used only if logged and mathematically conservative.

### Required implementation targets

- pass a grader closure from `tree_rollout.py` to `GearGate.estimate_node_async`;
- pass it into `ConditionalTVEstimator`;
- remove exception swallowing that converts grader failures to missing rewards in strict mode.

### Acceptance tests

1. Full runtime path with terminal rewards `[0,1]` gives `C_terminal > 0`.
2. The test must start from a tree parent and real gate/estimator construction, not direct invocation of the helper.
3. A terminal grader error aborts strict VDRA.
4. No terminal pilot is ungraded when `terminal_pilot_handling=include_in_dispersion`.

---

## P0.5 — Enforce the actual accumulated-query context bound

### Current failure

The current validation compares a worst-case query against `max_prompt * 8`, while `edges_to_dataproto` rejects any query longer than `data.max_prompt_length`.

### Required contract

Let `L_original,max` be the maximum tokenized prompt length after all preprocessing and chat templating. For maximum edge depth `d` and segment size `M`:

\[
L_{original,max}+(d-1)M\leq L_{edge,max}.
\]

`L_edge,max` must equal the limit used by `edges_to_dataproto` and the actor input path.

Recommended configuration split:

```yaml
data:
  max_original_prompt_length: ...
  max_edge_prompt_length: ...
```

Either:

1. pre-filter/truncate original prompts before rollout to reserve segment headroom; or
2. increase the actor edge prompt limit while remaining within model context length.

Remove the `max_prompt * 8` heuristic.

### Acceptance tests

1. Startup fails for a configuration that can create an overlength deepest edge.
2. Boundary equality is accepted.
3. A dataset scan reports observed maximum chat-templated prompt length.
4. A real tree at maximum configured depth converts to `DataProto` without truncation.

---

# 4. `weighted_reuse` P0 blockers

## P0.W1 — Preserve cluster metadata through candidate to sample conversion

### Current failure

The estimator writes cluster fields directly on candidate nodes, while `_candidate_to_sample()` reads only `vdra_sample_metadata`. Multiplicity therefore falls back to one.

### Required implementation

When creating `SegmentSample`, explicitly copy:

```text
vdra_cluster_id
vdra_cluster_multiplicity
vdra_representative_weight
vdra_original_pilot_indices
```

from the candidate into sample metadata.

The runtime must never reconstruct a missing known multiplicity as one without an explicit error in strict weighted mode.

### Acceptance test

Use the complete path:

```text
TV estimator
→ candidate annotation
→ _candidate_to_sample
→ pilot completion
→ _sample_child
→ extract_edges_from_tree
→ edges_to_dataproto
→ actor policy loss
```

Start with cluster multiplicities `[3,1]` and verify the actor receives `[3,1]`, not `[1,1]` or normalized probabilities.

---

## P0.W2 — Require coverage of all clusters, including terminal singleton clusters

### Current failure

The fallback guard derives the required count primarily from continuable reusable pilots. Terminal shortcut clusters can be omitted from coverage accounting.

### Required implementation

Set once during estimate recording:

```python
node["vdra_required_cluster_count"] = len(
    result.representative_index_per_cluster
)
```

This count includes:

- continuable representative clusters;
- terminal singleton clusters.

Weighted reuse is allowed only when:

```text
allocated_k >= vdra_required_cluster_count
```

Otherwise execute the configured whole-node fallback:

```text
fresh_iid or error
```

Do not select a subset of clusters and renormalize the remaining mass.

### Acceptance tests

1. Two continuable clusters plus three terminal clusters require five slots.
2. `allocated_k=3` triggers fallback.
3. `allocated_k=5` preserves all five clusters.
4. Fallback output carries no representative weights.

---

## P0.W3 — Define one actor weighting objective and implement it exactly

### Selected objective

For the current implementation, use multiplicity weighting equivalent to duplicating represented samples:

```text
representative edge weight = cluster_multiplicity
fresh edge weight          = 1
```

Parent reward aggregation normalizes these positive weights internally. Actor loss uses their raw multiplicities in the global weighted token mean.

Do not pass:

```text
cluster_multiplicity / parent_denominator
```

as the global actor weight, because global renormalization then changes relative parent contribution in an undocumented way.

### Required metadata

Each weighted edge must store:

```text
parent_segment_id
cluster_id
cluster_multiplicity
original_pilot_indices
edge_weight
weight_objective = duplicate_empirical_samples
```

### Acceptance tests

1. A representative with multiplicity three produces the same policy loss and gradient as three duplicated identical edges, within numerical tolerance.
2. Fresh children have weight one.
3. Weighted parent reward equals the value logged in tree artifacts.
4. Mixed parents do not silently switch between multiplicity and normalized-probability semantics.

### Paper restriction

Until this section passes, `weighted_reuse` must not be reported as the main method and must not be called an unbiased estimator.

---

# 5. P1 engineering and reporting fixes

## P1.1 — Use actual per-request sampling parameters

`TreeAgentLoop.run(sampling_params, ...)` must honor the supplied request parameters rather than relying only on static `self.rollout_config` values.

Requirements:

- generation parameters used by the server are logged;
- gate validation uses those same actual parameters;
- validation overrides are either supported or explicitly rejected for VDRA scoring;
- tanh-TV strict mode remains restricted to matching unwarped distributions until transformed scoring is implemented.

## P1.2 — Skip placeholder reward computation

The first-edge placeholder exists only to satisfy `AgentLoopOutput`. Avoid re-grading it by setting an explicit placeholder reward or by propagating a flag through the upstream reward path.

Preferred minimal fix:

```python
reward_score = 0.0
```

on the placeholder output, because the custom trainer uses tree edges rather than this score.

## P1.3 — Make allocation scope consistent everywhere

Replace conflicting values in config, tree artifacts, and run manifest with:

```text
per_queue_flush_within_tree
```

Paper language:

> VDRA solves an exact bounded integer allocation problem over the nodes present in each online queue flush. Queue membership depends on asynchronous expansion timing.

Do not claim whole-tree or whole-frontier optimal allocation for this runtime.

## P1.4 — Fix or remove the legacy Simulation Lemma implementation

If `tv` denotes `epsilon_T / 2`, the tight discounted expression requires the second denominator factor:

\[
1-\gamma+\gamma\,tv.
\]

The current legacy code uses `1-gamma+tv`.

Options:

1. fix the formula and add numeric tests against the paper expression; or
2. remove/quarantine `bound_form=simulation_lemma` from runnable experiment configs.

Main experiments remain `bound_form=linear`, `tail_mode=none`, with short-horizon proxy language.

## P1.5 — Strengthen deterministic edge identity

Include sufficient rollout identity to avoid collisions when `global_steps` does not advance:

```text
policy_snapshot_id
rollout_iteration
stable_question_id
tree_row_index
parent_segment_id
child_index
```

Do not rely on batch-local enumeration alone.

Acceptance test: two stochastic rollouts of the same question at the same optimizer step produce disjoint edge IDs.

## P1.6 — Complete scorer lifecycle management

- call scorer-client shutdown explicitly;
- do not assume `127.0.0.1` resolves to the correct rollout/scorer server in multi-node Ray;
- prefer Ray server handles or the shared async server manager;
- cache model resolution and connection pools at worker scope;
- enforce a process-level concurrency limit rather than one semaphore per short-lived client.

## P1.7 — Strengthen run-manifest validity

Compute `run_valid_for_main_results` from actual invariants, not only `allocation_proxy != oracle`.

Recommended condition:

```python
run_valid_for_main_results = (
    allocation_proxy != "oracle"
    and scorer_weight_version_verified
    and no_unexpected_fallback
    and not unexpected_token_cap_hit
    and all_node_accounting_invariants_passed
    and all_snapshot_invariants_passed
    and pilot_mode_is_homogeneous_or_explicitly_reported
)
```

Manifest must include:

```text
policy_snapshot_id
rollout_server_weight_version
scorer_server_weight_version
scorer_weight_version_verified
allocation_scope
pilot_execution_mode
weighted_reuse_fallback_count
token_cap_hit_count
successful_actor_updates
rollout_iterations
```

## P1.8 — Retire misleading reserve terminology

Primary fields should be:

```text
pruned_budget
expanded_budget
transferred_budget_within_flush
lower_bounds
upper_bounds
objective_before
objective_after
```

Keep reserve fields only as deprecated compatibility aliases.

---

# 6. Required test suite

## 6.1 Direct unit and integration tests

The following tests must execute behavior, not merely search source text:

1. **High-token-ID routing test**
   - prompt contains token IDs above 255;
   - generation does not crash.

2. **Unique request / sticky branch test**
   - sibling request IDs are unique;
   - pilot and continuation share server affinity.

3. **Dynamic snapshot test**
   - two optimizer steps produce edge snapshots `t` and `t+1`;
   - gate state follows each step.

4. **Real version mismatch test**
   - equal expected labels but different server versions must fail.

5. **One-minibatch old-log-prob test**
   - stored old log-prob differs from current log-prob;
   - actor ratio is not forced to one.

6. **Terminal pipeline test**
   - terminal pilot rewards `[0,1]` flow through the real gate construction;
   - `C_terminal > 0`.

7. **Context-bound test**
   - deepest legal edge succeeds;
   - one-token overflow fails before training.

8. **Weighted end-to-end metadata test**
   - estimator multiplicities survive to actor tensors.

9. **Weighted duplicate-equivalence test**
   - multiplicity-weighted gradient equals duplicated-edge gradient.

10. **Terminal-cluster coverage test**
    - terminal singleton clusters contribute to required coverage.

11. **No-critic test**
    - Ray role mapping contains no critic worker.

12. **Allocation-scope manifest test**
    - config, tree artifact, and manifest all report `per_queue_flush_within_tree`.

## 6.2 Real-stack smoke tests

Run on the actual intended Ray/FSDP/async-vLLM environment.

### Smoke A — SPO

- at least two successful actor updates;
- no request-ID collision;
- no placeholder reward overhead failure;
- stored generation log-probs align with actor tokens.

### Smoke B — VDRA `fresh_iid`

- at least five successful actor updates;
- scorer/rollout server versions match every step;
- no terminal reward omission;
- no query/response truncation;
- exact final-child count unless explicitly budget-capped;
- finite loss and gradients;
- replay commit/rollback works after one injected actor failure.

### Smoke C — VDRA `weighted_reuse`

Run only after P0.W1–P0.W3 pass.

- at least five successful actor updates;
- nonuniform multiplicities are observed and preserved;
- cluster coverage fallback is exercised at least once;
- weighted parent values match artifacts;
- multiplicity-weighted actor loss is finite;
- fallback trees are separately counted.

## 6.3 CI requirement

Add a CI workflow for CPU-safe unit tests. The repository currently has no workflow evidence for the audited commit.

GPU smoke evidence may remain an external artifact, but record:

```text
commit SHA
resolved config
model revision
VERL/vLLM/PyTorch versions
GPU type
smoke log path
successful optimizer-update count
```

---

# 7. Go/no-go gates

## 7.1 Main-paper `fresh_iid`

**GO only when:**

- P0.1–P0.5 are complete;
- Smoke A and Smoke B pass;
- scorer weight versions are server-verified;
- context bounds are validated from real dataset statistics;
- no unreported algorithm fallback occurs;
- manifest validity is true.

## 7.2 `weighted_reuse`

**GO as an efficiency ablation only when:**

- all common P0 items pass;
- P0.W1–P0.W3 pass;
- Smoke C passes;
- the paper describes it as a representative approximation;
- fallback frequency and effective sample weights are reported.

## 7.3 Long training prohibition

Do not launch long main-result runs merely because unit tests pass. A real-stack smoke must first establish:

```text
correct weights
correct snapshots
correct token alignment
correct estimator semantics
finite optimizer behavior
correct run manifest
```

---

# 8. Implementation order

Apply fixes in this order:

```text
1. unique request ID + sticky branch key
2. dynamic gate/scorer snapshot binding
3. real server-side weight-version handshake
4. terminal grader wiring
5. exact context-length contract
6. preserve cluster metadata candidate → sample
7. count all clusters including terminal shortcuts
8. multiplicity-based actor weighting objective
9. honor per-request sampling parameters
10. skip placeholder reward computation
11. unify allocation-scope reporting
12. strengthen edge IDs and manifest validity
13. scorer shutdown and multi-node routing
14. CPU CI
15. real SPO / fresh_iid / weighted_reuse smoke tests
```

After each item, add a behavioral test. Do not mark an item DONE based only on a source-string assertion.

---

# 9. Experiment policy

## Primary comparison

Use:

```text
SPO
VDRA-fresh_iid
GRPO or the selected trajectory-level baseline
```

Primary plots:

- performance by successful optimizer update;
- performance by wall-clock training time;
- realized generated-token compute;
- pilot/support/scoring overhead;
- allocation statistics per queue flush.

## Efficiency ablation

Use:

```text
VDRA-weighted_reuse
```

Report separately:

- cluster multiplicity distribution;
- fallback frequency;
- representative coverage;
- effective weighted sample count;
- speed/compute saving relative to `fresh_iid`;
- accuracy difference relative to `fresh_iid`.

Do not merge `fresh_iid` and fallback-weighted trees under one unlabeled method name.

---

# 10. Final readiness checklist

Before generating main-paper tables, confirm all boxes:

```text
[ ] request IDs are unique and sticky keys preserve branch affinity
[ ] token IDs above 255 are supported
[ ] dynamic snapshot reaches edge, gate, rollout server, and scorer server
[ ] scorer/rollout versions are verified from servers
[ ] terminal pilots are graded before dispersion
[ ] accumulated edge queries fit without truncation
[ ] stored behavior log-probabilities survive the one-minibatch path
[ ] no critic worker is created
[ ] fresh_iid reuses zero pilots
[ ] weighted metadata survives candidate → actor
[ ] weighted coverage includes terminal clusters
[ ] actor weights implement the documented multiplicity objective
[ ] allocation scope is per_queue_flush_within_tree everywhere
[ ] placeholder reward is not recomputed
[ ] deterministic edge IDs do not collide across postponed iterations
[ ] run manifest rejects scientifically invalid runs
[ ] CPU CI passes
[ ] SPO real-stack smoke passes
[ ] VDRA-fresh_iid real-stack smoke passes
[ ] VDRA-weighted_reuse real-stack smoke passes before weighted results are used
```

Only after this checklist is complete should the repository be treated as ready for long AAAI 2027 experiment runs.
