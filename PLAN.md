# PLAN.md — VDRA Runtime Correctness and Experiment Readiness

## 0. Audited baseline and objective

This plan is aligned to repository commit:

```text
711124e4184748c97cbda2ce588925872f9ed961
```

Audit date: 2026-07-17.

The objective is not merely to make training execute. Every reported result must correspond to a precisely defined and verifiable:

- behavior-policy snapshot;
- rollout and likelihood-scoring distribution;
- request identity and sticky-routing protocol;
- local node-dispersion proxy `C_s`;
- online queue-flush allocation scope and budget;
- node-value estimator;
- PPO behavior denominator;
- representative-reuse weighting objective;
- optimizer-update count;
- compute-accounting protocol.

Status labels:

- **DONE**: implemented and supported by a meaningful behavioral test.
- **PARTIAL**: plumbing exists, but the deployed end-to-end invariant is not yet established.
- **OPEN**: still inconsistent, ambiguous, or untested in the intended Ray/FSDP/async-vLLM stack.

**Do not launch long main-result runs until all common P0 items and the real async smoke test pass.**

**`weighted_reuse` remains an experimental efficiency variant until all weighted-reuse P0 items pass.**

---

# 1. Canonical algorithm contracts

## 1.1 Online queue-flush allocation

At queue flush index `tau`, let

\[
\mathcal Q_\tau = \{s_1,\ldots,s_m\}
\]

be the set of currently ready nodes collected by one online queue flush. VDRA solves

\[
\min_{\{k_s:s\in\mathcal Q_\tau\}}
\sum_{s\in\mathcal Q_\tau}\frac{C_s}{k_s}
\]

subject to

\[
\sum_{s\in\mathcal Q_\tau} k_s = B_\tau,
\qquad
\ell_s \le k_s \le u_s,
\qquad
k_s\in\mathbb Z_+.
\]

The canonical runtime scope is:

```text
allocation_scope = per_queue_flush_within_tree
```

### Mixed-depth queues are valid

Nodes in `Q_tau` are **not required to belong to the same tree depth**.

Queue membership is determined by asynchronous readiness, queue capacity, and timeout. A legal flush may therefore contain nodes from different depths or different active subtrees of the same tree.

This is intentional and must not be treated as a correctness failure. The runtime is an asynchronous online batched allocator, not a level-synchronous frontier allocator.

Required interpretation:

- `C_s` is a node-local dispersion/difficulty proxy with the same operational meaning for every ready node;
- the flush budget `B_tau` belongs to the current queue flush, not to a depth frontier;
- the solver preserves the exact budget within that flush;
- depth remains metadata and may influence how `C_s` is estimated, but it is not a feasibility constraint.

If a future estimator includes an explicit remaining-horizon correction, that correction belongs inside `C_s = C(s,h_s)`. It does not imply that queues must be partitioned by depth.

Do not use paper notation such as `F_d` and `B_d` for the deployed runtime unless discussing a separate level-synchronous abstraction.

Recommended paper language:

> At each queue flush, VDRA jointly reallocates the available branching budget among the currently ready nodes. These nodes need not belong to the same tree depth, allowing the allocator to operate asynchronously without imposing level-wise synchronization.

### Required allocation invariants

For every flush:

```text
sum(allocated_k) == flush_budget
lower_bound_s <= allocated_k_s <= upper_bound_s
all nodes use one frozen behavior-policy snapshot
allocation scope is logged as per_queue_flush_within_tree
```

A mixed-depth integration test should verify exact budget preservation and must **not** expect a same-depth assertion.

## 1.2 Main-paper estimator: `fresh_iid`

Pilots and support samples estimate redundancy and `C_s` only. After allocation, discard them and generate exactly `k_s` fresh children:

\[
x_{s,1},\ldots,x_{s,k_s}
\overset{iid}{\sim}
\pi_{\theta_t}(\cdot\mid s).
\]

The node estimator is

\[
\widehat V_s
=
\frac{1}{k_s}
\sum_{j=1}^{k_s}\widehat V(x_{s,j}).
\]

Required invariants:

```text
pilot_children_reused == 0
pilot_children_shortcut == 0
pilot_children_discarded == pilot_children_generated
final_children == allocated_k unless an explicitly logged token cap is hit
all final edge weights are absent or equal to one
```

`fresh_iid` is the default main-paper method.

## 1.3 Efficiency estimator: `weighted_reuse`

Every pilot belongs to exactly one representative cluster. Terminal pilots are singleton clusters unless a separate mathematically valid terminal clustering rule is introduced.

For representative `r` with multiplicity `m_r`, and fresh children with multiplicity one:

\[
\widehat V_s=
\frac{
\sum_{r\in\mathcal R_s}m_r\widehat V(r)
+
\sum_{f\in\mathcal F_s}\widehat V(f)
}{
\sum_{r\in\mathcal R_s}m_r+|\mathcal F_s|
}.
\]

This is a representative approximation to an empirical pilot sample plus fresh draws. It is not iid Monte Carlo and must not inherit the `fresh_iid` variance claim.

The current selected local edge objective is:

```text
representative edge_weight = cluster_multiplicity
fresh edge_weight          = 1
```

Do not use a parent-normalized probability as the global actor weight unless a separate parent-wise loss reduction is implemented and documented.

---

# 2. Current aligned implementation

The following components are substantially aligned at commit `711124e...`:

1. The bounded integer allocator preserves the exact budget within each queue flush.
2. Solver lower and upper bounds are written into node accounting.
3. `fresh_iid` discards pilots before final-child generation.
4. Generation-time token log-probabilities are stored on tree edges.
5. Replay batches force the stored old log-probabilities to remain the PPO denominator.
6. The unused critic is disabled in the shipped configuration.
7. Snapshot IDs are propagated to generated edges.
8. Request identity and sticky-routing identity are separated.
9. High vocabulary token IDs no longer break sticky-key hashing.
10. Pilot continuations can retain pilot server affinity.
11. Star clustering replaces connected-component clustering.
12. Terminal grading is wired into the runtime dispersion path.
13. Strict terminal handling rejects missing terminal rewards.
14. Candidate cluster metadata is copied into `SegmentSample` metadata.
15. Required weighted-reuse cluster coverage includes terminal singleton clusters.
16. Online weighted-reuse fallback is triggered when allocated slots cannot cover every cluster.
17. Raw cluster multiplicity reaches the local actor edge weight.
18. Placeholder `AgentLoopOutput` reward is set explicitly.
19. Config, tree artifacts, and the online manifest use `per_queue_flush_within_tree`.
20. Mixed-depth queue membership is consistent with the intended asynchronous algorithm.

These items still require real-stack smoke evidence. Unit plumbing alone is not final experimental evidence.

---

# 3. Common P0 blockers

## P0.1 — Real rollout/scorer weight-version verification

### Current problem

The scorer can query `/version`, `/health`, or `/models`, but a returned software version or static model ID does not necessarily change when actor weights change.

A client-assigned label such as

```python
scorer.weight_version = policy_snapshot_id
```

does not prove which weights the scorer server has loaded.

The production path must compare two independently obtained server-side versions:

```text
rollout_server_weight_version
scorer_server_weight_version
```

### Required contract

For optimizer update `t`:

\[
\theta_{\text{pilot}}
=
\theta_{\text{support}}
=
\theta_{\text{scorer}}
=
\theta_{\text{behavior}}
=
\theta_t.
\]

Each server-side version must change when actor weights are synchronized. Acceptable implementations include:

- a monotonic weight-update counter acknowledged by every replica;
- a loaded checkpoint revision;
- a lightweight deterministic parameter fingerprint;
- scoring through the same rollout server pool and weight-sync mechanism.

Strict mode must fail before pilot scoring when:

```text
rollout_server_weight_version is missing
scorer_server_weight_version is missing
rollout_server_weight_version != scorer_server_weight_version
```

The manifest must record:

```text
expected_policy_snapshot_id
rollout_server_weight_version
scorer_server_weight_version
weight_version_verified
```

### Acceptance tests

1. Equal client snapshot labels but different server versions fail.
2. A successful actor update advances the server-side version.
3. A two-update real-stack smoke records matching rollout/scorer versions at both updates.
4. A static model ID alone is not considered a verified changing weight version.

## P0.2 — One context-length contract for validation and tensorization

### Current problem

Startup validation can use `max_edge_prompt_length`, while `edges_to_dataproto` may still use `data.max_prompt_length`. A configuration can therefore pass startup validation and fail when converting a deep edge into an actor batch.

### Required contract

Let `L_original,max` be the maximum chat-templated original prompt length, `D` the maximum number of segments, and `M` the nonterminal segment length:

\[
L_{original,max}+(D-1)M \le L_{edge,max}.
\]

The same resolved `L_edge,max` must be used by:

- startup validation;
- `edges_to_dataproto`;
- actor prompt tensorization;
- rollout prompt configuration;
- model-context validation.

Also enforce:

\[
L_{edge,max}+L_{response,max}
\le L_{model\ context}.
\]

Recommended configuration:

```yaml
data:
  max_original_prompt_length: ...
  max_edge_prompt_length: ...
```

### Acceptance tests

1. The deepest legal edge converts without truncation.
2. A one-token overflow fails before training.
3. Boundary equality is accepted.
4. A dataset scan reports the observed maximum chat-templated prompt length.

## P0.3 — Real async stack smoke

Before long main experiments, run the intended Ray/FSDP/async-vLLM stack for multiple successful optimizer updates.

The smoke must establish:

```text
correct behavior snapshot
verified scorer/rollout weights
stored behavior log-probabilities used as PPO denominator
finite loss and gradients
no request-ID collisions
no query/response truncation
exact queue-flush budgets
legal mixed-depth flush behavior
correct final-child accounting
replay commit/rollback after an injected actor failure
```

---

# 4. `weighted_reuse` P0 blockers

## P0.W1 — Define multiplicity semantics below a reused representative

### Open question

A reused representative child may itself be expanded and clustered later. The current implementation attaches a local cluster multiplicity to each edge, but does not explicitly define whether descendant weights should include ancestor multiplicity.

Two mathematically distinct objectives are possible.

### Option A — trajectory-duplication semantics

For a path with local multiplicities `m_1,...,m_d`:

\[
w_{path}=\prod_{j=1}^{d}m_j.
\]

Every descendant edge inherits the ancestor path weight and multiplies it by its local multiplicity.

### Option B — local conditional-edge semantics

Each parent defines a fresh local conditional objective. Descendant edges use only their local multiplicity and do not inherit ancestor multiplicity.

This option can be valid, but it must not be described as duplicating complete empirical trajectories.

### Required decision

Choose one objective, implement it consistently in:

- tree value aggregation;
- edge extraction;
- actor loss weighting;
- artifact logging;
- paper language.

Until this decision is explicit, `weighted_reuse` remains an ablation only.

### Acceptance test

Construct a two-level example with ancestor multiplicity `3` and local multiplicity `2`. Verify the documented expected descendant weight is exactly `6` for Option A or exactly `2` for Option B.

## P0.W2 — Strict source-aware multiplicity validation

### Current problem

A sample with missing multiplicity may still default to one. That is correct for a genuinely fresh child but unsafe for a representative whose metadata was lost.

### Required metadata

Every weighted sample must store:

```text
vdra_sample_source = representative | terminal_representative | fresh
vdra_cluster_id
vdra_cluster_multiplicity
vdra_original_pilot_indices
edge_weight
```

Strict behavior:

```text
representative missing multiplicity -> error
terminal representative missing multiplicity -> error
fresh missing multiplicity -> assign one
```

### Acceptance test

Delete multiplicity from a representative after candidate conversion. Strict weighted reuse must fail rather than silently use one.

## P0.W3 — Share the same coverage/fallback dispatcher across runtimes

The online runtime checks whether

```text
allocated_k >= required_cluster_count
```

and falls back to `fresh_iid` or raises when coverage is impossible.

The legacy `depth_batch` path must use the same guard and the same whole-node fallback semantics, or `weighted_reuse + depth_batch` must be explicitly rejected.

### Acceptance tests

1. Two continuable clusters plus three terminal clusters require five slots.
2. `allocated_k=3` triggers the configured fallback in every supported runtime.
3. Fallback output contains no representative weights.
4. `allocated_k=5` retains every required cluster.

---

# 5. P1 engineering and reporting fixes

## P1.1 — Honor actual per-request sampling parameters

`TreeAgentLoop.run(sampling_params, ...)` must use the supplied request parameters rather than relying only on static worker configuration.

Requirements:

- actual generation parameters are logged;
- gate validation uses those exact parameters;
- validation overrides are either supported or explicitly rejected;
- strict tanh-TV remains restricted to matching rollout/scorer distributions.

## P1.2 — Strengthen globally stable tree and edge identity

Every replay edge should include:

```text
policy_snapshot_id
rollout_iteration
stable_question_id
stable_tree_id
parent_segment_id
child_segment_id
child_index
```

Two stochastic rollouts of the same question at the same optimizer step must produce disjoint edge IDs.

Sticky keys should also include a globally stable tree identity rather than only local paths such as `root/...`.

## P1.3 — Strengthen run-manifest validity

`run_valid_for_main_results` must be computed from runtime evidence, not only `allocation_proxy != oracle`.

Recommended condition:

```python
run_valid_for_main_results = (
    allocation_proxy != "oracle"
    and rollout_scorer_weight_version_verified
    and no_unexpected_fallback
    and not unexpected_token_cap_hit
    and all_node_accounting_invariants_passed
    and all_snapshot_invariants_passed
    and context_contract_passed
)
```

Manifest fields:

```text
policy_snapshot_id
rollout_server_weight_version
scorer_server_weight_version
weight_version_verified
allocation_scope
flush_depths
pilot_execution_mode
weighted_reuse_fallback_count
token_cap_hit_count
successful_actor_updates
rollout_iterations
```

`flush_depths` is descriptive metadata. Multiple depths in one flush are legal and must not invalidate the run.

Do not overwrite a run-level manifest with weaker per-tree defaults such as unconditional `run_valid_for_main_results=True`.

## P1.4 — Scorer lifecycle and multi-node routing

- close cached scorer clients at worker shutdown;
- do not assume `127.0.0.1` points to the correct scorer on every Ray node;
- prefer server handles or the shared async server manager;
- cache model resolution and connection pools at worker scope;
- enforce a process-level concurrency limit.

## P1.5 — Remove broad protocol `TypeError` fallbacks

Do not retry a generation request merely because any internal `TypeError` occurred. Detect sticky-key capability once through an explicit interface/version check.

This avoids accidentally issuing a request twice after partial server-side execution.

## P1.6 — Fix or quarantine legacy Simulation Lemma mode

If `tv = epsilon_T/2`, the intended tight discounted denominator contains:

\[
1-\gamma+\gamma\,tv.
\]

The legacy implementation currently uses `1-gamma+tv`.

Main experiments remain:

```text
bound_form = linear
tail_mode = none
eps_tail = 0
```

with short-horizon proxy language. Either fix the legacy expression and test it numerically or disable it in experiment configs.

## P1.7 — Resolve `n_min=0` semantics

The configuration may allow `n_min=0`, while the production integer solver currently floors allocations at one.

Either:

- reject `n_min=0`; or
- define the objective and node-value behavior for `k_s=0` and implement it consistently.

## P1.8 — Revisit the pilot-factor startup restriction

Strict validation currently requires:

```text
pilot_branch_factor > max_default_branch_factor
```

when residual redistribution is enabled. This blocks configurations such as `tree_shape=[8,8,8]` with `pilot_branch_factor=8`.

Confirm whether this is a true algorithmic requirement or an obsolete engineering assumption. The bounded solver can still redistribute budget when nonredundant upper bounds permit expansion.

---

# 6. Required tests

## 6.1 CPU-safe behavioral tests

1. High-token-ID sticky-key test.
2. Unique request ID and pilot-continuation affinity test.
3. Dynamic snapshot propagation test.
4. Real server-version mismatch test.
5. Stored old-log-probability PPO denominator test.
6. Runtime terminal grading test with terminal rewards `[0,1]` and `C_terminal > 0`.
7. Unified context-bound test.
8. Exact bounded integer allocation test.
9. **Mixed-depth queue-flush test:** nodes from multiple depths are accepted and the exact flush budget is preserved.
10. `fresh_iid` pilot-discard and exact final-child test.
11. Weighted metadata candidate-to-actor test.
12. Terminal-cluster coverage test.
13. Weighted source-aware strictness test.
14. Two-level multiplicity semantics test.
15. No-critic worker-map test.
16. Stable edge-ID collision test.
17. Manifest-validity test.

Tests must execute behavior. Source-string assertions alone are not sufficient evidence.

## 6.2 Real-stack smoke tests

### Smoke A — SPO

- at least two successful actor updates;
- unique request IDs;
- stored generation log-probabilities align with actor tokens;
- no placeholder reward overhead failure;
- finite optimizer behavior.

### Smoke B — VDRA `fresh_iid`

- at least five successful actor updates;
- rollout/scorer server versions match every update;
- no terminal reward omission;
- no query/response truncation;
- exact queue-flush budgets;
- at least one logged mixed-depth flush when asynchronous timing produces one, without failure;
- exact final-child count unless explicitly token-capped;
- finite loss and gradients;
- replay rollback succeeds after one injected actor failure.

### Smoke C — VDRA `weighted_reuse`

Run only after P0.W1-P0.W3 pass.

- at least five successful actor updates;
- nonuniform multiplicities are observed and preserved;
- the selected descendant multiplicity semantics are verified;
- cluster-coverage fallback is exercised;
- weighted parent values match artifacts;
- weighted actor loss is finite;
- fallback trees are separately counted.

## 6.3 CI requirement

Add a CPU CI workflow and record GPU smoke evidence with:

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

- P0.1-P0.3 pass;
- Smoke A and Smoke B pass;
- rollout/scorer weights are server-verified;
- context bounds are validated from real dataset statistics;
- no unreported fallback occurs;
- the run manifest is valid.

Mixed-depth queue flushes do not block GO.

## 7.2 `weighted_reuse`

**GO as an efficiency ablation only when:**

- all common P0 items pass;
- P0.W1-P0.W3 pass;
- Smoke C passes;
- the paper states the selected local/path multiplicity semantics;
- the method is described as a representative approximation;
- fallback frequency and effective sample weights are reported.

## 7.3 Long-training prohibition

Do not launch long main-result runs solely because unit tests pass. Real-stack smoke evidence must first establish:

```text
correct server weights
correct behavior snapshots
correct token alignment
correct estimator semantics
exact queue-flush budgets
finite optimizer behavior
scientifically valid manifest
```

---

# 8. Implementation order

```text
1. implement a real rollout/scorer weight-version handshake
2. use one edge-prompt context limit end to end
3. run real SPO and fresh_iid async smoke tests
4. honor actual per-request sampling parameters
5. strengthen stable question/tree/edge/sticky identities
6. strengthen manifest validity and logging
7. choose local or path multiplicity semantics for weighted_reuse
8. enforce source-aware weighted metadata strictness
9. share weighted coverage fallback across runtimes
10. close scorer clients and support multi-node routing
11. replace broad TypeError protocol fallbacks
12. fix or quarantine legacy Simulation Lemma mode
13. resolve n_min=0 and pilot-factor restrictions
14. add CPU CI
15. run weighted_reuse smoke tests
```

Do **not** add a same-depth queue restriction. Mixed-depth queue membership is part of the intended asynchronous algorithm.

---

# 9. Experiment policy

## Primary comparison

```text
SPO
VDRA-fresh_iid
GRPO or the selected trajectory-level baseline
```

Primary plots:

- performance by successful optimizer update;
- performance by wall-clock training time;
- generated-token compute;
- pilot/support/scoring overhead;
- allocation statistics per queue flush;
- queue size, timeout/capacity flush reason, and depth composition.

Depth composition is reported to characterize asynchronous behavior, not to impose same-depth batches.

## Efficiency ablation

```text
VDRA-weighted_reuse
```

Report separately:

- cluster multiplicity distribution;
- selected descendant multiplicity semantics;
- fallback frequency;
- representative coverage;
- effective weighted sample count;
- compute saving relative to `fresh_iid`;
- accuracy difference relative to `fresh_iid`.

Do not merge `fresh_iid` and weighted/fallback trees under one unlabeled method name.

---

# 10. Final readiness checklist

```text
[ ] request IDs are unique and sticky keys preserve branch affinity
[ ] sticky keys include globally stable tree identity
[ ] token IDs above 255 are supported
[ ] dynamic snapshot reaches edge, rollout server, and scorer server
[ ] rollout/scorer versions are independently server-verified
[ ] terminal pilots are graded before dispersion
[ ] accumulated edge queries fit without truncation
[ ] stored behavior log-probabilities remain the PPO denominator
[ ] no critic worker is created
[ ] exact budget is preserved for every queue flush
[ ] mixed-depth flushes are accepted and logged correctly
[ ] fresh_iid reuses zero pilots
[ ] weighted metadata survives candidate to actor
[ ] weighted coverage includes terminal clusters
[ ] weighted descendant multiplicity semantics are documented and tested
[ ] allocation scope is per_queue_flush_within_tree everywhere
[ ] placeholder reward is not recomputed
[ ] edge IDs do not collide across postponed iterations
[ ] run manifest rejects scientifically invalid runs
[ ] CPU CI passes
[ ] SPO real-stack smoke passes
[ ] VDRA-fresh_iid real-stack smoke passes
[ ] VDRA-weighted_reuse real-stack smoke passes before weighted results are used
```

Only after the relevant checklist items pass should the repository be treated as ready for long AAAI 2027 experiment runs.
