# PLAN.md — VDRA Tree Construction and Node-Balanced Policy Optimization

## 0. Baseline, objective, and stop condition

This plan supersedes the previous runtime-only checklist and is aligned to repository commit:

```text
31c9f30704541308d3a0f0f576166f57f257c730
```

Plan date: 2026-07-18.

The repository must be migrated from an SPO-compatible tree trainer with a global token-mean PPO loss to the canonical VDRA system:

```text
configurable segment definition
        +
VDRA online tree construction
        +
allocation-invariant node-balanced policy optimization
```

The central novelty remains online rollout allocation. The policy optimizer retains the PPO/GRPO-style clipped surrogate, but its aggregation must change because non-uniform branch factors must control estimator precision without unintentionally changing the importance of a reasoning state in the learning objective.

**Do not launch new long main-result runs until the node-balanced objective, group-preserving replay path, and real-stack smoke tests in this plan pass.**

Status labels:

- **DONE**: implemented and covered by behavioral tests.
- **PARTIAL**: plumbing exists but the deployed invariant is not established.
- **OPEN**: absent, inconsistent, or only covered by source-string tests.
- **LEGACY**: retained only for baseline or ablation use and not part of canonical VDRA.

---

# 1. Canonical method decomposition

Segment-level tree RLVR methods are represented by three design components:

1. **Segment definition** — how a response is partitioned into reasoning segments.
2. **Tree construction** — which expandable prefixes are expanded and how many continuation segments are generated from each prefix.
3. **Segment-level policy optimization** — how values, advantages, and segment losses from the resulting tree are aggregated into a policy update.

VDRA contributes primarily to component 2 and introduces a necessary adaptation to component 3.

VDRA does not require one fixed segmentation rule. It requires a minimal interface in which a segmenter/tree scaffold:

```text
- exposes expandable prefix states;
- can generate continuation segments from a prefix;
- exposes terminal/stopping information;
- permits the branch factor to be selected online.
```

Do not claim compatibility with literally every possible tree builder. Use:

> VDRA is segmentation-agnostic under an expandable-prefix interface.

The main implementation uses fixed-length SPO-style segments, but the allocator and policy aggregation must not contain assumptions that depend on fixed-length segmentation.

---

# 2. Canonical segment-tree representation

Follow the SPO-tree convention.

- The root represents the original prompt and has no generated segment.
- Every non-root node `n` represents one generated reasoning segment `z_n`.
- The node also stores the accumulated history ending at that segment:

\[
h_n = h_{\operatorname{Pa}(n)} \oplus z_n.
\]

- Expanding a prefix node `p` produces child segment nodes:

\[
\operatorname{Ch}(p)=\{c_{p,1},\ldots,c_{p,k_p}\}.
\]

A **parent decision group** is one expanded prefix and all realized continuation segments generated from it. The root may be a parent decision group even though it is not itself a segment.

Required stable identifiers:

```text
tree_id
parent_group_id
parent_segment_id          # null/sentinel for root
child_segment_id
queue_flush_id
child_index
policy_snapshot_id
```

The same logical parent group must keep the same `parent_group_id` from tree generation through replay, tensorization, microbatch packing, and actor loss reduction.

---

# 3. Canonical VDRA tree-construction contract

## 3.1 Online queue-flush allocation

At queue flush `tau`, let

\[
\mathcal Q_\tau=\{p_1,\ldots,p_m\}
\]

be the currently ready expandable prefixes. Mixed-depth queues are valid and intentional.

For each prefix `p`, VDRA estimates a local short-horizon dispersion proxy `C_p` and solves the bounded integer allocation problem

\[
\min_{\{k_p\}}
\sum_{p\in\mathcal Q_\tau}\frac{C_p}{k_p}
\]

subject to

\[
\sum_{p\in\mathcal Q_\tau}k_p\le B_\tau,
\qquad
\ell_p\le k_p\le u_p,
\qquad
k_p\in\mathbb Z_+.
\]

The realized spend must be

\[
B_\tau^{\mathrm{spent}}
=
\min\left(B_\tau,\sum_{p\in\mathcal Q_\tau}u_p\right)
\]

when all lower bounds are feasible. Do not require the impossible equality `sum(k_p) == B_tau` when hard upper caps make the budget infeasible.

Required per-flush invariants:

```text
sum(allocated_k) == spent_budget
spent_budget <= requested_budget
lower_bound_p <= allocated_k_p <= upper_bound_p
all prefixes use one frozen behavior-policy snapshot
allocation_scope == per_queue_flush_within_tree
mixed depths are accepted and logged
```

The hidden solver cap `max_k_per_node` must either be removed or exposed as an explicit configured upper bound `u_p` and reported in artifacts.

## 3.2 Main estimator: `fresh_iid`

Pilots and support blocks estimate `C_p` only. They are discarded before final expansion.

\[
c_{p,1},\ldots,c_{p,k_p}
\overset{iid}{\sim}
\pi_{\theta_t}(\cdot\mid h_p).
\]

The node value estimator is

\[
\widehat V(p)
=
\frac{1}{k_p}\sum_{j=1}^{k_p}\widehat V(c_{p,j}).
\]

Required invariants:

```text
pilot_children_reused == 0
pilot_children_shortcut == 0
pilot_children_discarded == pilot_children_generated
realized_trainable_children == allocated_k
sample_multiplicity == 1 for every final child
```

A naturally zero-advantage child is still a realized sample and remains in its parent group. Administrative placeholder rows marked `pruned=True` are not samples and must not enter the group denominator.

## 3.3 Proxy claim

The main configuration remains a short-horizon, model-intrinsic proxy:

```text
bound_form = linear
tail_mode = none
eps_tail = 0
```

Use paper language such as:

```text
short-horizon value-dispersion proxy
simulation-lemma-inspired allocation signal
```

Do not claim that the main configuration is a certified full-horizon value bound.

## 3.4 Optional `weighted_reuse`

`weighted_reuse` remains an efficiency ablation. Its multiplicity is local to the parent group unless a separate path-multiplicity objective is explicitly implemented.

For local multiplicities `m_{p,j}`:

\[
\widehat V(p)
=
\frac{\sum_j m_{p,j}\widehat V(c_{p,j})}
{\sum_j m_{p,j}}.
\]

Multiplicity changes the empirical distribution inside one parent group. It must not increase that parent's weight relative to other parent groups.

Replace the overloaded field name `edge_weight` with an explicit field:

```text
sample_multiplicity
```

Canonical `fresh_iid` always uses `sample_multiplicity = 1`.

---

# 4. New policy-optimization contract

## 4.1 Retained backbone

Retain:

```text
- stored generation-time old log probabilities;
- probability masking when enabled;
- PPO-style importance ratio and clipping;
- SPO-style local segment advantage in the main instantiation.
```

For child segment `c_{p,j}` generated from prefix `p`, the main advantage remains

\[
\widehat A_{p,j}
=
\widehat V(c_{p,j})-\widehat V(p),
\]

or its configured normalized variant.

The novelty is not a new clipping operator. The required change is the aggregation of segment contributions from a non-uniform tree.

## 4.2 Problem with the current global token mean

The current `treetune_ppo` path computes one masked mean over all active tokens. Consequently:

- longer segments receive more weight than shorter segments;
- parent groups with more children receive more weight than parent groups with fewer children;
- adaptive branch allocation changes both estimator precision and optimization importance.

Under VDRA, `k_p` is selected to improve estimation quality. It must not automatically make prefix `p` more important in the policy objective.

## 4.3 Canonical hierarchical reduction

Let `M_{p,j,t}` be the final active-token mask after response masking and optional probability masking. Let `ell_{p,j,t}` be the clipped PPO surrogate loss for token `t`.

### Stage 1 — token mean within one child segment

\[
L_{p,j}
=
\frac{\sum_t M_{p,j,t}\,\ell_{p,j,t}}
{\max(1,\sum_t M_{p,j,t})}.
\]

An actual child with an empty active-token mask contributes `L_{p,j}=0` and remains in the parent denominator. Log the empty-mask count.

### Stage 2 — child mean within one parent decision group

For `fresh_iid`:

\[
L_p
=
\frac{1}{k_p}
\sum_{j=1}^{k_p}L_{p,j}.
\]

For optional local `weighted_reuse`:

\[
L_p
=
\frac{\sum_j m_{p,j}L_{p,j}}
{\sum_j m_{p,j}}.
\]

### Stage 3 — parent-group mean within one tree

Let `P(T)` be the set of expanded parent decision groups whose child segments enter training, including the root group when applicable.

\[
L_T
=
\frac{1}{|P(T)|}
\sum_{p\in P(T)}L_p.
\]

### Stage 4 — tree/prompt mean within the actor update batch

For a batch of complete trees `B`:

\[
L_{\mathrm{VDRA}}
=
\frac{1}{|B|}
\sum_{T\in B}L_T.
\]

This is the canonical main-paper loss aggregation.

## 4.4 Queue decomposition and Jensen connection

If queue flushes partition the expanded parent groups of a tree,

\[
P(T)=\biguplus_{r=1}^{R}\mathcal Q_r,
\]

then

\[
L_T
=
\sum_{r=1}^{R}
\frac{|\mathcal Q_r|}{|P(T)|}
\left(
\frac{1}{|\mathcal Q_r|}
\sum_{p\in\mathcal Q_r}L_p
\right).
\]

The queue coefficient is the number of parent decision groups in the queue divided by the total number of parent decision groups in the tree. It is **not** the number of child edges in the queue divided by the total number of edges; the latter collapses back to the legacy edge mean.

Define the parent-level gradient estimate

\[
\widehat g_p
=
\frac{1}{k_p}
\sum_{j=1}^{k_p}
\widehat A_{p,j}H_{p,j}.
\]

The tree gradient is

\[
\widehat g_T
=
\frac{1}{|P(T)|}
\sum_{p\in P(T)}\widehat g_p.
\]

Using Jensen's inequality and the value-estimation error bound gives the motivating form

\[
\mathbb E\|\widehat g_T-g_T\|_2^2
\le
\frac{G^2}{|P(T)|}
\sum_{p\in P(T)}\frac{C_p}{k_p}.
\]

The implementation must match this parent-balanced estimator before the paper uses this derivation.

---

# 5. Required code migration

## P0.N1 — Add complete grouping metadata at tree generation

Targets:

```text
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/tree_rollout.py
verl/recipe/gear_tree/tree_logging.py
```

Every realized child segment must carry:

```text
tree_id
parent_group_id
parent_segment_id
child_segment_id
queue_flush_id
child_index
allocated_k
sample_multiplicity
policy_snapshot_id
```

Every tree artifact must also record:

```text
expanded_parent_group_count
trainable_child_count
queue_to_parent_group_counts
```

Acceptance conditions:

1. IDs are stable and unique across postponed rollout iterations.
2. All children of one prefix share exactly one `parent_group_id`.
3. `allocated_k` equals the number of realized trainable children in `fresh_iid`.
4. Root-generated segments are assigned to an explicit root parent group.

## P0.N2 — Preserve metadata through tree-to-edge extraction

Target:

```text
verl/recipe/gear_tree/tree_advantage.py
```

Required changes:

- propagate all grouping fields to every edge row;
- separate `sample_multiplicity` from any optimization coefficient;
- stop using `edge_weight` as a generic field;
- retain all actual final children even when their advantage is zero;
- exclude only administrative placeholder/pruned rows from the parent denominator;
- add strict validation that a fresh parent group has exactly `allocated_k` child rows.

Keep advantage computation separate from aggregation configuration:

```text
advantage_mode = spo_local | configured ablation
policy_aggregation = legacy_token_mean | vdra_node_balanced
```

Do not encode policy aggregation through `tree_update_mode`.

## P0.N3 — Remove misleading TreeRL/TreePO reproduction names

Targets:

```text
verl/recipe/gear_tree/gear_core/tree_update_modes.py
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
tests for tree update modes
```

Current `treepo_original` and `treerl_original` modes are not reproductions of the official methods.

Required action:

```text
- remove them from main configs; or
- rename them to treepo_style_ablation and treerl_style_ablation;
- add comments that they are scalar objective ablations, not official baselines.
```

The main VDRA path uses `advantage_mode=spo_local` until another estimator is implemented faithfully.

## P0.N4 — Add group tensors to DataProto

Targets:

```text
verl/recipe/gear_tree/tree_advantage.py
verl/recipe/gear_tree/gear_ray_trainer.py
```

Tensorization must produce at least:

```text
tree_group_ids:          int64 [batch]
parent_group_ids:        int64 [batch]
queue_group_ids:         int64 [batch]
allocated_k:             int64 [batch]
sample_multiplicity:     float32 [batch]
```

Do not broadcast group IDs as floating token tensors. Keep them as row-level tensors/non-tensor metadata accepted by the actor path.

Validation before the actor update:

```text
all rows with one parent_group_id share tree_group_id
all rows with one parent_group_id share allocated_k
fresh_iid groups have row_count == allocated_k
fresh_iid multiplicities are all one
no parent group is split or partially dropped
```

## P0.N5 — Implement a dedicated node-balanced PPO loss

Targets:

```text
verl/recipe/gear_tree/policy_loss.py
verl/verl/workers/actor/dp_actor.py
```

Register a separate loss mode:

```text
policy_loss.loss_mode = vdra_node_balanced_ppo
```

Do not silently change `treetune_ppo`; it must remain available as the SPO/legacy baseline.

Implementation order inside the new loss:

```text
1. compute PPO clipped token surrogate exactly as the retained backbone;
2. apply response/probability masks;
3. reduce active tokens to one scalar per child segment;
4. reduce children to one scalar per parent_group_id;
5. reduce parent groups to one scalar per tree_group_id;
6. average complete tree scalars.
```

The new loss must return the same diagnostic tuple expected by VERL, while additionally exposing metrics for each reduction stage.

Do not emulate the hierarchy by passing `1/k_p` into the current normalized `_weighted_masked_mean`. Normalizing again by the sum of token weights generally produces a different objective, especially with variable segment lengths.

## P0.N6 — Keep complete groups through replay and minibatching

Targets:

```text
verl/recipe/gear_tree/replay_buffer.py
verl/recipe/gear_tree/gear_ray_trainer.py
actor mini/microbatch packing path
```

The current edge-wise replay path may split siblings or trees. The canonical hierarchical loss requires complete groups.

Required behavior:

```text
- reserve/commit/rollback complete trees, not arbitrary individual edges;
- target_edges_per_update is a soft threshold;
- add complete trees until the threshold is met or exceeded;
- never enforce max_edges_per_question by cutting a tree or parent group;
- pack complete trees into actor minibatches;
- never split a parent group across microbatches;
- preferably keep a complete tree in one microbatch;
```

If a tree is too large for one microbatch, implement an exact coefficient-preserving accumulation path and prove parity with the full-tree reference reduction. Do not fall back silently to averaging microbatch scalar losses.

The SPO replay age policy may remain unchanged. This item concerns grouping correctness, not an on-policy replay claim.

## P0.N7 — Configuration and logging

Target:

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
```

Add explicit configuration:

```yaml
tree_policy:
  advantage_mode: spo_local
  policy_aggregation: vdra_node_balanced
  include_root_parent_group: true
  strict_group_integrity: true
```

Keep baseline configuration:

```yaml
tree_policy:
  policy_aggregation: legacy_token_mean
```

Required runtime metrics:

```text
vdra/parent_groups_per_tree
vdra/children_per_parent_mean
vdra/children_per_parent_std
vdra/empty_token_mask_children
vdra/group_integrity_failures
vdra/tree_split_count
vdra/parent_split_count
vdra/queue_parent_mass_sum
vdra/parent_weight_sum_per_tree
vdra/child_weight_sum_per_parent
vdra/effective_segment_weight_vs_branch_factor_corr
```

Strict main runs require:

```text
parent_split_count == 0
group_integrity_failures == 0
parent_weight_sum_per_tree == 1
child_weight_sum_per_parent == 1
```

## P0.N8 — Manifest contract

Add:

```text
policy_aggregation
advantage_mode
segment_definition
complete_tree_replay
complete_parent_microbatches
node_balanced_invariants_passed
```

A main run is invalid when:

```text
policy_aggregation != vdra_node_balanced
any parent group is partial
any tree reduction uses an undocumented fallback
fresh_iid row_count != allocated_k
node-balanced weights fail normalization
```

Legacy SPO runs remain valid baseline runs but must be labeled separately.

---

# 6. Old-code fixes retained from the previous plan

These remain required but are no longer the conceptual center of the plan.

## P0.R1 — Real rollout/scorer weight identity

The scorer and rollout replicas need an independently reported, changing weight-update version. A static model ID or a version fetched from the same scorer endpoint twice is not sufficient.

Strict VDRA runs must verify:

```text
rollout_server_weight_version == scorer_server_weight_version
```

before pilot likelihood scoring.

## P0.R2 — One context-length contract

Use the same resolved edge-prompt limit for startup validation, edge tensorization, actor input, rollout input, and model-context checks.

## P0.R3 — Feasible bounded allocation

Use `sum(k_p) <= B_tau` or an explicit residual slack. Do not raise merely because hard upper bounds prevent spending the full requested budget.

## P1.R4 — Actual per-request sampling parameters

`TreeAgentLoop.run(sampling_params, ...)` must honor and log the supplied request parameters. Static worker defaults cannot silently replace evaluation or ablation sampling settings.

## P1.R5 — Quarantine legacy Simulation Lemma mode

Main experiments use the linear short-horizon proxy. Disable the legacy discounted mode until its range, denominator, and reward scaling are correct and tested.

## P1.R6 — Remove obsolete pilot-factor restriction

Do not require `pilot_branch_factor > max_default_branch_factor` unless a mathematical necessity is documented. Configurations such as `888` with pilot factor `8` must not be rejected only by an obsolete residual-budget assumption.

## P1.R7 — Freeze unsupported paths

Until explicitly repaired and tested, keep the following out of main claims:

```text
weighted_reuse
depth_batch runtime
treerl_style_ablation
treepo_style_ablation
legacy simulation_lemma bound
```

---

# 7. Required behavioral tests

## 7.1 Pure reduction tests

Add a direct reference implementation in tests and verify the production loss against it.

1. **Uniform parity:** equal branch counts and equal segment lengths reproduce the expected legacy average.
2. **Non-uniform separation:** parent A with one child loss `2` and parent B with three child losses `4,4,4` gives node-balanced loss `3`, not edge-balanced loss `3.5`.
3. **Length invariance:** duplicating tokens inside a child without changing its token-mean loss does not change its parent weight.
4. **Child duplication invariance:** duplicating an identical child and updating `allocated_k` does not change that parent's loss.
5. **Parent balance:** changing another parent's branch factor does not change the first parent's tree-level weight.
6. **Queue decomposition:** direct tree reduction equals the weighted queue reduction using `|Q_r|/|P(T)|`.
7. **Wrong queue coefficient rejection:** an edge-count coefficient is shown not to reproduce the parent-balanced objective.
8. **Root group:** root-generated child segments are included exactly once when configured.
9. **Zero advantage:** a real zero-advantage child counts in the child denominator.
10. **Placeholder pruning:** an administrative pruned row does not count as a realized child.
11. **Empty probability mask:** an actual child with no active tokens contributes finite zero and remains in the denominator.
12. **Weighted reuse locality:** multiplicity changes only the within-parent empirical mean, not the parent weight in the tree.
13. **Permutation invariance:** row ordering does not change the result.
14. **Gradient parity:** autograd gradients match a small explicit hierarchical reference computation.

## 7.2 Metadata and replay tests

1. Tree generation creates unique stable tree/parent/child IDs.
2. Metadata survives tree -> edge -> replay -> DataProto -> actor.
3. `fresh_iid` parent group row count equals `allocated_k`.
4. Replay reservation never returns a partial parent group.
5. Replay reservation never cuts a tree to satisfy an edge threshold.
6. Rollback restores a complete tree group after an injected actor failure.
7. Minibatch packing never splits a parent group.
8. Full-batch and packed-microbatch losses/gradients are numerically equal.
9. Two trees for the same question remain distinct through replay.
10. Run manifest becomes invalid after any group-integrity failure.

## 7.3 Allocation/runtime tests

1. Mixed-depth queue flushes are accepted.
2. Infeasible requested budget spends `min(B_tau, sum upper bounds)` without raising.
3. Hidden max branch cap is absent or appears in configured/logged upper bounds.
4. `fresh_iid` discards all pilots and realizes exactly the allocated final children.
5. Rollout/scorer weight-version mismatch fails strict mode.
6. Deepest legal segment query tensorizes without truncation.

Source-string assertions are not sufficient for these contracts.

---

# 8. Real-stack smoke matrix

## Smoke A — SPO legacy baseline

```text
allocation = fixed/uniform
policy_aggregation = legacy_token_mean
```

Require at least two successful actor updates and finite losses. This preserves a reference baseline.

## Smoke B — VDRA construction with legacy aggregation

```text
allocation = VDRA
policy_aggregation = legacy_token_mean
```

This is an ablation only. It establishes the isolated effect of adaptive construction under the old optimizer.

## Smoke C — Uniform construction with node-balanced aggregation

```text
allocation = fixed/uniform
policy_aggregation = vdra_node_balanced
```

This isolates the policy-aggregation change.

## Smoke D — Full VDRA

```text
allocation = VDRA
policy_aggregation = vdra_node_balanced
pilot_execution_mode = fresh_iid
```

Require at least five successful actor updates and:

```text
finite loss and gradients
no parent/tree split
exact parent and tree normalization
verified rollout/scorer weights
stored old log probabilities used
no truncation
legal mixed-depth flushes
feasible budget accounting
exact fresh_iid final-child accounting
valid manifest
```

The four-way matrix is required for the paper ablation:

| Allocation | Policy aggregation | Role |
|---|---|---|
| Uniform | Legacy token mean | SPO-style baseline |
| VDRA | Legacy token mean | Construction-only ablation |
| Uniform | Node-balanced | Optimization-only ablation |
| VDRA | Node-balanced | Full method |

---

# 9. Experimental contract after migration

## RQ1 — Overall effectiveness and compute efficiency

Report performance by:

```text
successful optimizer update
wall-clock time
generated continuation tokens
pilot/support/scoring overhead
```

## RQ2 — Compatibility across segment definitions

For every tested segmentation/scaffold, run the controlled pair:

```text
same segmentation + uniform construction + same node-balanced update
same segmentation + VDRA construction + same node-balanced update
```

Do not call these full TreeRL or TreePO reproductions unless their native construction and credit-assignment algorithms are actually implemented.

## RQ3 — Allocation quality

Under one fixed segmentation and one fixed node-balanced optimizer, compare:

```text
fixed branch factor
random feasible allocation
simple uncertainty heuristic
empirical-variance allocation
VDRA
```

## RQ4 — Proxy and estimation validation

Validate:

```text
C_p versus empirical child-value dispersion
value-estimation MSE versus budget
value-induced gradient error versus a high-budget reference
```

Use correlation/MSE language, not “experimental proof of the theory.”

## RQ5 — Component ablation

Use the four-way allocation/aggregation matrix in Section 8, then separately test pilot size, budget, segment length, and queue timeout as sensitivity analyses.

---

# 10. Implementation order

```text
1. add canonical tree/parent/child/queue grouping metadata
2. preserve grouping metadata through edge extraction and DataProto
3. separate sample_multiplicity from optimization weighting
4. rename/remove misleading TreeRL/TreePO original modes
5. implement the standalone vdra_node_balanced_ppo reference reduction
6. add exact pure-loss and gradient-parity tests
7. make replay reserve complete trees and parent groups
8. add group-aware minibatch/microbatch packing
9. connect the new loss to dp_actor and trainer configuration
10. add normalization/group-integrity metrics and manifest fields
11. fix infeasible exact-budget behavior and expose all upper caps
12. implement real rollout/scorer weight-version verification
13. run the four smoke configurations
14. run the four-way allocation/aggregation ablation
15. only then launch long main experiments
```

Do not begin by adding more tree-builder variants. First make the canonical VDRA construction and policy objective mathematically and operationally identical.

---

# 11. Final go/no-go checklist

```text
[ ] non-root node == generated segment is used consistently
[ ] root parent group is explicit
[ ] tree_id, parent_group_id, child_segment_id, queue_flush_id are stable
[ ] fresh_iid realizes exactly allocated_k children per parent
[ ] token -> child -> parent -> tree reduction is implemented directly
[ ] queue mass uses parent-group count, not edge count
[ ] sample multiplicity is not used as global parent importance
[ ] legacy token mean remains available only as baseline/ablation
[ ] misleading TreeRL/TreePO original names are removed or renamed
[ ] replay never returns a partial parent group/tree
[ ] microbatch packing preserves exact hierarchical reduction
[ ] production gradients match the explicit reference implementation
[ ] mixed-depth queues remain legal
[ ] bounded allocation handles unspent budget caused by upper caps
[ ] rollout/scorer weight versions are independently verified
[ ] context limits are unified
[ ] actual sampling parameters are honored
[ ] all node-balanced normalization metrics pass
[ ] full VDRA real-stack smoke passes
[ ] four-way allocation/aggregation ablation is runnable
[ ] main-run manifest records the complete scientific contract
```

Only after the relevant checklist items pass should the repository be treated as ready for long AAAI 2027 VDRA experiments.
