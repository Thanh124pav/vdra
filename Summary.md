# VDRA: Value Dispersion-Guided Rollout Allocation for Segment-Level Policy Optimization

## 1. Central Research Question

We study the following problem:

$$
\boxed{
\text{Given a fixed rollout budget, how should branch factors be allocated across tree nodes?}
}
$$

Existing segment-level and tree-structured policy optimization methods commonly expand different nodes using the same branch factor. However, uniform allocation ignores substantial differences in the value distributions of their possible continuations.

For some nodes, different child continuations have nearly identical values. Generating additional branches from these nodes produces largely redundant samples. For other nodes, the values of possible continuations vary substantially, and more branches are required to obtain a reliable Monte Carlo estimate of the node value.

VDRA reduces the branch factor assigned to low-dispersion nodes and reallocates the saved rollout budget toward nodes whose continuation values are more dispersed.

The allocation mechanism does not require a specific definition of a node or a particular segmentation strategy. It can be integrated with fixed-length segmentation, semantic segmentation, or existing sequential tree-expansion methods.

---

## 2. Main Story of the Paper

The central argument of VDRA is:

$$
\boxed{
\text{Child-value dispersion}
\rightarrow
\text{Monte Carlo value-estimation variance}
\rightarrow
\text{Value-induced gradient error}
\rightarrow
\text{Rollout allocation}.
}
$$

For each node, VDRA estimates an upper bound on its child-value dispersion using short-horizon divergence between continuation distributions. It then solves a constrained resource-allocation problem to determine how many branches should be assigned to each node.

---

## 3. Value Dispersion and Monte Carlo Value Estimation

Consider a node $s$. Let

$$
U \sim \pi_\theta(\cdot \mid s)
$$

denote a child segment sampled from the current policy.

We define the intrinsic value dispersion of node $s$ as

$$
\boxed{
\sigma_s^2
:=
\operatorname{Var}_{U\sim\pi_\theta(\cdot\mid s)}
\left[V(U)\right].
}
$$

This quantity characterizes how much the values of possible continuations from $s$ differ.

Suppose that $k_s$ independent children are sampled:

$$
U_1,\ldots,U_{k_s}
\overset{\mathrm{iid}}{\sim}
\pi_\theta(\cdot\mid s).
$$

The value of node $s$ is estimated by

$$
\widehat V(s)
=
\frac{1}{k_s}
\sum_{i=1}^{k_s}V(U_i).
$$

Under conditional independence,

$$
\boxed{
\operatorname{Var}
\left(
\widehat V(s)\mid s
\right)
=
\frac{\sigma_s^2}{k_s}.
}
$$

Therefore:

- $\sigma_s^2$ represents the intrinsic difficulty of estimating the value of node $s$;
- $k_s$ represents the sampling resource allocated to node $s$;
- $\sigma_s^2/k_s$ is the remaining Monte Carlo value-estimation variance.

Nodes with larger child-value dispersion require more branches to achieve the same estimation accuracy.

---

## 4. Connection to Segment-Level Gradient Estimation

Let $p(s)$ denote the parent of segment $s$. The ideal gradient contribution associated with $s$ is

$$
g_s
=
\left(
V(s)-b_s
\right)H_s,
$$

where

$$
H_s
=
\nabla_\theta
\log \pi_\theta
\left(
s\mid p(s)
\right)
$$

and $b_s$ is a fixed baseline, such as the value estimate of the parent node.

In practice, the true value $V(s)$ is replaced by its Monte Carlo estimate:

$$
\widehat g_s
=
\left(
\widehat V(s)-b_s
\right)H_s.
$$

The gradient error induced specifically by value estimation is

$$
\widehat g_s-g_s
=
\left(
\widehat V(s)-V(s)
\right)H_s.
$$

Assume that the score-function second moment is bounded:

$$
\mathbb E
\left[
\|H_s\|_2^2
\right]
\le
G^2.
$$

Then

$$
\mathbb E
\left[
\|\widehat g_s-g_s\|_2^2
\right]
\le
G^2
\operatorname{Var}
\left(
\widehat V(s)\mid s
\right).
$$

Using the Monte Carlo variance expression,

$$
\boxed{
\mathbb E
\left[
\|\widehat g_s-g_s\|_2^2
\right]
\le
G^2
\frac{\sigma_s^2}{k_s}.
}
$$

VDRA does not claim to minimize the entire policy-gradient variance. Instead, it minimizes an upper bound on the component of the segment-level gradient error induced by value estimation.

The appropriate claim is:

> VDRA controls the value-induced component of segment-level gradient estimation error by reducing the Monte Carlo variance of node-value estimates.

---

## 5. Main Difficulty

The intrinsic value dispersion

$$
\sigma_s^2
=
\operatorname{Var}_{U\sim\pi_\theta(\cdot\mid s)}
\left[V(U)\right]
$$

cannot be computed accurately before the node has been sufficiently expanded.

This creates a circular resource-allocation problem:

1. We need to know which nodes have high value dispersion in order to allocate more rollouts to them.
2. However, accurately estimating value dispersion itself requires many rollouts.

VDRA addresses this problem by constructing a lightweight upper-bound proxy from short continuation samples and the likelihoods produced by the current policy.

---

## 6. Pairwise Representation of Value Dispersion

For two independent children

$$
U,U'
\overset{\mathrm{iid}}{\sim}
\pi_\theta(\cdot\mid s),
$$

the value dispersion satisfies

$$
\boxed{
\sigma_s^2
=
\frac{1}{2}
\mathbb E_{U,U'}
\left[
\left(
V(U)-V(U')
\right)^2
\right].
}
$$

Therefore, if a pairwise bound

$$
|V(U)-V(U')|
\le
B(U,U')
$$

is available, then

$$
\sigma_s^2
\le
\frac{1}{2}
\mathbb E_{U,U'}
\left[
B(U,U')^2
\right].
$$

Define

$$
\boxed{
C_s
:=
\frac{1}{2}
\mathbb E_{U,U'}
\left[
B(U,U')^2
\right].
}
$$

Then

$$
\boxed{
\sigma_s^2\le C_s.
}
$$

The resource-allocation problem can therefore be formulated using $C_s$, without observing the true continuation values.

---

## 7. Short-Horizon Divergence Assumption

For two child nodes $u_i$ and $u_j$, let

$$
P_i^L
\quad\text{and}\quad
P_j^L
$$

denote the distributions of their full continuations.

Let

$$
P_i^m
\quad\text{and}\quad
P_j^m
$$

denote the distributions of their first $m$ continuation tokens.

Define

$$
D_L^{ij}
=
D_{\mathrm{TV}}
\left(
P_i^L,P_j^L
\right)
$$

and

$$
D_m^{ij}
=
D_{\mathrm{TV}}
\left(
P_i^m,P_j^m
\right).
$$

Because the first $m$ tokens are a projection of the complete continuation,

$$
D_m^{ij}
\le
D_L^{ij}.
$$

VDRA assumes that the additional divergence emerging in the unobserved tail can be bounded as

$$
\boxed{
D_L^{ij}
\le
D_m^{ij}
+
\left(
1-D_m^{ij}
\right)
\epsilon_{\mathrm{tail}}.
}
$$

Here,

$$
\epsilon_{\mathrm{tail}}\in[0,1]
$$

measures the possibility that two continuations that appear similar within the first $m$ tokens diverge later.

Interpretation:

- Small $\epsilon_{\mathrm{tail}}$: reasoning modes are usually revealed within the short continuation.
- Large $\epsilon_{\mathrm{tail}}$: delayed branching is frequent and short-horizon divergence is less informative.

VDRA does not assume

$$
D_L^{ij}=D_m^{ij}
$$

and does not introduce an unknown proportionality constant of the form

$$
D_L^{ij}=cD_m^{ij}.
$$

---

## 8. From Divergence to Pairwise Value Difference

Let $f$ denote a value-difference bound obtained from a total-variation argument.

The main form used by VDRA is the linear bound

$$
f(D) = R_{\max} D,
$$

which is exact for a bounded terminal reward $V\in[0,R_{\max}]$:
$|\mathbb E_P[V]-\mathbb E_Q[V]|\le R_{\max}\,D_{\mathrm{TV}}(P,Q)$.
A discounted simulation-lemma form
$f(D)=R_{\max}\,\gamma D/\big((1-\gamma)(1-\gamma+D)\big)$ is retained only as
an ablation (`bound_form: simulation_lemma`); it is not part of the main
method.

Then

$$
|V(u_i)-V(u_j)|
\le
f\left(D_L^{ij}\right).
$$

Using the short-horizon tail bound,

$$
|V(u_i)-V(u_j)|
\le
f
\left(
D_m^{ij}
+
\left(
1-D_m^{ij}
\right)
\epsilon_{\mathrm{tail}}
\right).
$$

Define

$$
B_{ij}
=
f
\left(
D_m^{ij}
+
\left(
1-D_m^{ij}
\right)
\epsilon_{\mathrm{tail}}
\right).
$$

For $k_0$ pilot children with weights $q_i$, the node-level value-dispersion bound is

$$
\boxed{
C_s
=
\frac{1}{2}
\sum_{i=1}^{k_0}
\sum_{j=1}^{k_0}
q_iq_jB_{ij}^2.
}
$$

Under uniform pilot sampling,

$$
q_i=\frac{1}{k_0},
$$

and therefore

$$
\boxed{
C_s
=
\frac{1}{2k_0^2}
\sum_{i=1}^{k_0}
\sum_{j=1}^{k_0}
B_{ij}^2.
}
$$

---

## 9. Likelihood-Based Short-Horizon TV Estimator

VDRA estimates short-horizon total variation using likelihoods from the current policy.

For a pair of child nodes $u_i,u_j$, define

$$
P_i^m
=
P_\theta(\cdot\mid u_i)
$$

and

$$
P_j^m
=
P_\theta(\cdot\mid u_j).
$$

Let

$$
M_{ij}
=
\frac{1}{2}
\left(
P_i^m+P_j^m
\right).
$$

Total variation can be written as

$$
D_{\mathrm{TV}}
\left(
P_i^m,P_j^m
\right)
=
\mathbb E_{z\sim M_{ij}}
\left[
\frac{
\left|
P_i^m(z)-P_j^m(z)
\right|
}{
P_i^m(z)+P_j^m(z)
}
\right].
$$

Using the identity

$$
\frac{|a-b|}{a+b}
=
\left|
\tanh
\left(
\frac{\log a-\log b}{2}
\right)
\right|,
$$

we obtain

$$
D_m^{ij}
=
\mathbb E_{z\sim M_{ij}}
\left[
\left|
\tanh
\left(
\frac{
\log P_i^m(z)-\log P_j^m(z)
}{2}
\right)
\right|
\right].
$$

Suppose that $r$ samples are drawn from each distribution:

$$
Z_i
=
\left\{
z_{i,1},\ldots,z_{i,r}
\right\},
\qquad
z_{i,\ell}\sim P_i^m,
$$

and

$$
Z_j
=
\left\{
z_{j,1},\ldots,z_{j,r}
\right\},
\qquad
z_{j,\ell}\sim P_j^m.
$$

The Monte Carlo estimator is

$$
\boxed{
\widehat D_m^{ij}
=
\frac{1}{2r}
\sum_{z\in Z_i\cup Z_j}
\left|
\tanh
\left(
\frac{
\log P_i^m(z)-\log P_j^m(z)
}{2}
\right)
\right|.
}
$$

This estimator:

- does not require sampled sequences to be identical;
- uses likelihoods from the current policy;
- does not require an external embedding model;
- does not require a process Gaussian model;
- can be computed before full rollout completion;
- has Monte Carlo estimation error that decreases as the number of pilot samples increases.

Each sampled block $z$ must be scored under both distributions $P_i^m$ and $P_j^m$.

---

## 10. Rollout-Allocation Objective

Since

$$
\operatorname{Var}
\left(
\widehat V(s)\mid s
\right)
=
\frac{\sigma_s^2}{k_s}
$$

and

$$
\sigma_s^2\le C_s,
$$

we obtain

$$
\operatorname{Var}
\left(
\widehat V(s)\mid s
\right)
\le
\frac{C_s}{k_s}.
$$

For a set of nodes $\mathcal Q$ in the current allocation queue, VDRA solves

$$
\boxed{
\begin{aligned}
\min_{\{k_s\}_{s\in\mathcal Q}}
\quad&
\sum_{s\in\mathcal Q}
\frac{C_s}{k_s}
\\
\text{subject to}
\quad&
\sum_{s\in\mathcal Q}k_s
\le
B_{\mathcal Q},
\\
&
k_s\ge k_{\min}.
\end{aligned}
}
$$

Here:

- $B_{\mathcal Q}$ is the rollout budget assigned to the queue;
- $k_{\min}$ is a minimum branch allocation;
- the allocation floor prevents a node from being completely discarded because of an inaccurate proxy.

Ignoring the lower-bound constraint, the continuous relaxation has the solution

$$
\boxed{
k_s^\star
=
B_{\mathcal Q}
\frac{
\sqrt{C_s}
}{
\sum_{j\in\mathcal Q}\sqrt{C_j}
}.
}
$$

With an allocation floor, a practical formulation is

$$
\boxed{
k_s
=
k_{\min}
+
\left(
B_{\mathcal Q}
-
|\mathcal Q|k_{\min}
\right)
\frac{
\sqrt{C_s}
}{
\sum_{j\in\mathcal Q}\sqrt{C_j}
}.
}
$$

The continuous allocations are converted into integer branch factors using a budget-preserving rounding method, such as largest-remainder rounding.

The solution is optimal for the continuous relaxation of the current allocation batch. VDRA does not claim global optimality over all future nodes in the complete tree.

---

### 10.1 Pruning, Demand Caps, and Residual Reallocation

VDRA uses two distinct node signals. The predicted useful branch demand
$\widehat k_s^{\mathrm{need}}$ controls pruning and provides an upper demand
cap. The dispersion bound $C_s$ controls how saved branches are prioritized.
They must not be merged.

For default branch factor $n_s^{\mathrm{default}}$ and $k_{\min}=1$, define

$$
k_s^{\mathrm{cap}}=\max(k_{\min},\widehat k_s^{\mathrm{need}}),\qquad
k_s^{\mathrm{base}}=\min(n_s^{\mathrm{default}},k_s^{\mathrm{cap}}).
$$

The pruned budget and remaining demand are

$$
r_s=n_s^{\mathrm{default}}-k_s^{\mathrm{base}},\qquad
d_s=\max(k_s^{\mathrm{cap}}-k_s^{\mathrm{base}},0).
$$

Saved branches enter a shared residual pool. For a queue $\mathcal Q$, VDRA
solves the bounded problem

$$
\min_{\{k_s\}}\sum_{s\in\mathcal Q}\frac{C_s}{k_s}
\quad\text{subject to}\quad
k_s^{\mathrm{base}}\le k_s\le k_s^{\mathrm{cap}},\qquad
\sum_s k_s\le B_{\mathcal Q}.
$$

The continuous solution is capped water filling,

$$
k_s^\star=\operatorname{clip}
\left(\sqrt{C_s/\lambda_{\mathrm{dual}}},
k_s^{\mathrm{base}},k_s^{\mathrm{cap}}\right),
$$

where the internal dual variable $\lambda_{\mathrm{dual}}>0$ is chosen so
that

$$
\sum_s k_s^\star=
\min\left(B_{\mathcal Q},\sum_s k_s^{\mathrm{cap}}\right).
$$

This dual variable is not a public hyperparameter and is unrelated to the
historical threshold called `budget_lambda`. The VDRA allocation priority is
always $\sqrt{C_s}$; no term of the form $\sqrt{C_s-\lambda}$ is used.

Integer allocations use capped rounding (largest-remainder by default;
nearest-with-repair and stochastic rounding are ablations). Every node records
`default_k`, `predicted_k`, `base_k`, `saved_k`, `unmet_demand`,
`dispersion_C`, `additional_k`, and `allocated_k`.

**Pilot handling.** Pilots are processed in three groups:

1. *Shortcut pilots.* A pilot that terminates (EOS) inside the short first
   phase is a complete trajectory. It is excluded from TV estimation, attached
   directly as a graded leaf child, and counted against the node's branch
   budget. If a node has more terminal pilots than allocated branches, all of
   them are kept (`shortcut_overage`); discarding finished trajectories would
   waste generation.
2. *Duplicate pruning.* Among continuable pilots, a pair with short-horizon TV
   below the duplicate threshold is treated as redundant. VDRA repeatedly
   prunes the pilot with the most duplicate partners until no duplicate pair
   remains. `predicted_k` is the number of shortcut pilots plus surviving
   continuable pilots.
3. *Uniform random reuse.* When the node expands, the reused pilots are drawn
   uniformly at random (seeded, reproducible) from the post-pruning survivors —
   never ranked by likelihood. Missing branches are generated fresh.

Two caveats follow. First, residual redistribution requires
$k_0 > \max_s n_s^{\mathrm{default}}$, otherwise no node can report unmet
demand. Second, pruning and reuse mean the final child set is not an exactly
i.i.d. sample from $\pi_\theta(\cdot\mid s)$: duplicates are removed by
construction. VDRA treats this as a controlled limitation — the selection rule
carries no likelihood bias, but $\operatorname{Var}(\widehat V) = \sigma_s^2/k_s$
holds exactly only for the freshly generated branches.

### 10.2 Budget Reporting Modes

The default `fixed_main` mode holds the main-expansion branch budget fixed and
reports pilot generation (first phase and second-phase support blocks) and
likelihood scoring as additional compute. It must not be described as a
fixed-total-compute comparison.

The `fixed_total_generated` mode places pilot, support, and main-expansion
generation under one generated-token cap, set to the expected token count of
the uniform SPO tree with the same shape. When the cap is exhausted, remaining
nodes are graded as truncated leaves. Likelihood-scoring tokens and the
token-level forward-pass proxy are still reported separately. Experimental
claims must name the selected mode; the runtime records the mode, the cap, and
cap-hit counts in the run manifest.


## 11. Online Queue-Based Tree Expansion

During one rollout-generation phase, the policy parameters are frozen at $\theta_t$.

For each eligible node $s$:

1. Generate $k_0$ pilot children.
2. Route pilots that terminated (EOS) within the first phase directly into the
   training set as graded leaf children, counted against the branch budget; no
   TV is computed for them.
3. Generate short continuations of length $m$ for the continuable pilots.
4. Compute pairwise likelihood-based TV estimates.
5. Prune duplicate pilots by duplicate degree; compute the value-dispersion
   upper bound $C_s$.
6. Insert node $s$ into an allocation queue.
7. Trigger allocation when the queue is full or when its timeout is reached.
8. Allocate the remaining rollout budget using the VDRA allocation rule.
9. Expand the nodes: reuse a uniform random subset of surviving pilots, then
   generate only the missing branches.
10. Perform the policy update after rollout generation is completed.

Sibling frontier nodes are expanded concurrently, so nodes genuinely co-occupy
allocation queues; a serial builder would degenerate every queue flush to a
single node and disable the batchwise $\sqrt{C_s}$ rule.

The generation pipeline is

$$
\boxed{
\text{Frozen policy}
\rightarrow
\text{Pilot expansion}
\rightarrow
\text{Dispersion-bound estimation}
\rightarrow
\text{Queue allocation}
\rightarrow
\text{Full expansion}
\rightarrow
\text{Policy update}.
}
$$

To avoid stale scores, all nodes in the same queue should be generated under the same policy snapshot.

Queue timeout changes which nodes are optimized together, but it does not invalidate the batchwise allocation solution for the nodes currently contained in the queue.

---

# Method Novelty

## Novelty 1: Intrinsic Policy-Likelihood-Based Node Scoring

VDRA estimates node-specific continuation dispersion using likelihoods from the current policy.

Unlike approaches relying on an external process model or embedding model, the proposed score:

- is aligned with the current policy;
- changes online as the policy changes;
- requires no separately trained uncertainty model;
- can be evaluated during sequential tree construction.

---

## Novelty 2: Short-Horizon Divergence to Value-Dispersion Bound

VDRA establishes the following estimation chain:

$$
\boxed{
\text{Short-horizon policy divergence}
\rightarrow
\text{Full-horizon TV upper bound}
\rightarrow
\text{Pairwise value-difference bound}
\rightarrow
\text{Node value-dispersion bound}.
}
$$

This converts inexpensive short-continuation likelihood information into a resource-allocation score with an explicit connection to node-value estimation.

---

## Novelty 3: Value-Induced Gradient-Error-Aware Allocation

VDRA relates branch factor to Monte Carlo value-estimation variance through

$$
\operatorname{Var}
\left(
\widehat V(s)\mid s
\right)
=
\frac{\sigma_s^2}{k_s}.
$$

Under a bounded score-function moment condition, this quantity controls an upper bound on the value-induced segment-gradient error:

$$
\mathbb E
\left[
\|\widehat g_s-g_s\|_2^2
\right]
\le
G^2
\frac{\sigma_s^2}{k_s}.
$$

Thus, the allocation is motivated by policy-update quality rather than only by search diversity.

---

## Novelty 4: Closed-Form Pruning and Reallocation

VDRA formulates rollout allocation as

$$
\min_{\{k_s\}}
\sum_s
\frac{C_s}{k_s}
$$

under a fixed rollout budget.

The closed-form continuous solution

$$
\boxed{
k_s^\star
\propto
\sqrt{C_s}
}
$$

reduces expansion at low-dispersion nodes and reallocates their rollout budget toward nodes with larger estimated value dispersion.

---

## Novelty 5: Compatibility with Sequential Online Tree Construction

VDRA operates on frontier nodes that arrive sequentially during online tree expansion.

Its queue-based implementation:

- does not require all nodes to be known in advance;
- does not require prompt-level simultaneous candidates;
- computes node scores before full rollout completion;
- is compatible with segment-level and tree-structured online RL algorithms;
- can be applied to different trajectory-segmentation schemes.

---

# Claims of the Paper

## Claims that VDRA should make

1. VDRA minimizes an upper bound on Monte Carlo node-value estimation variance.
2. VDRA controls an upper bound on the value-induced component of segment-gradient error.
3. VDRA provides an intrinsic likelihood-based proxy for node value dispersion.
4. VDRA is compatible with different segmentation and tree-construction methods.
5. VDRA is batchwise optimal for the nodes contained in each allocation queue.
6. VDRA reduces redundant main-expansion generation under the declared budget mode.

## Claims that VDRA should avoid

1. VDRA minimizes the full policy-gradient variance.
2. Short-horizon divergence always equals full-horizon divergence.
3. The value-dispersion upper bound always preserves the exact ordering of true node variances.
4. VDRA provides a globally optimal allocation over the complete future tree.
5. A low-divergence node can never branch into different reasoning modes later.
6. Pilot estimation has zero computational cost.
7. The final child set is an exactly i.i.d. sample from $\pi_\theta(\cdot\mid s)$
   (duplicate pruning and pilot reuse remove redundant samples by construction;
   $\operatorname{Var}(\widehat V)=\sigma_s^2/k_s$ is exact only for freshly
   generated branches).

---

# Experimental Questions

## RQ1: Does VDRA improve task performance under a fixed rollout budget?

Compare VDRA under the declared budget mode. The default fixed_main comparison matches the main-expansion budget and reports pilot/scoring overhead separately. The fixed_total_generated option matches total generated tokens including pilots.

Report:

- Pass@1 or task accuracy;
- average reward;
- training stability;
- total generated tokens;
- wall-clock training time;
- number of expanded nodes;
- number of full rollouts;
- average branch factor by tree depth.

Baselines should include:

- uniform branch allocation;
- fixed branch factor;
- random non-uniform allocation;
- empirical reward-variance allocation;
- an external uncertainty-model allocation;
- VIP or another adaptive-rollout baseline;
- oracle allocation on a small evaluation subset.

Pilot-generation and likelihood-scoring costs must always be reported; fixed_total_generated also charges pilot generation to its generated-token cap.

---

## RQ2: Does short-horizon TV predict full-horizon TV?

For a representative set of nodes and child pairs, estimate

$$
\widehat D_m
$$

at different horizons

$$
m\in\{1,2,4,8,16,32,\ldots\}
$$

and compare it with a long-horizon or full-completion estimate

$$
\widehat D_L.
$$

Report:

$$
\operatorname{Spearman}
\left(
\widehat D_m,\widehat D_L
\right),
$$

$$
\operatorname{Pearson}
\left(
\widehat D_m,\widehat D_L
\right),
$$

and

$$
\widehat D_L-\widehat D_m.
$$

Since the allocator primarily relies on node ranking, Spearman correlation is especially important.

---

## RQ3: How large is the tail-divergence residual?

For each calibrated child pair, compute

$$
\widehat r_{ij}
=
\frac{
\left(
\widehat D_L^{ij}
-
\widehat D_m^{ij}
\right)_+
}{
1-\widehat D_m^{ij}+\delta
}.
$$

Estimate

$$
\widehat\epsilon_{\mathrm{tail}}
=
Q_{1-\alpha}
\left(
\left\{
\widehat r_{ij}
\right\}
\right),
$$

where $Q_{1-\alpha}$ is a high quantile, such as $0.90$, $0.95$, or $0.99$.

Evaluate:

- empirical coverage of the bound;
- bound tightness;
- variation across tree depths;
- variation across datasets;
- variation across model sizes;
- sensitivity of final performance to $\epsilon_{\mathrm{tail}}$.

Possible variants include:

$$
\epsilon_{\mathrm{tail}},
$$

$$
\epsilon_{\mathrm{tail}}(d),
$$

and

$$
\epsilon_{\mathrm{tail}}(h),
$$

where $d$ is tree depth and $h$ is the remaining generation horizon.

---

## RQ4: Does the proposed bound predict true value dispersion?

For a subset of nodes, generate a large number of complete continuations and construct an oracle estimate

$$
\sigma_{s,\mathrm{oracle}}^2.
$$

Compare this quantity with the proposed proxy $C_s$.

Report:

$$
\operatorname{Spearman}
\left(
C_s,
\sigma_{s,\mathrm{oracle}}^2
\right),
$$

$$
\operatorname{Pearson}
\left(
C_s,
\sigma_{s,\mathrm{oracle}}^2
\right),
$$

and the bound ratio

$$
\frac{
C_s
}{
\sigma_{s,\mathrm{oracle}}^2+\delta
}.
$$

A useful allocation proxy may be loose in absolute magnitude while still preserving the ranking of nodes.

---

## RQ5: Does VDRA improve node-value estimation?

Use a high-budget value estimate as a reference:

$$
V_{\mathrm{ref}}(s).
$$

Measure

$$
\operatorname{MSE}_V
=
\mathbb E
\left[
\left(
\widehat V(s)-V_{\mathrm{ref}}(s)
\right)^2
\right].
$$

Compare:

- uniform allocation;
- VDRA allocation;
- empirical-variance allocation;
- oracle allocation.

This is the most direct experiment for validating the optimization objective.

---

## RQ6: Does improved value estimation improve gradient quality?

Construct a high-budget reference gradient

$$
g_{\mathrm{ref}}.
$$

For each allocation method, measure

$$
\cos
\left(
\widehat g,g_{\mathrm{ref}}
\right),
$$

$$
\|\widehat g-g_{\mathrm{ref}}\|_2^2,
$$

and gradient variability across independent rollout seeds.

This experiment validates the claim that reducing value-estimation variance improves the value-induced component of gradient estimation.

---

## RQ7: Does pilot cost amortize over later tree expansion?

The pilot cost has three components:

$$
C_{\mathrm{pilot}}
=
\underbrace{k_0 L_1}_{\text{first-phase generation}}
+
\underbrace{k_0 r m}_{\text{support-block generation}}
+
\underbrace{k_0^2 r\,(L_{\mathrm{ctx}} + m)}_{\text{likelihood scoring (prefill)}},
$$

where:

- $k_0$ is the number of pilot children;
- $L_1$ is the first-phase pilot length;
- $r$ is the number of samples per continuation distribution;
- $m$ is the short-continuation length;
- $L_{\mathrm{ctx}}$ is the scored prefix length (every support block is
  scored under every pilot prefix).

First-phase generation is partially recovered because surviving pilots and
terminal (shortcut) pilots are reused as tree children. Scoring is prefill
compute and is reported separately from decode tokens.

If VDRA avoids $\Delta k$ unnecessary full continuations with expected remaining length $L_{\mathrm{rem}}$, then the immediate saved cost is approximately

$$
C_{\mathrm{save}}
\approx
\Delta kL_{\mathrm{rem}}.
$$

The allocation is computationally beneficial when

$$
C_{\mathrm{pilot}}
<
C_{\mathrm{save}}.
$$

For deep trees, removing an unnecessary branch may also prevent the generation of its descendants, resulting in additional savings.

Report:

- pilot-token overhead;
- likelihood-scoring overhead;
- number of avoided full rollouts;
- total saved tokens;
- total forward-pass cost;
- wall-clock speed;
- savings as a function of tree depth.

---

# Implementation Directions for Assumption Validation

## Direction A: Full-Completion Calibration

1. Randomly sample a subset of tree nodes during training.
2. Generate $k_0$ pilot children from each node.
3. Generate short continuations at multiple horizons.
4. Compute $\widehat D_m$ for every child pair.
5. Continue the same branches to completion.
6. Estimate $\widehat D_L$.
7. Compute empirical tail ratios.
8. Estimate $\epsilon_{\mathrm{tail}}$.
9. Compare $C_s$ with oracle value dispersion.

This is the most reliable validation approach, although it should be applied only to a limited calibration subset because of its compute cost.

---

## Direction B: Multi-Horizon Stabilization

Estimate divergence successively at

$$
m,\;2m,\;4m,\ldots
$$

until

$$
\left|
\widehat D_{2m}-\widehat D_m
\right|
\le
\eta.
$$

The first horizon satisfying this criterion can be treated as an adaptive lookahead length.

This method can reduce unnecessary lookahead computation, but it cannot fully exclude very late divergence. It should therefore be treated as an approximation or ablation rather than the sole validation of the assumption.

---

## Direction C: Repeated-Expansion Oracle

For a small subset of nodes:

1. Generate a large number of full child continuations.
2. Estimate oracle continuation values.
3. Compute oracle value dispersion.
4. Restrict the estimator to only $k_0$ pilot children.
5. Compute the proposed proxy $C_s$.
6. Repeat pilot selection across multiple random seeds.
7. Measure ranking accuracy and estimator variability.

This experiment determines how many pilot children and short-continuation samples are required for a stable allocation score.

---

## Direction D: Allocation-Regret Evaluation

Using oracle node variances, define

$$
J(\mathbf k)
=
\sum_s
\frac{
\sigma_{s,\mathrm{oracle}}^2
}{
k_s
}.
$$

The oracle continuous allocation is

$$
k_s^{\mathrm{oracle}}
\propto
\sigma_{s,\mathrm{oracle}}.
$$

Compare

$$
J
\left(
\mathbf k_{\mathrm{uniform}}
\right),
$$

$$
J
\left(
\mathbf k_{\mathrm{VDRA}}
\right),
$$

and

$$
J
\left(
\mathbf k_{\mathrm{oracle}}
\right).
$$

Define allocation regret as

$$
\mathcal R_{\mathrm{VDRA}}
=
J
\left(
\mathbf k_{\mathrm{VDRA}}
\right)
-
J
\left(
\mathbf k_{\mathrm{oracle}}
\right).
$$

This directly evaluates whether the proposed bound produces a better resource allocation than uniform branching.

---

# Required Ablation Studies

The main hyperparameters are:

$$
m:
\text{short-continuation length},
$$

$$
r:
\text{number of likelihood samples per distribution},
$$

$$
k_0:
\text{pilot branch factor},
$$

$$
k_{\min}:
\text{minimum branch allocation},
$$

$$
|\mathcal Q|:
\text{allocation-queue size},
$$

$$
T_{\mathrm{queue}}:
\text{queue timeout},
$$

and

$$
\epsilon_{\mathrm{tail}}:
\text{tail-divergence calibration}.
$$

The following structural ablations should be included:

1. Uniform allocation.
2. VDRA without tail correction.
3. VDRA with tail correction.
4. VDRA without an allocation floor.
5. VDRA without queue batching.
6. Empirical reward-variance allocation.
7. External-model-based node scoring.
8. Direct short-TV allocation without the simulation-lemma transformation.
9. Oracle value-dispersion allocation.
10. Different rounding strategies.
11. Different queue sizes and timeouts.
12. Global versus depth-dependent $\epsilon_{\mathrm{tail}}$.
13. Different pilot branch factors $k_0$.
14. Different short-continuation lengths $m$.
15. Different likelihood sample counts $r$.

---

# Suggested Contributions for the Introduction

Our main contributions are as follows:

1. We formulate node-level rollout allocation in segment-level policy optimization through the Monte Carlo variance of node-value estimates. We show that allocating $k_s$ branches to node $s$ reduces its value-estimation variance as

   $$
   \frac{\sigma_s^2}{k_s},
   $$

   which further controls an upper bound on the value-induced component of segment-level gradient estimation error.

2. We introduce an intrinsic likelihood-based estimator of short-horizon total variation between continuation distributions. The estimator uses likelihoods from the current policy and does not require an external process model or embedding model.

3. We derive a tail-corrected transformation from short-horizon policy divergence to an upper bound on pairwise continuation-value differences and, consequently, an upper bound $C_s$ on node value dispersion.

4. We formulate rollout allocation as minimizing

   $$
   \sum_s\frac{C_s}{k_s}
   $$

   under a fixed generation budget and obtain the closed-form batchwise allocation rule

   $$
   k_s^\star
   \propto
   \sqrt{C_s}.
   $$

5. We develop a queue-based implementation that enables adaptive pruning and rollout reallocation during sequential online tree construction while remaining compatible with different reasoning-segmentation strategies.

---

# Short Paper Summary

VDRA studies how a fixed rollout budget should be distributed across nodes in online tree-structured policy optimization. Uniform branch allocation ignores that some nodes have nearly identical continuation values while others contain highly variable continuations. VDRA characterizes the intrinsic estimation difficulty of a node by the variance of its child values,

$$
\sigma_s^2
=
\operatorname{Var}_{U\sim\pi_\theta(\cdot\mid s)}
\left[V(U)\right],
$$

and observes that allocating $k_s$ branches reduces the Monte Carlo node-value estimation variance to

$$
\frac{\sigma_s^2}{k_s}.
$$

Under a bounded score-function moment condition, this quantity also controls an upper bound on the value-induced component of segment-level gradient estimation error.

Because $\sigma_s^2$ is unavailable before full expansion, VDRA estimates a lightweight upper bound using short-horizon total variation between continuation distributions. The divergence is estimated directly from policy likelihoods and corrected by a tail-divergence term that accounts for reasoning modes that may separate only later in the continuation. A simulation-lemma-based value-difference bound then converts pairwise continuation divergence into a node-level value-dispersion bound $C_s$.

Given $C_s$, VDRA allocates rollout branches by solving

$$
\min_{\{k_s\}}
\sum_s
\frac{C_s}{k_s}
$$

under a fixed rollout budget, yielding the continuous allocation

$$
k_s^\star
\propto
\sqrt{C_s}.
$$

A minimum allocation floor prevents premature removal of nodes whose dispersion is underestimated. A queue-based implementation applies the allocation batchwise during sequential online tree expansion. VDRA therefore reallocates computation away from redundant low-dispersion nodes and toward nodes whose values are harder to estimate, without requiring an external uncertainty model or a specific trajectory-segmentation method.

---

# Proposed Paper Title

$$
\boxed{
\text{VDRA: Value Dispersion-Guided Rollout Allocation for Segment-Level Policy Optimization}
}
$$