# H1 Report — Measured FSDP2 Distributed Semantics of the Gear-Tree Actor Losses

Status (original): **EVIDENCE COMPLETE — STOP** at commit `0e19dcd`; all
measurements below come from the test-only harness
`verl/recipe/gear_tree/tests/test_fsdp_canonical_parity.py`.

> RESOLUTION (2026-07-21): the user reviewed this evidence and chose a
> design beyond the Options A/B below — the CANONICAL objective now follows
> the paper (`policy_aggregation=segment_mean` default / `token_mean`), and
> the tree-balanced `1/(N_T·N_seg)` objective (whose measured non-parity is
> documented in §4.2) is demoted to the labeled `tree_balanced_segment_mean`
> ablation. The distributed handling is Option-A-style (trainer stamps the
> global pre-filter denominators `M_B`/`T_B`; dp_actor compensates the
> averaging reducer with `loss_scale_factor = dp_size`), extended with a
> sparse logical-slot ledger. The contract is recorded in PLAN.md §1.2/§1.3
> and implemented across commits H1-B1..H1-B6; the harness now asserts
> segment_mean distributed PARITY through the real production path. See
> `docs/pre_server_sweep_report.md` for the post-implementation status.

---

## 1. Environment

```text
host        WSL2 (Linux 6.6.87.2-microsoft-standard), conda env `deeplearning`
python      3.12, torch 2.11.0+cu128, gloo backend
GPU         1x GTX 1650 4GB (sm_75) — multi-rank GPU parity impossible here
world size  2 processes, CPU device mesh, REAL fully_shard (FSDP2)
```

Components exercised (all REAL production code, no mirrors):

```text
verl.utils.fsdp_utils.apply_fsdp2  (fully_shard wrap, CPU mesh)
DataParallelPPOActor.update_policy (real mini/micro loop, real _optimizer_step)
fsdp2_clip_grad_norm_              (DTensor grad-norm path)
edges_to_dataproto                 (real replay tensorization, stored old log-probs)
DataProto.chunk                    (contiguous production dispatch)
compute_policy_loss_vdra_segment_mean (both current production paths)
```

NOT exercised (server GPU-smoke items — see §6):

```text
FSDP1 (FullyShardedDataParallel)   — torch 2.11 CANNOT run it on CPU:
                                     "FSDP needs a non-CPU accelerator device"
                                     (pinned by test_fsdp1_cpu_unsupported_documented)
bf16 / mixed-precision reduction   — CPU fp32 only; GTX 1650 has no bf16 either
sequence parallel (ulysses)        — sp = 1 throughout
real multi-GPU NCCL reducers, vLLM rollout
```

## 2. Batch geometry (production-faithful)

```text
global optimizer data   512 rows  (edges_to_dataproto)
per-rank shard          256 rows  (DataProto.chunk(2), contiguous)
global mini-batch       128 rows  → local ppo_mini_batch_size 64
                        (fsdp_workers.py:1174 divides by dp world size)
local micro-batch       32 rows   (gradient_accumulation = 2)
optimizer steps         4 per update_policy (verified on every rank and cell)
optimizer               SGD lr=0.05, grad_clip 1e9 (no-op) → param_delta = -lr·grad
reference               single-rank PLAIN model; its k-th 128-row optimizer batch
                        is the UNION of both ranks' k-th local mini-batches
                        (row-aligned with what the distributed run processed)
```

Initial weights are bit-identical across the FSDP2 wrap and the reference
(seeded TinyLM; asserted).

## 3. Measured reducer behavior

**FSDP2's gradient reducer is EXACTLY the average over ranks.** In all
8 cells, the distributed step-0 gradient equals the plain average of the two
ranks' local-denominator gradients with **0.000e+00** deviation (bitwise,
same-shape kernels):

```text
g_dist = (1/W) * Σ_r g_local(r)        measured identity, every cell
```

There is **no world-size compensation anywhere in the actor path**
(`dp_actor.py` sets `loss_scale_factor = 1.0` for the VDRA modes and never
multiplies or divides by world size).

Tolerances: same-shape identities are exact (atol 1e-6, measured 0.0).
Comparisons against the reference cross micro-batch kernel shapes (32-row
vs 64-row fp32 GEMMs) and carry ≤ ~1.5e-6 absolute noise on gradients of
magnitude 1e-4..1e-3; those assertions use atol 5e-6. The semantic mismatch
below is ~5 orders of magnitude larger than this noise floor.

## 4. Results per loss mode

### 4.1 `segment_mean` (uniform slot mean, `L = Σ rows / N_B`, pre-filter N_B) — the PAPER objective

This is the current `batch_slot_mean_ablation` path with
`segment_token_reduction=mean`; it equals the paper's
`L = (1/M) Σ_u [token-mean of segment u]` with `M` = pre-filter slot count.

**Distributed parity holds in EVERY scenario** — per-step gradients, 4-step
parameter delta, per-step loss identity `(L_rank0 + L_rank1)/2 == L_ref`,
and per-step grad norms all match the single-rank reference:

| scenario (trees, order) | reduction | max step-grad rel-dev | verdict |
|---|---|---|---|
| balanced `[8]×64` contiguous | mean | 3.048e-03 (noise-floor) | PARITY |
| balanced `[8]×64` contiguous | sum  | 2.776e-03 (noise-floor) | PARITY |
| uneven `[4]×64 + [32]×8` | mean | 3.048e-03 (noise-floor) | PARITY |
| interleaved `[8]×64` | mean | 5.542e-03 (noise-floor) | PARITY |
| token-skew (1 vs 4 tokens/row) | mean | 3.517e-03 (noise-floor) | PARITY |

(The rel-dev column is dominated by small-norm bias tensors; absolute
deviations are ≤ ~1.5e-6 — pure fp32 kernel-shape noise, see §3.)

Why it is safe: each rank's denominator `N_B_local = len(local mini_batch)`
satisfies `W · N_B_local = N_B_union` **by construction** (slots shard
evenly), so the averaging reducer reproduces the global objective exactly —
independent of tree composition and token lengths.

### 4.2 `tree_balanced` (current canonical `w = 1/(N_T·N_seg)`, LOCAL `N_T`)

`dp_actor.py:438-442` computes `N_T = unique(tree_group_ids)` on the LOCAL
rank mini-batch. Measured:

| scenario | N_T local (r0, r1) | N_T union | measured distributed behavior |
|---|---|---|---|
| balanced | 8, 8 | 16 | PARITY (only because `W·8 == 16`) |
| uneven | 16, 2 | 18 | **MISMATCH**: step-0 grad deviates **49.55%** from its own single-rank reference; 4-step param-delta deviates **35.02%**; the distributed gradient equals EXACTLY the uniform `segment_mean` gradient (collapse identity: rank0 rows `1/(16·4)=1/64`, rank1 rows `1/(2·32)=1/64` → uniform `1/128` after averaging) |
| interleaved | 64, 64 | 64 | **`g_dist == g_ref / 2` exactly** — the naked 1/W factor (every rank counts all trees, reducer still divides by W) |

Conclusion: the tree-balanced objective is **not implemented correctly under
multi-rank FSDP for general dispatch**. It self-consistently holds only when
every rank×mini-batch cell contains exactly `N_T_union/W` whole trees.
Production replay makes no such guarantee (contiguous chunking, variable
tree sizes). This corroborates demoting it to a labeled ablation.

### 4.3 `token_mean` probe (paper option 1 — not yet in production)

Arithmetic on the REAL tensorized token-skew batch, combined with the
measured average reducer: if a future `token_mean` used the LOCAL pre-filter
token count as its denominator, the effective per-token weight would be
`1/(W·T_local)` instead of `1/T_union`, i.e. off by
`d = T_union/(W·T_local)`:

```text
token-skew scenario, every step:  rank0 T_local=64,  d = 2.5000
                                  rank1 T_local=256, d = 0.6250
balanced scenario: T_local = [160,160] per rank — equal only by coincidence
                   of the regular test length pattern, not a guarantee
```

**`token_mean` therefore REQUIRES the global pre-filter token count of the
original optimizer batch** (unlike `segment_mean`, whose local slot counts
are exact by construction). This is the one place a production change to
`dp_actor`/trainer plumbing is unavoidable if `token_mean` is adopted.

### 4.4 Denominator reconstructibility from edge-level replay (user gate)

The user required: stop if edge replay cannot reconstruct the exact
original denominators.

```text
segment_mean  M = len(mini_batch)     — exact, pre-filter (dense rows kept;
                                        zero-advantage filtering is compute-only
                                        inside the loss). NO CONFLICT.
token_mean    T = response_mask.sum() — exact pre-filter VALID-token count per
                                        row is carried by the tensorized batch
                                        itself; the ORIGINAL-batch total needs
                                        cross-rank aggregation (§5 options).
                                        NO CONFLICT in reconstructibility.
```

No silent substitution is needed for either objective.

## 5. Implementation design proposal (per the user's objective decision)

User decision 2026-07-20: canonical = paper objectives `token_mean` and
`segment_mean` (pre-filter denominators); `w = 1/(N_T·N_seg)` demoted to
labeled ablation `tree_balanced_segment_mean`; `segment_mean` must NOT be
described as Dr. GRPO. PLAN.md §1.3 must be updated before production edits.

### 5.1 Schema (reuses the EXISTING `policy_aggregation` key)

`tree_policy.policy_aggregation` already exists (enum today:
`global_segment_mean` = tree-balanced canonical, `legacy_token_mean` = SPO
baseline, `vdra_node_balanced` = ablation), duplicated-and-must-agree with
the actor config per the M5 pattern. Proposed enum after migration:

```text
segment_mean                (NEW — paper canonical; maps to loss_mode
                             vdra_segment_mean_ppo, uniform pre-filter slot mean)
token_mean                  (NEW — paper canonical option; same loss_mode,
                             global pre-filter token denominator)
tree_balanced_segment_mean  (RENAMED from global_segment_mean — labeled
                             ablation, the measured-non-parity 1/(N_T·N_seg) path)
legacy_token_mean           (unchanged — verl-native SPO baseline overlay;
                             distinct from the paper token_mean: verl's
                             token-mean normalizes per micro-batch locally)
vdra_node_balanced          (unchanged ablation)
```

- `PolicyLossConfig` gains `policy_aggregation` mirroring the same
  must-agree validation as `segment_token_reduction` today.
- Strict main triple becomes `spo / {segment_mean|token_mean} /
  vdra_segment_mean_ppo`.
- The old name `global_segment_mean` FAILS FAST with a rename message (no
  silent alias) — decision point Q4 below.
- `batch_slot_mean_ablation` flag becomes redundant (its math IS
  `segment_mean`): remove it and migrate its consumers/tests — decision
  point Q3 below.
- `token_mean` ignores `segment_token_reduction`; configuring
  `token_mean` + `sum` raises (no silently ignored knobs).
- A later `dr_grpo_fixed_length` (fixed max-length denominator) stays a
  separate future option; `segment_mean` is never labeled Dr. GRPO.

### 5.2 Loss math (all pre-filter, zero-filtering stays compute-only)

```text
segment_mean: L_local = Σ_{s ∈ retained} tokred(s) / N_B_local
              N_B_local = len(local mini_batch)  (pre-filter, fixed pre-split)
              loss_scale_factor = 1.0
              → distributed-correct as measured (W·N_B_local == N_B_union)

token_mean:   L_local = Σ_{s ∈ retained} Σ_t mask·L_t / T_union
              T_union = PRE-FILTER valid-token count of the ORIGINAL
                        (global 128-row) optimizer batch, fixed pre-split
              loss_scale_factor = world_size   (cancels the average reducer)
              → algebra: avg_r [ W·Σ_r/T_union ] = Σ_all/T_union  ✓
```

### 5.3 `token_mean` global-denominator plumbing — two options

**Option T1 (recommended): all-reduce in the actor.** In
`update_policy`, per mini-batch and BEFORE micro-splitting, compute
`T_local = mask.sum()` and `dist.all_reduce(SUM)` over the data-parallel
group to get `T_union`; pass it (plus `loss_scale_factor = world_size`)
into the loss. Fail fast with `NotImplementedError` when
`ulysses_sequence_parallel_size > 1` (choosing the correct group for sp is
a separate, unneeded-for-canonical design).

- Pros: robust to any dispatch layout; no trainer coupling; testable on
  CPU gloo with the existing harness. Cons: introduces one collective per
  mini-batch into `dp_actor` (touches the PLAN.md stop-list — which is
  exactly what this approval is for).

**Option T2: trainer stamps the denominator.** The gear trainer computes
`T_union(k)` for each global optimizer batch k (it knows the chunk layout
and normalized mini size) and stamps a per-row tensor
`optimizer_batch_token_count`; `dp_actor` just reads it.

- Pros: actor stays collective-free. Cons: silently breaks if verl's
  dispatch layout ever changes; duplicates layout arithmetic outside the
  place that owns it; more invasive DataProto surface.

### 5.4 Consumers to migrate (single coherent change-set)

```text
verl/verl/workers/config/actor.py        PolicyLossConfig field + validation
verl/recipe/gear_tree/policy_loss.py     dispatch on policy_aggregation;
                                         segment_mean = promoted slot path;
                                         token_mean = new denominator path;
                                         tree_balanced kept behind ablation label
verl/verl/workers/actor/dp_actor.py      token_mean denominator + loss_scale=W
                                         (only if token_mean adopted / T1)
verl/recipe/gear_tree/config_validation.py  new enum, strict triple, pair rules
verl/recipe/gear_tree/config/*.yaml      gear_tree_trainer default, smoke_d;
                                         smoke_a/b/c unchanged (legacy/node ablations)
verl/recipe/gear_tree/run_manifest.py + manifest tests   aggregation labels
scripts/check_hydra_composition.py       composed-value assertions
PLAN.md §1.3 (+ §1.1 canonical defaults) BEFORE any of the above (user order)
tests: test_segment_mean_loss, test_batch_slot_normalization,
       test_canonical_dataproto_no_objective_weights, test_policy_loss_config_wiring,
       test_distributed_grad_parity (ablation flag -> segment_mean),
       test_distributed_grad_scaling, test_zero_advantage_sparsity,
       test_fsdp_canonical_parity (assert parity of the NEW canonical paths),
       test_smoke_matrix_configs, test_manifest_* , test_gear_gate
```

Explicitly out of scope (still prohibited without separate approval):
`global_step`/`total_training_steps` units, scheduler cadence, checkpoint
naming, replay consumption, per-mini-batch zero skipping (H2/H3/H4).

## 6. What only the server GPU smoke can verify afterwards

```text
FSDP1 wrap parity (CPU-impossible), FSDP2 on CUDA/NCCL reducer,
bf16 mixed-precision reduction, vLLM rollout integration,
sequence-parallel interaction, real-model throughput
```

## 7. Decision points for the user (STOP — no production edit until answered)

```text
Q1  Default canonical aggregation for main runs / smoke_d:
    segment_mean (paper's stated aggregation; recommended) or token_mean?
Q2  token_mean plumbing: T1 actor all-reduce (recommended) or T2 trainer
    stamping — or defer token_mean entirely (implement schema + segment_mean
    now, token_mean in a follow-up)?
Q3  batch_slot_mean_ablation flag: remove and migrate consumers
    (recommended) or keep as a deprecated alias?
Q4  Old name global_segment_mean: fail-fast rename error (recommended) or
    accept as alias for tree_balanced_segment_mean?
```
