# Pre-Server Sweep — Canonical Paper Objective (PLAN.md §1.2/§1.3)

Purpose: before the GPU smoke on the server, confirm that every
non-GPU logic/runtime error is caught locally, so only genuinely
GPU-dependent issues can remain. Environment: conda `deeplearning`,
torch 2.11.0+cu128, single GTX 1650 4GB (sm_75).

## Verified locally (green)

```text
bash scripts/pre_gpu_check.sh                     PRE_GPU_CHECK=PASS (20 steps)
root CPU suite (tests/)                            166 passed
gear_tree recipe suite                             all passed (+1 pre-existing xfail)
FSDP2 2-rank canonical parity harness              segment_mean PARITY (all scenarios)
2-rank dummy-padding no-hang + all-zero skip       passed
smoke_{a,b,c,d} Hydra compose + cross-level        validate cleanly
```

Canonical contract exercised end to end on CPU:

- `policy_aggregation=segment_mean` (default) and `token_mean` compute the
  paper objectives with the trainer-stamped pre-filter denominators
  (`M_B`, and for token_mean the mask-MATCHING `T_B_response` or
  `T_B_prob_mask`); the tree-balanced `1/(N_T·N_seg)` objective is a labeled
  ablation (`tree_balanced_segment_mean`).
- `use_prob_mask` is first-class in BOTH values with an authoritative
  `probability_mask_threshold` (strict `<`, default 0.9). One shared
  predicate (`recipe/gear_tree/prob_mask.py`) serves the actor mask and
  extraction-time active-token counting, so they cannot drift; the trainer
  propagates the resolved values to every rollout request.
- Dummy padding rows are masked EXPLICITLY and contribute to no
  denominator, numerator, or diagnostic under either `use_prob_mask` value.
- Logical batches carry a status (`trainable` / `all_zero_advantage` /
  `zero_active_tokens`); both skip reasons are reported separately and
  `expected_optimizer_steps` counts only TRAINABLE batches, so a mixed
  update is not marked accounting-invalid.
- Strict canonical sparse mode requires `entropy_coeff == 0`,
  `use_kl_loss == false`, `kl_loss_coef == 0` — sparse omission preserves
  the policy-gradient term exactly but not a dense auxiliary objective.
- Replay checkpoints persist the objective-mask identity (logical-record
  schema v2); restore fails fast on a mask/threshold mismatch or a legacy
  checkpoint lacking active-token counts, with an explicit
  `reset_replay_on_objective_mismatch` opt-out that DISCARDS the rows.
  Canonical runs additionally require the complete schema-v2 denominator
  metadata on every trainable edge and never recompute it silently; a
  checkpoint never claims v2 while holding incomplete records.
- Canonical logical-batch VDRA supports the POLICY-GRADIENT objective only:
  entropy/KL are rejected at startup regardless of `strict_vdra`, because
  their normalization is not yet guaranteed to be invariant to the
  micro-batch partitioning of a logical batch.
- Anti-livelock guards: `gear_tree.max_consecutive_skipped_updates` (50)
  aborts a run whose reservations keep carrying no learning signal (skips
  never advance `global_step`), and `gear_tree.max_rollout_iterations`
  stops a run whose `global_step` is stuck below `total_training_steps`.
  Both are disabled by `null`/`<= 0`.
- Every iteration records `last_iteration_status` (updated /
  all_zero_skipped / zero_active_skipped / mixed_zero_signal_skipped /
  postponed / no_sample / failed_before_actor / actor_failed); the legacy
  `actor_update_skipped` boolean is derived from it. Fully skipped
  iterations write a timing row (including cumulative and wall-clock time)
  and persist the manifest, with bookkeeping I/O failures logged rather
  than fatal — the replay reservation is already committed.
- Sparse tensor execution over the logical-slot ledger: advantages from the
  complete sibling set, zero-advantage segments become metadata-only slots
  that count toward reservation / caps / `target_edges_per_iteration` /
  divisibility / `M_B` / `T_B` but never tensor rows.
- Real 2-rank FSDP2 (`apply_fsdp2` + `DataParallelPPOActor`): local numerator
  over the global denominator with `loss_scale_factor = dp_size` reproduces
  the single-rank gradient exactly; dummy-padded logical batches do not hang
  collectives; all-zero logical batches skip their optimizer step on every
  rank consistently.
- Fail-fast (no silent fallback) on: missing `M_B`/`T_B` stamp, dp-size
  mismatch, sequence-parallel > 1, retired `global_segment_mean` name,
  retired `batch_slot_mean_ablation` flag, `token_mean` + `sum`, strict main
  without `only_adv_greater_than_zero=true`.

## Known environmental non-issues (NOT introduced here, not GPU-fixable locally)

```text
import recipe.gear_tree.main_gear_tree in a bare process raises
  ray.actor.ActorClassInheritanceException
```

This is structural and PRE-EXISTING (baseline de9ff6f already declares
`@ray.remote class GearTreeTaskRunner(TaskRunner)`). It only triggers when
the module is imported standalone without a Ray runtime; the real entrypoint
runs under `ray.init`. It is unrelated to the objective migration.

## Only the server GPU smoke can verify (expected remaining risk surface)

```text
FSDP1 (strategy="fsdp") parity — impossible on CPU under torch 2.11
FSDP2 / NCCL reducer on CUDA (the CPU gloo run proves the averaging
  contract and the dp_size compensation, not the CUDA kernels)
bf16 / mixed-precision reduction (GTX 1650 sm_75 has no bf16; CPU is fp32)
sequence-parallel (ulysses) interaction — canonical path rejects sp>1;
  enabling it needs a separate design
vLLM rollout + real model generation
multi-GPU throughput / memory
end-to-end smoke counters on a real model: rollout_iteration, global_step,
  scheduler/LR, num_optimizer_steps_total, selected edges, replay ages,
  vdra/all_zero_logical_batches, vdra/skipped_zero_gradient_updates,
  RunManifest.is_valid_main_run()
```

## Recommended server smoke command

```bash
MODEL_PATH=<small hf model> TRAIN=<train.parquet> VAL=<val.parquet> \
  STEPS=5 bash verl/recipe/gear_tree/run_smoke_matrix.sh smoke_d
```

### Required smoke report fields (per iteration)

```text
rollout_iteration
global_step
scheduler / LR step count
num_optimizer_steps_total
training/expected_optimizer_steps
train/logical_slots
train/trainable_tensor_rows
train/dummy_rows
train/real_response_tokens
vdra/all_zero_advantage_logical_batches
vdra/zero_active_token_logical_batches
vdra/skipped_zero_gradient_updates
vdra/prob_mask_active_token_fraction
train/mean_edge_age, train/max_edge_age
RunManifest.is_valid_main_run() verdict
```

A failure that is not in the "server GPU smoke" list above should be
treated as a regression to fix before the paper run. GPU/vLLM results must
be reported from the actual run — CPU tests never establish them.
