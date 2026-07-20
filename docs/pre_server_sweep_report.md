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
  paper objectives with the trainer-stamped pre-filter `M_B`/`T_B`
  denominators; the tree-balanced `1/(N_T·N_seg)` objective is a labeled
  ablation (`tree_balanced_segment_mean`).
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

Report the counters above per iteration; a failure that is not in the
"server GPU smoke" list above should be treated as a regression to fix
before the paper run.
