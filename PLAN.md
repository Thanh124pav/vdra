# PLAN.md — Pre-GPU Correctness Checklist

## Goal

Complete every task below before launching a GPU smoke run. After the checklist passes, no known algorithmic, batching, replay, metadata, config, or CPU-integration blocker should remain. GPU smoke may still expose CUDA, Ray, vLLM, FSDP, memory, or distributed-runtime issues.

Canonical main path:

```text
fixed-length SPO-style segments
+ VDRA online allocation
+ fresh_iid final children
+ SPO local segment advantage
+ allocation-invariant node-balanced PPO
```

Do not launch a long training run until `scripts/pre_gpu_check.sh` and CPU CI both pass.

---

## P0.1 — Preserve the correct child sample set

**Targets**

```text
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
verl/recipe/gear_tree/tree_advantage.py
```

**Required changes**

- Set the main VDRA config to:

```yaml
only_adv_greater_than_zero: false
```

- Keep every realized child, including children whose final advantage is zero.
- Exclude administrative `pruned=True` placeholder rows from training and from the parent denominator.
- In `fresh_iid`, require:

```text
number of realized training rows for parent p == allocated_k[p]
sample_multiplicity == 1 for every realized child
```

**Acceptance tests**

- A zero-advantage realized child remains in the parent group.
- A pruned placeholder does not enter `DataProto` or the loss.
- Removing either behavior causes `validate_group_integrity` to fail.

---

## P0.2 — Make every tree instance globally unique

**Targets**

```text
verl/recipe/gear_tree/tree_rollout.py
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/tree_advantage.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/replay_buffer.py
```

**Required changes**

- Create one `tree_instance_id` when a stochastic tree is started.
- The ID must distinguish repeated trees for the same question and policy snapshot. Include at least:

```text
policy_snapshot_id
rollout_iteration
stable_question_id
per-tree UUID or monotonic counter
```

- Use `tree_instance_id` as `tree_id` throughout tree generation, edge extraction, replay, tensorization, and manifest logging.
- Derive `parent_group_id` and `edge_id` from this unique tree ID.
- `ReplayBuffer.add` must raise on a duplicate `edge_id`; never overwrite silently.

**Acceptance tests**

- Two trees for the same question and snapshot have different IDs.
- Their edges coexist in replay without collision.
- IDs survive JSON checkpoint save/load unchanged.

---

## P0.3 — Precompute exact objective weights on the full update batch

**Targets**

```text
verl/recipe/gear_tree/tree_data.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/policy_loss.py
```

**Required changes**

For every realized child `j` of parent `p` in tree `T`, compute before actor batching:

\[
w_{p,j}
=
\frac{1}{N_{\mathrm{tree}}}
\frac{1}{|P(T)|}
\frac{m_{p,j}}{\sum_{j'}m_{p,j'}}.
\]

- Add row-level tensor:

```text
objective_weights: float32 [batch]
```

- In `fresh_iid`, `m_{p,j}=1`, so siblings receive equal weight inside their parent.
- Validate:

```text
sum_j local_child_weight[p,j] == 1 for every parent
sum_p parent_weight[T,p] == 1 for every tree
sum_all_rows objective_weights == 1 for the full update batch
```

- `allocated_k` remains an integrity field; it must not be used as an extra optimization multiplier.

**Acceptance tests**

- Non-uniform branch factors do not change parent importance.
- Variable segment lengths do not change child importance after token averaging.
- Uniform trees match the explicit hierarchical reference objective.

---

## P0.4 — Make mini/microbatch gradients exactly match the full-batch objective

**Targets**

```text
verl/verl/workers/actor/dp_actor.py
verl/recipe/gear_tree/policy_loss.py
```

**Required changes**

- `vdra_node_balanced_ppo` must:

```text
1. compute PPO-clipped token losses;
2. take a token mean for each child row;
3. return sum(objective_weights * child_loss).
```

- Do not recompute parent/tree means inside each microbatch.
- Preserve `objective_weights` through row reordering, mini-batch splitting, dynamic batching, and microbatch splitting.
- Do not apply the legacy `1 / gradient_accumulation` scaling again to an already globally weighted VDRA loss.
- Account for data-parallel gradient averaging so the final distributed gradient equals the full-batch weighted sum.
- Do not apply `ratio_threshold` as a per-microbatch skip on the canonical VDRA path. Keep clipping and report the ratio metric; retain legacy skip behavior only for `treetune_ppo`.
- `treetune_ppo` must remain unchanged for the SPO baseline.

**Acceptance tests**

Compare loss and gradients for the same examples under:

```text
full batch
multiple mini-batches
multiple microbatches
permuted row order
dynamic batching
simulated two-rank partition + averaged gradients
```

All results must match the explicit full-batch reference within tolerance.

---

## P0.5 — Verify rollout/scorer weights using real runtime evidence

**Targets**

```text
verl/recipe/gear_tree/async_tree_rollout.py
verl/recipe/gear_tree/gear_ray_trainer.py
verl/recipe/gear_tree/gear_core/gear/vllm_scorer.py
verl/recipe/gear_tree/config/gear_tree_trainer.yaml
```

**Required changes**

- Do not silently use `scorer_api_base` as `rollout_api_base`.
- Support exactly two explicit modes:

```text
scorer_uses_rollout_server: true
```

or

```text
rollout_api_base: <rollout endpoint>
scorer_api_base: <independent scorer endpoint>
```

- Fetch the server-reported weight version from every configured endpoint.
- In strict VDRA, fail before allocation when a version is missing or mismatched.
- Propagate the verified versions and boolean result to the run manifest.

**Acceptance tests**

- Same-server explicit mode passes with one verified version.
- Two-server matching versions pass.
- Missing, stale, or mismatched versions fail strict mode.

---

## P0.6 — Make the manifest report observed facts only

**Targets**

```text
verl/recipe/gear_tree/manifest_lifecycle.py
verl/recipe/gear_tree/run_manifest.py
verl/recipe/gear_tree/gear_ray_trainer.py
```

**Required changes**

- Do not infer `complete_parent_microbatches`, scorer verification, or normalization success from config values.
- Set fields only from runtime checks:

```text
unique tree IDs verified
complete parent groups verified
fresh_iid row count == allocated_k
objective_weights globally sum to 1
full-batch vs split-gradient invariant enabled
rollout/scorer versions verified
no truncation
stored old log-probs used
```

- A canonical main manifest is invalid until at least one successful actor update passes all checks.
- Any later failure keeps the run invalid.

**Acceptance tests**

- A clean synthetic update produces a valid manifest.
- Each individual invariant failure makes it invalid.
- Save/load preserves all fields and counters.

---

## P0.7 — Put all correctness tests in CPU CI

**Targets**

```text
.github/workflows/cpu-ci.yml
verl/recipe/gear_tree/tests/
tests/
```

**Required changes**

CPU CI must run the relevant tests from both test roots, including:

```text
node-balanced loss and gradient parity
group metadata and zero-advantage handling
unique tree/edge IDs
complete-tree replay
objective-weight normalization
manifest lifecycle
bounded allocation slack semantics
Smoke A-D config composition
```

Also run:

```bash
python -m compileall vdra_core verl/recipe/gear_tree
```

- Do not exclude a new correctness test merely because it imports the tree recipe.
- Install the minimum CPU dependencies needed by the targeted tests.
- Python 3.10 and 3.12 jobs must both pass.

**Acceptance criteria**

```text
0 failed tests
0 collection errors
0 import errors
GitHub Actions status == success
```

---

## P0.8 — Add one pre-GPU gate command

**Target**

```text
scripts/pre_gpu_check.sh
```

**Required behavior**

The script must fail fast and return non-zero unless all of the following pass:

```text
compileall
all targeted CPU tests
main Hydra config composition
Smoke A-D config composition
main config uses fresh_iid + linear bound + node-balanced loss
only_adv_greater_than_zero == false
strict group integrity enabled
unique tree-ID test passes
full-vs-split gradient parity passes
manifest synthetic lifecycle passes
```

Print one final line only after success:

```text
PRE_GPU_CHECK=PASS
```

---

# Definition of done

The repository is ready for a GPU smoke run only when:

```text
[ ] P0.1–P0.8 are complete
[ ] scripts/pre_gpu_check.sh prints PRE_GPU_CHECK=PASS
[ ] CPU CI is green on Python 3.10 and 3.12
[ ] no known test is xfailed/skipped for a correctness blocker
[ ] Smoke D config composes without fallback or deprecated modes
```

Then run **Smoke D only** for at least five successful actor updates. Do not start long experiments until Smoke D confirms finite loss/gradients, valid manifest, no group-integrity failure, and correct scorer/rollout versions.

The synthetic RQ3/RQ4 scripts are scaffolding, not pre-GPU blockers and not paper evidence. Do not expand them until the canonical GPU smoke path is stable.
