# Local MATH Experiments

These scripts are for local end-to-end checks on tiny MATH slices, not paper
scale runs.

## Prepare data

The full SPO dataset archive is downloaded with:

```bash
bash scripts/download_and_prepare_dataset.sh
```

Tiny deterministic MATH subsets are stored as HuggingFace `DatasetDict`s:

```bash
conda activate deeplearning
python scripts/prepare_math_local_subsets.py --source data/math --sizes 10 30 100
```

This creates:

- `data/math-local-10`
- `data/math-local-30`
- `data/math-local-100`

## Compare GEAR, SPO, GRPO

Run GEAR tree 3-9-27, SPO tree 3-9-27, and GRPO with 27 rollouts:

```bash
conda activate deeplearning
WANDB_PROJECT=gear-local \
MODEL=qwen3_0_6b \
MATH_LOCAL_SIZE=10 \
bash scripts/run_local_math_compare_3_9_27.sh
```

Useful overrides:

- `MODEL=qwen2_5_0_5b_instruct` for the existing 0.5B model override.
- `MATH_LOCAL_SIZE=30` or `100`.
- `GEAR_GPU=1` or `GEAR_GPUS=0,1`.
- `GEAR_TOTAL_NUM_ITERATIONS=1` for the shortest run.
- `WANDB_MODE=offline` when internet upload is unavailable.

Performance and time are logged to wandb. Timing is also written locally to:

```text
experiments/<exp_name>/training_timing.jsonl
```

## Reward Variance Logging

Run the SPO-tree-shaped GEAR budget-allocation path, where TV estimates are
used only to estimate per-node reward variance and allocate rollout budget:

```bash
conda activate deeplearning
WANDB_PROJECT=gear-local \
MODEL=qwen3_0_6b \
MATH_LOCAL_SIZE=10 \
bash scripts/run_local_math_spo_reward_variance.sh
```

Outputs:

```text
experiments/<exp_name>/gear_demos/reward_variance_nodes.jsonl
experiments/<exp_name>/gear_demos/reward_variance_nodes.csv
```

Each row contains `reward`, `reward_std`,
`empirical_child_reward_variance`, `gear_reward_variance`,
`gear_sigma2`, `gear_sigma4`, TV support counts, and allocated branch budget
for one node. The wandb run also gets summary metrics under
`gear/reward_variance_nodes/*`.
