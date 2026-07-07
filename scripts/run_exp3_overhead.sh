#!/usr/bin/env bash
# Exp 3 (PLAN.md §4): Overhead measurement.
#   Trains GEAR on a small fixed slice and prints LP-scoring time, BST time,
#   total wallclock vs SPO on identical data. We piggy-back on wandb's
#   `timing/episode_generation/*` metrics already emitted by SPO and the
#   `gear/*` metrics our episode generator adds.
#
# Usage: bash scripts/run_exp3_overhead.sh [GEAR_TREE=666]

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GEAR_TREE="${GEAR_TREE:-666}"
NUM_ITER="${NUM_ITER:-10}"

# Cap iterations for the cost run.
EXTRA_OVERRIDES="--debug False --override 'num_iterations=${NUM_ITER}'"

EXP_NAME="${APP_EXPERIMENT_NAME:-exp3-overhead-${GEAR_TREE}}"
APP_EXPERIMENT_NAME="${EXP_NAME}-spo" \
  GEAR_TREE="${GEAR_TREE}" bash "${GEAR_ROOT}/scripts/run_baseline.sh" spo_tree_MATH ${EXTRA_OVERRIDES}
APP_EXPERIMENT_NAME="${EXP_NAME}-gear" \
  GEAR_TREE="${GEAR_TREE}" bash "${GEAR_ROOT}/scripts/train_gear_tree_MATH.sh" ${EXTRA_OVERRIDES}
