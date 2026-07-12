#!/usr/bin/env bash
# Shared setup + launch helpers for the GEAR/Tree recipe scripts.
#
# Env knobs (all overridable):
#   MODEL   HF model path             (default SmolLM2-135M)
#   TRAIN   train parquet             (default data/math/train.parquet)
#   VAL     val parquet               (default data/math/test.parquet)
#   TREE    tree shape (tree algos)   (default 6,6,6)
#   CHAIN   chain shape (chain algos) (default 1,1,1)
#   N       group size (GRPO/RLOO)    (default 8)
#   GPUS    n_gpus_per_node           (default 1)
#   EXP_NAME experiment name
set -euo pipefail

export MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M}"
export TRAIN="${TRAIN:-data/math/train.parquet}"
export VAL="${VAL:-data/math/test.parquet}"
export TREE="${TREE:-6,6,6}"
export CHAIN="${CHAIN:-1,1,1}"
export N="${N:-8}"
export GPUS="${GPUS:-1}"

# Run one of the tree-recipe algorithms (delegates to run_gear_tree.sh).
recipe_run() {
  local algo="$1"; shift
  ALGO="$algo" EXP_NAME="${EXP_NAME:-gear_tree-$algo}" \
    bash "$(dirname "${BASH_SOURCE[0]}")/../run_gear_tree.sh" \
      trainer.n_gpus_per_node="$GPUS" "$@"
}

# Run a flat GRPO/RLOO algorithm through verl-native main (main_flat.py).
flat_run() {
  local est="$1"; shift
  python -m recipe.gear_tree.main_flat \
    --config-path "$(pwd)/recipe/gear_tree/config" \
    --config-name flat_trainer \
    algorithm.adv_estimator="$est" \
    actor_rollout_ref.model.path="$MODEL" \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    actor_rollout_ref.rollout.n="$N" \
    trainer.n_gpus_per_node="$GPUS" \
    trainer.experiment_name="${EXP_NAME:-flat-$est}" \
    "$@"
}
