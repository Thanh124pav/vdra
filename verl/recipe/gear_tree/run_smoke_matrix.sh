#!/usr/bin/env bash
# PLAN.md §8: run the four-cell smoke matrix in sequence.
#
# Usage:
#   MODEL_PATH=<hf-model> TRAIN=<train.parquet> VAL=<val.parquet> \
#     bash verl/recipe/gear_tree/run_smoke_matrix.sh [smoke_a|smoke_b|smoke_c|smoke_d|all]
#
# Each smoke run reuses gear_tree_trainer.yaml and layers one of the
# smoke_{a,b,c,d}_*.yaml overlays on top. This script does NOT launch
# long main-result training runs; it is the pre-flight smoke matrix.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH=<hf-model>}"
TRAIN="${TRAIN:?set TRAIN=<train parquet>}"
VAL="${VAL:?set VAL=<val parquet>}"
STEPS="${STEPS:-5}"
OUT_ROOT="${OUT_ROOT:-runs/smoke_matrix}"
WHICH="${1:-all}"

CONFIG_ROOT="recipe/gear_tree/config"

run_one() {
  local name="$1"
  local overlay="$2"
  local target_steps="$3"
  local exp_dir="${OUT_ROOT}/${name}"
  mkdir -p "${exp_dir}"
  echo "=== smoke ${name} -> ${exp_dir} (steps=${target_steps}) ==="
  python -m recipe.gear_tree.main_gear_tree \
    --config-path "${CONFIG_ROOT}" \
    --config-name gear_tree_trainer \
    --config-name "${overlay}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="${TRAIN}" \
    data.val_files="${VAL}" \
    trainer.default_local_dir="${exp_dir}" \
    trainer.total_training_steps="${target_steps}" \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "${@:4}"
}

case "${WHICH}" in
  smoke_a)  run_one smoke_a smoke_a_spo_baseline 2 ;;
  smoke_b)  run_one smoke_b smoke_b_vdra_alloc_legacy_loss 2 ;;
  smoke_c)  run_one smoke_c smoke_c_uniform_alloc_node_balanced 2 ;;
  smoke_d)  run_one smoke_d smoke_d_full_vdra "${STEPS}" ;;
  all)
    run_one smoke_a smoke_a_spo_baseline 2
    run_one smoke_b smoke_b_vdra_alloc_legacy_loss 2
    run_one smoke_c smoke_c_uniform_alloc_node_balanced 2
    run_one smoke_d smoke_d_full_vdra "${STEPS}"
    ;;
  *) echo "unknown target ${WHICH} (use smoke_a|smoke_b|smoke_c|smoke_d|all)" >&2; exit 1 ;;
esac

echo "=== smoke matrix (${WHICH}) done ==="
