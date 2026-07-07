#!/usr/bin/env bash
# Exp 1 (PLAN.md §4): Sample efficiency.
#   For each (Model x Dataset x Tree size), train SPO baseline + GEAR and
#   compare Pass@1 vs #problems-seen.
#
# Models: Qwen2.5-1.5B (MATH), Rho-1.1B-SFT (GSM8K).
# Trees:  4-4-4, 6-6-6, 8-8-8.
#
# Usage: bash scripts/run_exp1_sample_efficiency.sh [tree=666] [models=qwen,rho]

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREES="${TREES:-444 666 888}"
MODELS="${MODELS:-qwen rho}"
TAG="${EXP1_TAG:-exp1}"

run_one() {
  local model="$1" tree="$2"
  if [[ "${model}" == "qwen" ]]; then
    GEAR_TREE="${tree}" APP_EXPERIMENT_NAME="${TAG}-spo-tree-${tree}-qwen1.5b-math" \
      bash "${GEAR_ROOT}/scripts/run_baseline.sh" spo_tree_MATH
    GEAR_TREE="${tree}" APP_EXPERIMENT_NAME="${TAG}-gear-tree-${tree}-qwen1.5b-math" \
      bash "${GEAR_ROOT}/scripts/train_gear_tree_MATH.sh"
  else
    GEAR_TREE="${tree}" APP_EXPERIMENT_NAME="${TAG}-spo-tree-${tree}-rho1.1b-gsm8k" \
      bash "${GEAR_ROOT}/scripts/run_baseline.sh" spo_tree_GSM8K
    GEAR_TREE="${tree}" APP_EXPERIMENT_NAME="${TAG}-gear-tree-${tree}-rho1.1b-gsm8k" \
      bash "${GEAR_ROOT}/scripts/train_gear_tree_GSM8K.sh"
  fi
}

for model in ${MODELS}; do
  for tree in ${TREES}; do
    echo "[exp1] model=${model} tree=${tree}"
    run_one "${model}" "${tree}"
  done
done
