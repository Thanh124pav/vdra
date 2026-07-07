#!/usr/bin/env bash
# Run GEAR on every supported (model, dataset) combo. One script per pair so
# we can sweep them all in a CI-style run.
#
# Override which configs to run with MODELS="qwen15b_math rho_gsm8k".

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GEAR_TREE="${GEAR_TREE:-666}"
MODELS="${MODELS:-qwen15b_math deepseekR1Qwen_math deepseekR1Qwen_point24 rho_gsm8k rho_math qwen05b_gsm8k}"
TAG="${EXP_TAG:-allmodels}"

run() {
  local key="$1" inner="$2"
  APP_EXPERIMENT_NAME="${TAG}-${key}-${GEAR_TREE}" \
  GEAR_TREE="${GEAR_TREE}" \
    bash "${GEAR_ROOT}/scripts/${inner}"
}

for m in ${MODELS}; do
  case "$m" in
    qwen15b_math)   run "$m" "train_gear_tree_MATH.sh" ;;
    deepseekR1Qwen_math)    run "$m" "train_gear_tree_deepseekR1Qwen_MATH.sh" ;;
    deepseekR1Qwen_point24) run "$m" "train_gear_tree_deepseekR1Qwen_point24.sh" ;;
    rho_gsm8k)      run "$m" "train_gear_tree_GSM8K.sh" ;;
    rho_math)       run "$m" "train_gear_tree_rho_MATH.sh" ;;
    qwen05b_gsm8k)  run "$m" "train_gear_tree_qwen05b_GSM8K.sh" ;;
    *) echo "[allmodels] unknown model: $m" >&2; exit 2 ;;
  esac
done
