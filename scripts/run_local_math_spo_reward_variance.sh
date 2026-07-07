#!/usr/bin/env bash
# Train the GEAR budget-allocation path on tiny MATH and dump per-node
# reward/variance approximation records. This is the SPO-tree-shaped run with
# TV used only for reward-variance estimation and budget allocation.
#
# Output:
#   ${APP_DIRECTORY}/${APP_EXPERIMENT_NAME}/gear_demos/reward_variance_nodes.jsonl
#   ${APP_DIRECTORY}/${APP_EXPERIMENT_NAME}/gear_demos/reward_variance_nodes.csv

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-qwen2_5_0_5b_instruct}"
MATH_LOCAL_SIZE="${MATH_LOCAL_SIZE:-10}"
EXP_NAME="${APP_EXPERIMENT_NAME:-local-math-spo-variance-3_9_27-${MATH_LOCAL_SIZE}-${MODEL}}"

SUBSET_CFG="${GEAR_ROOT}/configs/local/math_local_${MATH_LOCAL_SIZE}.jsonnet"
RUNTIME_CFG="${GEAR_ROOT}/configs/local/math_local_runtime.jsonnet"
VAR_CFG="${GEAR_ROOT}/configs/local/gear_log_reward_variance.jsonnet"
TREE_CFG="${GEAR_ROOT}/configs/local/tree_3_9_27.jsonnet"

[[ -f "${SUBSET_CFG}" ]] || {
  echo "Missing ${SUBSET_CFG}. Run: conda activate deeplearning && python scripts/prepare_math_local_subsets.py" >&2
  exit 2
}

CFGS="$(resolve_math_config gear_tree "${MODEL}"),${TREE_CFG},${SUBSET_CFG},${VAR_CFG},${RUNTIME_CFG}"

# Optionally disable FlashAttention at runtime (useful if CUDA/FlashAttention
# build is not available). Set DISABLE_FLASH_ATTENTION=1 when invoking the
# script to propagate APP_DISABLE_FLASH_ATTENTION=1 into the Jsonnet config.
if [[ "${DISABLE_FLASH_ATTENTION:-0}" == "1" ]]; then
  export APP_DISABLE_FLASH_ATTENTION=1
fi
gear_run "${EXP_NAME}" "${CFGS}" "$@"
