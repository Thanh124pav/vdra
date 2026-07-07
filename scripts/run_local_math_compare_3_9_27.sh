#!/usr/bin/env bash
# Local MATH smoke comparison:
#   GEAR tree 3-9-27, SPO tree 3-9-27, GRPO with 27 rollouts.
#
# Defaults favor a single local GPU and tiny datasets. Override:
#   MODEL=qwen3_0_6b|qwen2_5_0_5b_instruct
#   MATH_LOCAL_SIZE=10|30|100
#   WANDB_PROJECT=...

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-qwen3_0_6b}"
MATH_LOCAL_SIZE="${MATH_LOCAL_SIZE:-10}"
TAG="${EXP_TAG:-local-math-${MATH_LOCAL_SIZE}-${MODEL}-3_9_27}"

SUBSET_CFG="${GEAR_ROOT}/configs/local/math_local_${MATH_LOCAL_SIZE}.jsonnet"
RUNTIME_CFG="${GEAR_ROOT}/configs/local/math_local_runtime.jsonnet"
TREE_CFG="${GEAR_ROOT}/configs/local/tree_3_9_27.jsonnet"
GRPO_CFG="${GEAR_ROOT}/configs/local/grpo_27rolls.jsonnet"

[[ -f "${SUBSET_CFG}" ]] || {
  echo "Missing ${SUBSET_CFG}. Run: conda activate deeplearning && python scripts/prepare_math_local_subsets.py" >&2
  exit 2
}

run_one() {
  local name="$1"
  local cfgs="$2"
  echo "[local-compare] ${name}"
  APP_EXPERIMENT_NAME="${TAG}-${name}" gear_run "${TAG}-${name}" "${cfgs}" "$@"
}

run_one "gear-tree-3_9_27" "$(resolve_math_config gear_tree "${MODEL}"),${TREE_CFG},${SUBSET_CFG},${RUNTIME_CFG}" "$@"
run_one "spo-tree-3_9_27" "$(resolve_math_config spo_tree "${MODEL}"),${TREE_CFG},${SUBSET_CFG},${RUNTIME_CFG}" "$@"
run_one "grpo-27rolls" "$(resolve_math_config grpo "${MODEL}"),${GRPO_CFG},${SUBSET_CFG},${RUNTIME_CFG}" "$@"
