#!/usr/bin/env bash
# Train GEAR-tree on GSM8K with Qwen-0.5B (smallest model — fast iteration).

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${GEAR_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-gear-tree-${TREE}-qwen05b-gsm8k}"
CFGS="${GEAR_ROOT}/configs/polIter_qwen05b_gear_tree_GSM8K.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
gear_run "${EXP_NAME}" "${CFGS}" "$@"
