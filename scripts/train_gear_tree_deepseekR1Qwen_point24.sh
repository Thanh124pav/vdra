#!/usr/bin/env bash
# Train GEAR-tree on Point24 (24-game) with DeepSeek-R1-Distill-Qwen-1.5B.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${GEAR_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-gear-tree-${TREE}-deepseekR1Qwen-point24}"
CFGS="${GEAR_ROOT}/configs/polIter_deepseekR1Qwen_gear_tree_point24.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
gear_run "${EXP_NAME}" "${CFGS}" "$@"
