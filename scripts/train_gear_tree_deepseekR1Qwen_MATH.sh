#!/usr/bin/env bash
# Train GEAR-tree on MATH with DeepSeek-R1-Distill-Qwen-1.5B (long-CoT).
# This is the "long-CoT" GEAR setup whose break-even depth was discussed in
# the analysis notes — set GEAR_TREE=4444 to actually realise it.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${GEAR_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-gear-tree-${TREE}-deepseekR1Qwen-math}"
CFGS="${GEAR_ROOT}/configs/polIter_deepseekR1Qwen_gear_tree_MATH.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
gear_run "${EXP_NAME}" "${CFGS}" "$@"
