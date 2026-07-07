#!/usr/bin/env bash
# Exp 2 (PLAN.md §4): Online prune/share rate per depth + advantage variance.
# Adds the abl7 oracle config so PRUNE/SHARE edges remain in the dataset and
# are emitted to wandb metrics for later post-hoc inspection.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${GEAR_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-exp2-prune-share-${TREE}-qwen1.5b-math}"

CFGS="${GEAR_ROOT}/configs/polIter_qwen1_5b_base_gear_tree_MATH.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
CFGS+=",${GEAR_ROOT}/configs/ablations/abl7_oracle_record.jsonnet"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
