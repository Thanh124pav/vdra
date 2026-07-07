#!/usr/bin/env bash
# Train GEAR-tree on MATH with Rho-1.1B-SFT.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${GEAR_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-gear-tree-${TREE}-rho1.1b-math}"
CFGS="${GEAR_ROOT}/configs/polIter_rho1bSft2_gear_tree_MATH.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
gear_run "${EXP_NAME}" "${CFGS}" "$@"
