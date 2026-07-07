#!/usr/bin/env bash
# Train GEAR-tree on GSM8K with Rho-1.1B-SFT.
# Tree shape via TREE=<digits> (GEAR_TREE also accepted).  See
# train_gear_tree_MATH.sh for the auto-generation rules.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${GEAR_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-gear-tree-${TREE}-rho1.1b-gsm8k}"
CFGS="${GEAR_ROOT}/configs/polIter_rho1bSft2_gear_tree_GSM8K.jsonnet"
CFGS+=",$(ensure_tree_config "${TREE}")"
gear_run "${EXP_NAME}" "${CFGS}" "$@"
