#!/usr/bin/env bash
# Tiny smoke-test run: 2 iterations, depth-2 tree W=2.
# Verifies that the GEAR + SPO stack actually starts training end-to-end
# without burning GPU hours. Logs go to experiments/gear-debug-*.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXP_NAME="${APP_EXPERIMENT_NAME:-gear-debug-$(date +%H%M%S)}"
CFGS="${GEAR_ROOT}/configs/polIter_qwen05b_gear_tree_GSM8K.jsonnet"
CFGS+=",${GEAR_ROOT}/configs/debug.jsonnet"
gear_run "${EXP_NAME}" "${CFGS}" "$@"
