#!/usr/bin/env bash
# Train RLOO on GSM8K.  Currently only rho1bSft2 base is shipped.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-rho1bSft2}"
EXP_NAME="${APP_EXPERIMENT_NAME:-rloo-${MODEL}-gsm8k}"

CFGS="${GEAR_ROOT}/configs/polIter_${MODEL}_rloo_GSM8K.jsonnet"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
