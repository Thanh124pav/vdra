#!/usr/bin/env bash
# Train DPO (positive variant) on MATH.  Default model: rho1bSft2.
# Override base with MODEL={rho1bSft2,deepseekSft2}.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-rho1bSft2}"
EXP_NAME="${APP_EXPERIMENT_NAME:-dpo-${MODEL}-math}"

CFGS="$(resolve_math_config dpo_positive "${MODEL}")"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
