#!/usr/bin/env bash
# Train VinePPO on MATH.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-deepseekR1Qwen}"
EXP_NAME="${APP_EXPERIMENT_NAME:-vineppo-${MODEL}-math}"

CFGS="$(resolve_math_config vineppo "${MODEL}")"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
