#!/usr/bin/env bash
# Train RLOO on MATH.  Default model: deepseekR1Qwen.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-deepseekR1Qwen}"
EXP_NAME="${APP_EXPERIMENT_NAME:-rloo-${MODEL}-math}"

CFGS="$(resolve_math_config rloo "${MODEL}")"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
