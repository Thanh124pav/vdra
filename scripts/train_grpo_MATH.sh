#!/usr/bin/env bash
# Train GRPO on MATH.  Default model: deepseekR1Qwen.
# Override base with MODEL={deepseekR1Qwen,rho1bSft2,deepseekSft2,qwen1_5b_base}.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-deepseekR1Qwen}"
EXP_NAME="${APP_EXPERIMENT_NAME:-grpo-${MODEL}-math}"

CFGS="$(resolve_math_config grpo "${MODEL}")"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
