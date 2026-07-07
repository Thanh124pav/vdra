#!/usr/bin/env bash
# Train PPO on MATH.  Default model: Qwen2.5-1.5B (rho1bSft2 base).
# Override base with MODEL={deepseekR1Qwen,rho1bSft2,deepseekSft2,qwen1_5b_base}.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-rho1bSft2}"
EXP_NAME="${APP_EXPERIMENT_NAME:-ppo-${MODEL}-math}"

CFGS="$(resolve_math_config ppo "${MODEL}")"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
