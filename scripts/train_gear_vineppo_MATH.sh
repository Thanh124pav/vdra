#!/usr/bin/env bash
# Train GEAR-VinePPO on MATH.
#
# Default TREE=6 preserves VinePPO's root->response trajectory assumption.
# Current VinePPO parsing supports one tree depth only, so TREE must be one
# digit 1-9.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-deepseekR1Qwen}"
TREE="${TREE:-${GEAR_TREE:-6}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-gear-vineppo-${TREE}-${MODEL}-math}"

if ! [[ "${TREE}" =~ ^[1-9]$ ]]; then
  echo "[gear-vineppo] TREE=${TREE} is not supported; VinePPO currently expects one tree depth (e.g. TREE=6)." >&2
  exit 1
fi

CFGS="$(resolve_math_config gear_vineppo "${MODEL}")"
CFGS+=",$(ensure_tree_config "${TREE}")"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
