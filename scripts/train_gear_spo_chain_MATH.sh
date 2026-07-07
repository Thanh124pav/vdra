#!/usr/bin/env bash
# Train GEAR-SPO-chain on MATH.
#
# Default TREE=6 keeps the run chain-compatible: one GEAR-controlled frontier
# expansion per prompt, with pruning/allocation at the root. Set TREE=66/666
# for deeper tree ablations.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-deepseekR1Qwen}"
TREE="${TREE:-${GEAR_TREE:-6}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-gear-spo-chain-${TREE}-${MODEL}-math}"

CFGS="$(resolve_math_config gear_spo_chain "${MODEL}")"
CFGS+=",$(ensure_tree_config "${TREE}")"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
