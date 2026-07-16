#!/usr/bin/env bash
# Train GEAR-tree on MATH.
#
# Tree shape via TREE=<digits> (or GEAR_TREE=<digits> for back-compat).
# Any shape works — if the matching branch_factor_<shape>.jsonnet does
# not exist, _common.sh:ensure_tree_config auto-generates one under
# configs/episode_generators/_generated/.
#   TREE=666     -> depth 3, M=600 (default)
#   TREE=6666    -> depth 4, M=500
#   TREE=8888    -> depth 4, M=500 (auto-generated)
#   TREE=3456    -> depth 4, M=500 (auto-generated; mixed widths)
# Override M for any tree config with TREE_M=<positive-int>.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TAIL_MODE="${TAIL_MODE:-${GEAR_TAIL_MODE:-none}}"
if [[ "${TAIL_MODE}" == "calibrated" ]]; then
  : "${EPS_TAIL_CALIBRATION_PATH:?set EPS_TAIL_CALIBRATION_PATH=<artifact.json> when TAIL_MODE=calibrated}"
fi

MODEL="${MODEL:-deepseekR1Qwen}"
TREE="${TREE:-${GEAR_TREE:-666}}"
EXP_NAME="${APP_EXPERIMENT_NAME:-gear-tree-${TREE}-${MODEL}-math}"

CFGS="$(resolve_math_config gear_tree "${MODEL}")"
CFGS+=",$(ensure_tree_config "${TREE}")"

gear_run "${EXP_NAME}" "${CFGS}" "$@"
