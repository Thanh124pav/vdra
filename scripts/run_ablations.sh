#!/usr/bin/env bash
# Run active GEAR ablations on MATH/Qwen2.5-1.5B with 6-6-6 trees.
# Caller can restrict via ABLATIONS="abl7" etc.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TREE="${TREE:-${GEAR_TREE:-666}}"
ABLATIONS="${ABLATIONS:-abl7}"

BASE_CFG="${GEAR_ROOT}/configs/polIter_qwen1_5b_base_gear_tree_MATH.jsonnet"
TREE_CFG="$(ensure_tree_config "${TREE}")"

run() {
  local exp="$1"
  local addon="$2"
  local cfgs="${BASE_CFG},${TREE_CFG},${addon}"
  APP_EXPERIMENT_NAME="${exp}" gear_run "${exp}" "${cfgs}"
}

for abl in ${ABLATIONS}; do
  case "${abl}" in
    abl7)
      run "abl7-oracle-record" "${GEAR_ROOT}/configs/ablations/abl7_oracle_record.jsonnet"
      ;;
    noresidual)
      run "no-residual-budget" "${GEAR_ROOT}/configs/ablations/abl_no_residual_budget.jsonnet"
      ;;
    noallocation)
      run "no-allocation" "${GEAR_ROOT}/configs/ablations/abl_no_allocation.jsonnet"
      ;;
    *)
      echo "[ablations] Unknown: ${abl}" >&2
      exit 2
      ;;
  esac
done
