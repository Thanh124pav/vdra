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
    # --- VDRA ablations (Summary.md "Required Ablation Studies") -------------
    notail)        # #2: no tail correction (eps_tail = 0)
      run "vdra-no-tail" "${GEAR_ROOT}/configs/ablations/abl_no_tail.jsonnet"
      ;;
    tail)          # #3: calibrated global eps_tail
      run "vdra-tail-calibrated" "${GEAR_ROOT}/configs/ablations/abl_tail_calibrated.jsonnet"
      ;;
    taildepth)     # #12: depth-dependent eps_tail(d)
      run "vdra-tail-depth" "${GEAR_ROOT}/configs/ablations/abl_tail_depth.jsonnet"
      ;;
    simlemma)      # #8: gamma simulation-lemma bound vs direct linear TV bound
      run "vdra-bound-simlemma" "${GEAR_ROOT}/configs/ablations/abl_bound_simulation_lemma.jsonnet"
      ;;
    legacytv)      # estimator ablation: degenerate legacy |exp(LP)| TV
      run "vdra-tv-legacy-abs" "${GEAR_ROOT}/configs/ablations/abl_tv_legacy_abs.jsonnet"
      ;;
    nofloor)       # #4: no allocation floor
      run "vdra-no-floor" "${GEAR_ROOT}/configs/ablations/abl_no_floor.jsonnet"
      ;;
    floor1)        # #4: n_min = 1 floor
      run "vdra-floor-1" "${GEAR_ROOT}/configs/ablations/abl_floor_1.jsonnet"
      ;;
    noqueue)       # #5: no queue batching
      run "vdra-no-queue" "${GEAR_ROOT}/configs/ablations/abl_no_queue.jsonnet"
      ;;
    queue8)        # #11: queue size/timeout sweep
      run "vdra-queue-8" "${GEAR_ROOT}/configs/ablations/abl_queue_8.jsonnet"
      ;;
    m30)           # #14: short-continuation length sweep
      run "vdra-m-30" "${GEAR_ROOT}/configs/ablations/abl_m_30.jsonnet"
      ;;
    m120)          # #14: short-continuation length sweep
      run "vdra-m-120" "${GEAR_ROOT}/configs/ablations/abl_m_120.jsonnet"
      ;;
    k0sweep)       # #13: pilot branch factor k0 sweep (n_tv_estimates 4/16)
      run "vdra-k0-4" "${GEAR_ROOT}/configs/ablations/abl_budget_allocation_n4.jsonnet"
      run "vdra-k0-16" "${GEAR_ROOT}/configs/ablations/abl_budget_allocation_n16.jsonnet"
      ;;
    *)
      echo "[ablations] Unknown: ${abl}" >&2
      exit 2
      ;;
  esac
done

# Offline validation (RQ2/RQ3/RQ4 + Direction D — oracle sigma^2, tail
# quantiles, allocation regret incl. the empirical/oracle-variance baselines):
#   python scripts/calibrate_tail_divergence.py --api-base ... --model ... \
#     --prompts-file ... --grade --out results/tail_calibration.json
