#!/usr/bin/env bash
# Deep-tree compute Pareto: matches SPO and GEAR at D in {2,3,4,5}.
# Demonstrates the break-even formula:
#     T_GEAR / T_SPO  ~  rho^(D-1) * kappa
# i.e. GEAR wins more as D grows and the savings rho^(D-1) shrink kappa
# down towards (or below) 1.
#
# Default trees:
#   D=2 -> depth_2_W6
#   D=3 -> branch_factor_666
#   D=4 -> branch_factor_4444
#   D=5 -> branch_factor_33333
# Override via DEPTHS="2 3 4" or TREES_<D> env vars.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DEPTHS="${DEPTHS:-2 3 4 5}"
TAG="${EXP_TAG:-exp-deep-tree}"

cfg_for_depth() {
  case "$1" in
    2) echo "depth_2_W6" ;;
    3) echo "branch_factor_666" ;;
    4) echo "branch_factor_4444" ;;
    5) echo "branch_factor_33333" ;;
    *) echo "branch_factor_666" ;;
  esac
}

for d in ${DEPTHS}; do
  tree_cfg="$(cfg_for_depth "${d}")"
  echo "[deep-tree] depth=${d} tree=${tree_cfg}"

  # Matched SPO baseline.
  APP_EXPERIMENT_NAME="${TAG}-spo-d${d}" \
    bash "${GEAR_ROOT}/scripts/run_baseline.sh" spo_tree_MATH "$@" \
    --override "episode_generator.inference_strategy.max_depth=${d}" \
    --configs "${GEAR_ROOT}/configs/episode_generators/${tree_cfg}.jsonnet"

  # GEAR at the same depth.
  CFGS="${GEAR_ROOT}/configs/polIter_qwen1_5b_base_gear_tree_MATH.jsonnet"
  CFGS+=",${GEAR_ROOT}/configs/episode_generators/${tree_cfg}.jsonnet"
  APP_EXPERIMENT_NAME="${TAG}-gear-d${d}" \
    gear_run "${TAG}-gear-d${d}" "${CFGS}" "$@"
done
