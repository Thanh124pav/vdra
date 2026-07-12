#!/usr/bin/env bash
# Launch a GEAR/Tree-family algorithm on verl.
#
# Usage:
#   ALGO=gear_spo_tree MODEL=<hf_path> TRAIN=<parquet> VAL=<parquet> \
#     bash recipe/gear_tree/run_gear_tree.sh [extra hydra overrides...]
#
# ALGO in:
#   spo_chain | spo_tree | treerl | treepo
#   gear_spo_chain | gear_spo_tree | gear_treerl | gear_treepo | gear_vineppo
#
# Each ALGO maps to (tree_shape, tree_update_mode, gear.enabled). Everything is a
# thin CLI overlay on config/gear_tree_trainer.yaml — no per-variant YAML needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"

ALGO="${ALGO:-gear_spo_tree}"
MODEL="${MODEL:?set MODEL=<hf model path>}"
TRAIN="${TRAIN:?set TRAIN=<train parquet>}"
VAL="${VAL:?set VAL=<val parquet>}"
TREE="${TREE:-6,6,6}"            # tree shape for tree variants
CHAIN="${CHAIN:-1,1,1}"          # shape for chain variants

VINEPPO_K=0
case "$ALGO" in
  vineppo)         SHAPE="$CHAIN"; MODE=spo;             GEAR=false; VINEPPO_K="${VINEPPO_K_OVERRIDE:-9}" ;;
  spo_chain)       SHAPE="$CHAIN"; MODE=spo;             GEAR=false ;;
  spo_tree)        SHAPE="$TREE";  MODE=spo;             GEAR=false ;;
  treerl)          SHAPE="$TREE";  MODE=treerl_original; GEAR=false ;;
  treepo)          SHAPE="$TREE";  MODE=treepo_original; GEAR=false ;;
  gear_spo_chain)  SHAPE="$CHAIN"; MODE=spo;             GEAR=true  ;;
  gear_spo_tree)   SHAPE="$TREE";  MODE=spo;             GEAR=true  ;;
  gear_treerl)     SHAPE="$TREE";  MODE=treerl_original; GEAR=true  ;;
  gear_treepo)     SHAPE="$TREE";  MODE=treepo_original; GEAR=true  ;;
  gear_vineppo)    SHAPE="$TREE";  MODE=spo;             GEAR=true;  VINEPPO_K="${VINEPPO_K_OVERRIDE:-9}" ;;
  *) echo "unknown ALGO=$ALGO (use scripts/train_grpo.sh / train_rloo.sh for GRPO/RLOO)" >&2; exit 1 ;;
esac

EXP_NAME="${EXP_NAME:-gear_tree-${ALGO}}"

VDRA_OVERRIDES=()
if [[ "${GEAR}" == "true" ]]; then
  : "${EPS_TAIL_CALIBRATION_PATH:?set EPS_TAIL_CALIBRATION_PATH=<artifact.json> for VDRA}"
  SCORER_API_BASE="${SCORER_API_BASE:-http://127.0.0.1:8000/v1}"
  VDRA_OVERRIDES+=(
    "gear_tree.gear.eps_tail_calibration_path=${EPS_TAIL_CALIBRATION_PATH}"
    "gear_tree.gear.scorer_api_base=${SCORER_API_BASE}"
  )
fi

python -m recipe.gear_tree.main_gear_tree \
  --config-path "$(pwd)/recipe/gear_tree/config" \
  --config-name gear_tree_trainer \
  actor_rollout_ref.model.path="$MODEL" \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  "gear_tree.tree_shape=[${SHAPE}]" \
  "gear_tree.tree_update_mode=${MODE}" \
  "gear_tree.gear.enabled=${GEAR}" \
  "gear_tree.vineppo_K=${VINEPPO_K}" \
  "${VDRA_OVERRIDES[@]}" \
  trainer.experiment_name="$EXP_NAME" \
  "$@"
