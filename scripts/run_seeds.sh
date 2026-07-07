#!/usr/bin/env bash
# Multi-seed driver: replays a given training script with APP_SEED=42,123,7
# (override via SEEDS=...). Useful for the variance-bar runs in the paper.
#
# Usage:
#   bash scripts/run_seeds.sh scripts/train_gear_tree_MATH.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_seeds.sh scripts/train_gear_tree_GSM8K.sh

set -euo pipefail
GEAR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INNER="${1:?Usage: run_seeds.sh <inner_script> [...args]}"
shift
SEEDS="${SEEDS:-42 123 7}"

[[ -x "${INNER}" ]] || INNER="${GEAR_ROOT}/${INNER}"

for seed in ${SEEDS}; do
  echo "[run_seeds] seed=${seed} inner=${INNER}"
  APP_SEED="${seed}" \
  APP_EXPERIMENT_NAME="${APP_EXPERIMENT_NAME:-seedrun}-seed${seed}" \
    bash "${INNER}" "$@"
done
