#!/usr/bin/env bash
# Tail the human-readable GEAR demos file in a running experiment.
# Usage:
#   bash scripts/tail_demos.sh <exp_name>
#   bash scripts/tail_demos.sh <exp_name> jsonl     # raw JSONL stream
#
# Resolves to ${APP_DIRECTORY:-experiments}/<exp>/gear_demos/demos.{md,jsonl}.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXP="${1:?Usage: tail_demos.sh <exp_name> [md|jsonl]}"
KIND="${2:-md}"

DEMOS_DIR="${APP_DIRECTORY}/${EXP}/gear_demos"
case "${KIND}" in
  md)    f="${DEMOS_DIR}/demos.md" ;;
  jsonl) f="${DEMOS_DIR}/demos.jsonl" ;;
  *) echo "unknown KIND: ${KIND}; use md or jsonl" >&2; exit 2 ;;
esac

if [[ ! -e "${f}" ]]; then
  echo "[tail_demos] not found yet: ${f}" >&2
  echo "(it will appear once the first tree finishes inference)" >&2
  exit 1
fi

exec tail -F "${f}"
