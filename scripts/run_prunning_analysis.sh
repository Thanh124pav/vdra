#!/usr/bin/env bash
# Run analysis-only pruning diagnostics. This never trains or writes checkpoints.

set -euo pipefail

GEAR_ROOT="${GEAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${GEAR_ROOT}:${PYTHONPATH:-}"

BACKEND="${BACKEND:-replay}"
BUILDER="${BUILDER:-spo_step}"
TREE="${TREE:-${GEAR_TREE:-666}}"
TREE_M="${TREE_M:-600}"
MODEL="${MODEL:-sshleifer/tiny-gpt2}"

python "${GEAR_ROOT}/prunning_analysis/run_prunning_analysis.py" \
  --backend "${BACKEND}" \
  --builder "${BUILDER}" \
  --tree-shape "${TREE}" \
  --tree-m "${TREE_M}" \
  --model "${MODEL}" \
  "$@"
