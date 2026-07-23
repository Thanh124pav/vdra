#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the old monorepo root. The real script lives under
# verl/scripts so the nested verl/ directory can be split into its own repo.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec bash "${REPO_ROOT}/verl/scripts/download_data_and_train.sh" "$@"
