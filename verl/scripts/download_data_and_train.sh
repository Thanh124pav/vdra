#!/usr/bin/env bash
set -euo pipefail

# Wrapper for Docker images whose WORKDIR is the nested verl/ directory.
# The real script lives at repository_root/scripts/download_data_and_train.sh
# because it also prepares data before cd-ing into verl for training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${REPO_ROOT}/scripts/download_data_and_train.sh" "$@"
