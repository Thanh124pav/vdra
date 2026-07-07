#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEAR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/_common.sh"

usage() {
  cat <<'EOF'
Usage:
  DATA_UPLOAD_TARGET=oss://bucket/path/data.zip bash scripts/package_and_upload_data.sh
  DATA_UPLOAD_TARGET="https://.../upload-url" bash scripts/package_and_upload_data.sh

Optional env vars:
  DATA_ARCHIVE_NAME   Archive file name. Default: data_non_local_<timestamp>.zip
  DATA_LOCAL_PATTERN  Glob excluded from upload. Default: math-local-*
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${DATA_UPLOAD_TARGET:-${1:-}}"
if [[ -z "${target}" ]]; then
  echo "[upload] missing target. Set DATA_UPLOAD_TARGET or pass it as the first argument." >&2
  usage >&2
  exit 1
fi

archive_name="${DATA_ARCHIVE_NAME:-data_non_local_$(date +%Y%m%d_%H%M%S).zip}"
archive_path="${GEAR_ROOT}/${archive_name}"
local_pattern="${DATA_LOCAL_PATTERN:-math-local-*}"

if [[ ! -d "${GEAR_ROOT}/data" ]]; then
  echo "[upload] data/ does not exist under ${GEAR_ROOT}" >&2
  exit 1
fi

rm -f "${archive_path}"

cd "${GEAR_ROOT}"

mapfile -t data_entries < <(find data -mindepth 1 -maxdepth 1 ! -name "${local_pattern}" -print | sort)

if [[ ${#data_entries[@]} -eq 0 ]]; then
  echo "[upload] no non-local entries found under data/ (pattern: ${local_pattern})" >&2
  exit 1
fi

zip -qr "${archive_path}" "${data_entries[@]}"

echo "[upload] created ${archive_path}"

if [[ "${target}" == oss://* ]]; then
  if command -v ossutil >/dev/null 2>&1; then
    ossutil cp "${archive_path}" "${target}"
  else
    echo "[upload] ossutil is required for oss:// targets but was not found in PATH" >&2
    exit 1
  fi
elif [[ "${target}" == s3://* ]]; then
  if command -v aws >/dev/null 2>&1; then
    aws s3 cp "${archive_path}" "${target}"
  else
    echo "[upload] aws CLI is required for s3:// targets but was not found in PATH" >&2
    exit 1
  fi
elif [[ "${target}" == http://* || "${target}" == https://* ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl --fail --show-error --upload-file "${archive_path}" "${target}"
  else
    echo "[upload] curl is required for http(s) upload targets but was not found in PATH" >&2
    exit 1
  fi
else
  echo "[upload] unsupported target scheme: ${target}" >&2
  echo "[upload] use oss://, s3://, or a pre-signed http(s) upload URL" >&2
  exit 1
fi

echo "[upload] uploaded ${archive_path} to ${target}"