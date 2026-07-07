#!/usr/bin/env bash
# Wrapper around SPO's cached-inference download (saves a few hours of
# bootstrap inference for first-iteration episodes).
#
# Usage: bash scripts/download_cached.sh

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

cd "${GEAR_ROOT}"
python3 scripts/download_cached_inference_result.py "$@"
