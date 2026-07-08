#!/usr/bin/env bash
# Train RLOO on MATH (verl-native rloo estimator + treetune_ppo loss).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
flat_run rloo "$@"
