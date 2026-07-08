#!/usr/bin/env bash
# Train GRPO on MATH (verl-native grpo estimator + treetune_ppo loss).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
flat_run grpo "$@"
