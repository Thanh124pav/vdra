#!/usr/bin/env bash
# Train spo_chain on MATH (verl GEAR/Tree recipe).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
recipe_run spo_chain "$@"
