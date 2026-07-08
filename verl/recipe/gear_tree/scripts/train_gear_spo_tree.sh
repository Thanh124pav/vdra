#!/usr/bin/env bash
# Train gear_spo_tree on MATH (verl GEAR/Tree recipe).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
recipe_run gear_spo_tree "$@"
