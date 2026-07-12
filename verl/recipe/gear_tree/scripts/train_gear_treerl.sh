#!/usr/bin/env bash
# Train gear_treerl on MATH (verl GEAR/Tree recipe).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
recipe_run gear_treerl "$@"
