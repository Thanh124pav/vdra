#!/usr/bin/env bash
# Train treerl on MATH (verl GEAR/Tree recipe).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
recipe_run treerl "$@"
