"""Shared GEAR action labels.

The old reference-solution gating code was removed from the production GEAR path.
Budget allocation and sibling-local TV gates still annotate nodes with these
stable action strings so episode generation and logging remain compatible.
"""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    EXPAND = "expand"
    SHARE = "share"
    PRUNE = "prune"
