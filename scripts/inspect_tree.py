"""Pretty-print one GEAR tree JSON with per-node action / TV / gap stats.

Usage:
    python scripts/inspect_tree.py <experiment_dir>/iteration_0001/episodes/0000/<idx>.json

Prints a depth-first tree where each node line shows:
    [depth] action  segment_id  reward  AvgLP_K  TV_m  gap_m

Useful for debugging share/prune decisions or screenshotting for the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict


def fmt(v: Any, width: int = 7) -> str:
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        return f"{v:+.3f}".rjust(width) if v == v else "  nan ".rjust(width)
    return str(v).rjust(width)


def walk(node: Dict[str, Any], depth: int = 0) -> None:
    indent = "  " * depth
    action = node.get("gear_action", "-")
    seg_id = node.get("gear_segment_id", "<root>")
    reward = node.get("reward")
    avg_lp_K = node.get("gear_avg_lp_K")
    tv = node.get("gear_tv_m")
    gap = node.get("gear_gap_m")
    target = node.get("gear_share_target")

    line = (
        f"{indent}[{depth}] {action:<6} {seg_id:<24} "
        f"R={fmt(reward)}  AvgLP_K={fmt(avg_lp_K)}  "
        f"TV={fmt(tv)}  gap={fmt(gap)}"
    )
    if target:
        line += f"  share->{target}"
    print(line)

    for ch in node.get("children", []) or []:
        walk(ch, depth + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree_path")
    args = ap.parse_args()

    with open(args.tree_path, "r") as f:
        tree = json.load(f)
    if isinstance(tree, str):  # SPO writes JSON-encoded JSON sometimes
        tree = json.loads(tree)

    stats = tree.get("gear_stats", {}) if isinstance(tree, dict) else {}
    if stats:
        print("GEAR stats:")
        for k, v in sorted(stats.items()):
            print(f"  {k:<32} {v}")
        print()

    walk(tree)
    return 0


if __name__ == "__main__":
    sys.exit(main())
