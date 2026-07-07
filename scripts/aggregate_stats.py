"""Aggregate GEAR trees in an iteration directory into a CSV report.

Walks `<experiment_dir>/iteration_*/episodes/*/*.json`, parses the
`gear_stats` block and the per-node action breakdown, and writes a CSV with
one row per tree:

    iteration, tree_idx, n_nodes, n_expand, n_share, n_prune,
    share_rate, prune_rate, avg_tv_share, avg_gap_prune

Useful for plotting Figure 4 (prune/share rate vs depth) without re-running
inference.

Usage:
    python scripts/aggregate_stats.py <experiment_dir> -o stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Counter, Dict, Iterable


def iter_trees(root: Path) -> Iterable[tuple[int, int, dict]]:
    for it_dir in sorted(root.glob("iteration_*")):
        try:
            it = int(it_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for tree_path in sorted(it_dir.rglob("*.json")):
            try:
                with tree_path.open() as f:
                    tree = json.load(f)
                if isinstance(tree, str):
                    tree = json.loads(tree)
            except Exception:
                continue
            try:
                tidx = int(tree_path.stem)
            except ValueError:
                tidx = -1
            yield it, tidx, tree


def count_actions(node: dict) -> Counter:
    c: Counter = Counter()
    stack = [node]
    while stack:
        n = stack.pop()
        a = n.get("gear_action")
        if a:
            c[a] += 1
        stack.extend(n.get("children") or [])
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("-o", "--output", default="stats.csv")
    args = ap.parse_args()

    rows = []
    for it, idx, tree in iter_trees(Path(args.experiment_dir)):
        stats = tree.get("gear_stats", {}) if isinstance(tree, dict) else {}
        actions = count_actions(tree if isinstance(tree, dict) else {})
        total = sum(actions.values())
        rows.append({
            "iteration": it,
            "tree_idx": idx,
            "n_nodes": total,
            "n_expand": actions.get("expand", 0),
            "n_share": actions.get("share", 0),
            "n_prune": actions.get("prune", 0),
            "share_rate": stats.get("gear/share_rate", 0.0),
            "prune_rate": stats.get("gear/prune_rate", 0.0),
            "avg_tv_share": stats.get("gear/avg_tv_when_share", 0.0),
            "avg_gap_prune": stats.get("gear/avg_gap_when_prune", 0.0),
        })

    if not rows:
        print(f"[aggregate] No trees found under {args.experiment_dir}")
        return 1

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[aggregate] Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
