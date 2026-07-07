"""Pretty-print SHARE / PRUNE demos written by the GEAR episode generator.

Usage:
    python scripts/inspect_demos.py <exp_dir>/gear_demos/demos.jsonl       \\
        [--action share|prune|all] [--limit 20] [--depth 2]

Reads the JSONL produced during training (each line = one tree) and prints
the demos to stdout, optionally filtered by action / depth. Designed for
offline servers: needs only the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def render_demo(d: dict) -> str:
    parts: List[str] = [
        f"  depth={d.get('depth')}  seg={d.get('seg_id')}  action={d.get('action')}",
    ]
    if d.get("avg_lp_K") is not None:
        parts.append(f"  AvgLP_K={d['avg_lp_K']:+.3f}")
    if d.get("tv_m") is not None:
        parts.append(f"  TV_m={d['tv_m']:.3f}")
    if d.get("gap_m") is not None:
        parts.append(f"  gap_m={d['gap_m']:.3f}")
    parts.append(f"  (eta={d.get('eta', 0):.3f}, tau={d.get('tau', 0):.3f})")
    out = ["".join(parts)]
    out.append(f"    parent : {d.get('parent_text', '')}")
    out.append(f"    child  : {d.get('child_text', '')}")
    if d.get("target_text"):
        out.append(
            f"    shared->{d.get('target_seg_id', '')} : {d['target_text']}"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="path to demos.jsonl")
    ap.add_argument(
        "--action", default="all", choices=("all", "share", "prune"),
        help="show only this action class (default: all)",
    )
    ap.add_argument("--depth", type=int, default=None, help="filter by depth")
    ap.add_argument(
        "--limit", type=int, default=20,
        help="stop after this many demos (default: 20; 0 = no limit)",
    )
    ap.add_argument(
        "--summary", action="store_true",
        help="print share/prune totals across the file and exit",
    )
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    n_share = n_prune = n_trees = 0
    n_printed = 0
    for rec in iter_records(path):
        n_trees += 1
        share = rec.get("demos", {}).get("share", []) or []
        prune = rec.get("demos", {}).get("prune", []) or []
        n_share += len(share)
        n_prune += len(prune)

        if args.summary:
            continue

        groups = []
        if args.action in ("all", "share"):
            groups.append(("SHARE", share))
        if args.action in ("all", "prune"):
            groups.append(("PRUNE", prune))

        for label, items in groups:
            for d in items:
                if args.depth is not None and d.get("depth") != args.depth:
                    continue
                if args.limit and n_printed >= args.limit:
                    return 0
                qid = rec.get("question_id")
                tidx = rec.get("tree_idx")
                print(f"[tree #{tidx} qid={qid}] {label}")
                print(render_demo(d))
                print()
                n_printed += 1

    if args.summary:
        print(f"trees:       {n_trees}")
        print(f"share demos: {n_share}")
        print(f"prune demos: {n_prune}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
