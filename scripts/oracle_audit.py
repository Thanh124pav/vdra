"""Abl 7 oracle audit (PLAN.md §5).

Reads a saved trajectory pickle from an GEAR run (with
emit_pruned_edges=True) and a parallel SPO run on the same problems, then
estimates the false-share / false-prune rates by checking:

  * False Share: |V*(s) - V*(target)| > epsilon, where V* is approximated
    from the vanilla SPO tree's MC reward at the matching node.
  * False Prune: any descendant of `s` reached the gold answer in the
    vanilla SPO tree.

Outputs a small markdown report to stdout.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def index_by_seg(tree, out: Dict[str, dict] | None = None) -> Dict[str, dict]:
    out = {} if out is None else out
    sid = tree.get("gear_segment_id")
    if sid is not None:
        out[sid] = tree
    for ch in tree.get("children", []) or []:
        index_by_seg(ch, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gear_trajectories", required=True)
    ap.add_argument("--spo_trajectories", required=True)
    ap.add_argument("--epsilon", type=float, default=0.02)
    args = ap.parse_args()

    gear = load_pickle(Path(args.gear_trajectories))
    spo = load_pickle(Path(args.spo_trajectories))

    spo_index_by_problem = {}
    for tr in spo:
        pid = tr.get("data_instance", {}).get("_treetune__idx")
        if pid is None:
            continue
        spo_index_by_problem[pid] = index_by_seg(tr.get("tree", tr))

    false_share = 0
    total_share = 0
    false_prune = 0
    total_prune = 0
    for tr in gear:
        pid = tr.get("data_instance", {}).get("_treetune__idx")
        idx = spo_index_by_problem.get(pid, {})
        for sid, node in index_by_seg(tr.get("tree", tr)).items():
            action = node.get("gear_action")
            if action == "share":
                total_share += 1
                tgt = node.get("gear_share_target")
                spo_node = idx.get(sid)
                spo_tgt = idx.get(tgt) if tgt else None
                if spo_node is not None and spo_tgt is not None:
                    if abs((spo_node.get("reward") or 0) - (spo_tgt.get("reward") or 0)) > args.epsilon:
                        false_share += 1
            elif action == "prune":
                total_prune += 1
                spo_node = idx.get(sid)
                if spo_node is not None and (spo_node.get("reward") or 0) > 0:
                    false_prune += 1

    print(f"# Oracle Audit\n")
    print(f"- Total Share: {total_share}, False Share: {false_share} ({false_share / max(total_share, 1):.2%})")
    print(f"- Total Prune: {total_prune}, False Prune: {false_prune} ({false_prune / max(total_prune, 1):.2%})")


if __name__ == "__main__":
    main()
