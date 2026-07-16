"""Tree logging for the GEAR/Tree recipe — matches treetune's offline demos.

Reuses the vendored ``gear_core.gear.logging_helpers`` /
``gear_core.gear.tree_policy_logging`` (byte-identical to treetune) to emit, per
tree:
  * ``<demos_dir>/demos.jsonl`` — one record/tree: stats, per-depth counts,
    SHARE/PRUNE/budget demo rows, ``tree_construction_seconds``.
  * ``<demos_dir>/demos.md``   — human-readable Markdown (same as treetune).
  * ``<demos_dir>/full_trees/tree_<idx>.json`` — one **complete** example tree
    (rate-limited), plus its Markdown appended to ``demos.md``.
  * console line with the tree's stats.

Training-time logging (``training_timing.jsonl``) is written by
``RayGearTreeTrainer.fit`` (per step: generation / update / wall seconds), mirroring
treetune's ``policy_iteration_runtime`` timing log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from recipe.gear_tree.gear_core.gear import logging_helpers as lh
from recipe.gear_tree.gear_core.gear import tree_policy_logging as tpl
from vdra_core.logging_schema import (
    COMPUTE_PROXY_DEFINITION,
    budget_claim_for_mode,
    persist_vdra_artifacts,
)


def basic_tree_stats(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Framework-agnostic tree shape/reward stats (works for SPO/TreeRL/TreePO)."""
    nodes = list(tpl.iter_tree_nodes(tree))
    by_depth: Dict[int, list] = {}
    n_leaf = 0
    for n in nodes:
        d = int(n.get("depth", 0))
        by_depth.setdefault(d, []).append(float(n.get("reward", 0.0) or 0.0))
        if n.get("leaf"):
            n_leaf += 1
    per_depth = {
        f"depth_{d}": {
            "n": len(rs),
            "reward_mean": float(np.mean(rs)) if rs else 0.0,
            "reward_std": float(np.std(rs)) if rs else 0.0,
        }
        for d, rs in sorted(by_depth.items())
    }
    return {
        "num_nodes": len(nodes),
        "num_leaves": n_leaf,
        "max_depth": max((int(n.get("depth", 0)) for n in nodes), default=0),
        "root_reward": float(tree.get("reward", 0.0) or 0.0),
        "root_reward_std": float(tree.get("reward_std", 0.0) or 0.0),
        "tree_construction_seconds": tree.get("tree_construction_seconds"),
        "per_depth": per_depth,
    }


class TreeDemoLogger:
    def __init__(
        self,
        demos_dir: Optional[str | Path],
        *,
        demo_examples_per_tree: int = 4,
        full_tree_every_n_trees: int = 25,
        full_tree_max_trees: int = 5,
        print_stats: bool = True,
    ) -> None:
        self.dir = Path(demos_dir) if demos_dir else None
        self.n_each = max(int(demo_examples_per_tree), 0)
        self.full_every_n = int(full_tree_every_n_trees)
        self.full_max = int(full_tree_max_trees)
        self.print_stats = print_stats
        self._seen = 0
        self._full_dumped = 0
        if self.dir is not None:
            (self.dir / "full_trees").mkdir(parents=True, exist_ok=True)
            self._jsonl = (self.dir / "demos.jsonl").open("a", buffering=1)
            self._md = (self.dir / "demos.md").open("a", buffering=1)
        else:
            self._jsonl = self._md = None

    def _index_by_seg_id(self, tree: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        idx: Dict[str, Dict[str, Any]] = {}
        for n in tpl.iter_tree_nodes(tree):
            sid = n.get("gear_segment_id")
            if sid is not None:
                idx[sid] = n
        return idx

    def log_tree(self, tree: Dict[str, Any], question_id: Any) -> Dict[str, Any]:
        self._seen += 1
        basic = basic_tree_stats(tree)
        stored_stats = dict(tree.get("gear_stats", {}) or {})
        try:
            derived_stats = lh.aggregate_tree_stats(tree)
        except Exception:
            derived_stats = {}
        # Runtime counters stored by rollout/queue code take precedence over
        # recomputed summaries when the same metric key exists.
        gear_stats = {**derived_stats, **stored_stats}
        try:
            per_depth = lh.per_depth_action_counts(tree)
        except Exception:
            per_depth = {}
        index = self._index_by_seg_id(tree)
        try:
            demo_rows = lh.collect_demo_rows(tree, index, question_id, self.n_each)
        except Exception:
            demo_rows = {}

        stats = {**gear_stats, "gear/basic": basic}

        if self.print_stats:
            pd = " ".join(
                f"d{d.split('_')[1]}:n={v['n']},r={v['reward_mean']:.2f}"
                for d, v in basic["per_depth"].items()
            )
            print(
                f"[tree #{self._seen} q={question_id}] nodes={basic['num_nodes']} "
                f"leaves={basic['num_leaves']} depth={basic['max_depth']} "
                f"root_reward={basic['root_reward']:.3f} "
                f"build={basic['tree_construction_seconds']} | {pd}",
                flush=True,
            )

        if self._jsonl is not None:
            record = lh.to_jsonl_record(
                self._seen, question_id, stats, per_depth, demo_rows,
                tree_construction_seconds=tree.get("tree_construction_seconds"),
            )
            self._jsonl.write(json.dumps(record, default=str) + "\n")
            self._md.write(lh.render_md_section(self._seen, question_id, stats, demo_rows))

            # One full example tree, rate-limited (matches treetune).
            if (
                self.full_every_n > 0
                and self._full_dumped < self.full_max
                and (self._seen - 1) % self.full_every_n == 0
            ):
                tpl.write_json(
                    self.dir / "full_trees" / f"tree_{self._seen}.json",
                    tpl.serialize_full_tree(tree),
                )
                self._md.write(
                    tpl.render_full_tree_markdown(tree, tree_idx=self._seen, question_id=question_id)
                )
                self._full_dumped += 1

            persist_vdra_artifacts(
                self.dir,
                tree,
                run_id=str(tree.get("run_id", "verl")),
                tree_id=str(question_id),
                queue_flushes=tree.get("vdra_queue_flush_records") or [],
                run_manifest={
                    "algorithm_requested": tree.get("gear_algorithm_mode", "gear_tree"),
                    "algorithm_executed": tree.get("gear_algorithm_mode", "gear_tree"),
                    "run_valid_for_main_results": True,
                    "allocation_scope": tree.get("vdra_allocation_scope", "one_tree"),
                    "budget_mode": tree.get("vdra_budget_mode", "fixed_main"),
                    "budget_claim": budget_claim_for_mode(tree.get("vdra_budget_mode", "fixed_main")),
                    "compute_proxy_definition": COMPUTE_PROXY_DEFINITION,
                },
            )

        return {"tree_idx": self._seen, **basic}

    def close(self) -> None:
        for h in (self._jsonl, self._md):
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass
