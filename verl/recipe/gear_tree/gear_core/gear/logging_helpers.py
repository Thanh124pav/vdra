"""Pure helpers used by the GEAR episode generator's wandb logging.

Lives in `core/` (no SPO deps) so it can be unit-tested without installing
the full SPO Python stack.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import pstdev
from typing import Any, Dict, List, Optional

_DEMO_TEXT_TRUNC = 240


DEMO_COLUMNS = [
    "question_id",
    "action",
    "depth",
    "seg_id",
    "parent_text",
    "child_text",
    "target_text",
    "target_seg_id",
    "avg_lp_K",
    "tv_m",
    "gap_m",
    "eta",
    "tau",
]

BUDGET_DEMO_COLUMNS = [
    "question_id",
    "action",
    "depth",
    "seg_id",
    "parent_text",
    "node_text",
    "budget_weight",
    "reward_variance",
    "allocated_branch_factor",
    "built_children",
    "discarded_candidates",
]


def truncate(s: Optional[str], n: int = _DEMO_TEXT_TRUNC) -> str:
    if not s:
        return ""
    s = s.replace("\n", " \\n ")
    return s if len(s) <= n else s[: n - 3] + "..."


def _node_depth(node) -> Optional[int]:
    depth = node.get("gear_depth", node.get("depth"))
    try:
        return int(depth)
    except (TypeError, ValueError):
        return None


def _is_budget_allocation_tree(tree) -> bool:
    return tree.get("gear_algorithm_mode") == "budget_allocation"


def aggregate_tree_stats(
    tree,
    max_depth: Optional[int] = None,
    branch_factor_by_depth: Optional[Dict[int, int]] = None,
) -> Dict[str, float]:
    """Aggregate tree-level GEAR logging metrics.

    Budget-allocation runs intentionally do not emit legacy SHARE/PRUNE rates.
    They report how many child nodes were actually built against the node budget
    assigned by the SPO-style branch-factor schedule.
    """

    if _is_budget_allocation_tree(tree):
        built_nodes = 0
        allocated_budget = 0
        requested_budget = 0
        expandable_parent_nodes = 0
        allocated_branch_factors: List[int] = []
        stack = [tree]
        while stack:
            node = stack.pop()
            children = node.get("children") or []
            allocated = node.get("gear_allocated_branch_factor")
            if allocated is not None:
                allocated_int = int(allocated)
                expandable_parent_nodes += 1
                allocated_budget += allocated_int
                allocated_branch_factors.append(allocated_int)
                built_nodes += len(children)
                depth = _node_depth(node)
                if branch_factor_by_depth is not None and depth is not None:
                    requested_budget += int(branch_factor_by_depth.get(depth, 0))
            stack.extend(children)

        if requested_budget == 0:
            requested_by_depth = tree.get("gear_requested_node_budget_by_depth") or {}
            requested_budget = sum(int(v) for v in requested_by_depth.values())
        if allocated_budget == 0:
            allocated_budget = int(
                sum((tree.get("gear_allocated_branch_factor_by_depth") or {}).values())
            )
        if built_nodes == 0:
            built_nodes = int(
                sum((tree.get("gear_built_nodes_by_depth") or {}).values())
            )

        out: Dict[str, float] = {
            "gear/budget/built_nodes": float(built_nodes),
            "gear/budget/allocated_node_budget": float(allocated_budget),
            "gear/budget/requested_node_budget": float(requested_budget),
            "gear/budget/expandable_parent_nodes": float(expandable_parent_nodes),
        }
        out["gear/budget/built_to_allocated_ratio"] = (
            float(built_nodes) / allocated_budget if allocated_budget > 0 else 0.0
        )
        out["gear/budget/built_to_requested_ratio"] = (
            float(built_nodes) / requested_budget if requested_budget > 0 else 0.0
        )
        out["gear/budget/allocated_branch_factor_mean"] = (
            float(sum(allocated_branch_factors) / len(allocated_branch_factors))
            if allocated_branch_factors
            else 0.0
        )
        out["gear/budget/allocated_branch_factor_min"] = (
            float(min(allocated_branch_factors)) if allocated_branch_factors else 0.0
        )
        out["gear/budget/allocated_branch_factor_max"] = (
            float(max(allocated_branch_factors)) if allocated_branch_factors else 0.0
        )
        out["gear/budget/allocated_branch_factor_std"] = (
            float(pstdev(allocated_branch_factors))
            if len(allocated_branch_factors) > 1
            else 0.0
        )
        return out

    counts: Counter = Counter()
    stack = [tree]
    while stack:
        n = stack.pop()
        a = n.get("gear_action")
        if a is not None:
            counts[a] += 1
        stack.extend(n.get("children") or [])

    expanded = counts.get("expand", 0)
    shared = counts.get("share", 0)
    pruned = counts.get("prune", 0)
    total = expanded + shared + pruned
    if total == 0:
        return {}
    return {
        "gear/expanded_count": expanded,
        "gear/shared_count": shared,
        "gear/pruned_count": pruned,
        "gear/share_rate": shared / total,
        "gear/prune_rate": pruned / total,
    }


def per_depth_action_counts(tree) -> Dict[str, float]:
    """Return per-depth GEAR metrics.

    Budget-allocation runs log SPO-compatible budget utilization by expansion
    depth: distribution statistics for allocated branch factors across parent
    nodes, built child nodes, and built/budget ratios. Legacy SHARE/PRUNE counts
    are emitted only for share-prune mode.
    """

    if _is_budget_allocation_tree(tree):
        branch_factors_by_depth: Dict[int, List[int]] = defaultdict(list)
        built_by_depth: Counter = Counter()
        allocated_by_depth: Counter = Counter()
        requested_by_depth = {
            int(k): int(v)
            for k, v in (tree.get("gear_requested_node_budget_by_depth") or {}).items()
        }
        parent_count_by_depth: Counter = Counter()

        stack = [tree]
        while stack:
            node = stack.pop()
            children = node.get("children") or []
            allocated = node.get("gear_allocated_branch_factor")
            depth = _node_depth(node)
            if allocated is not None and depth is not None:
                allocated_int = int(allocated)
                branch_factors_by_depth[depth].append(allocated_int)
                allocated_by_depth[depth] += allocated_int
                built_by_depth[depth] += len(children)
                parent_count_by_depth[depth] += 1
            stack.extend(children)

        out: Dict[str, float] = {}
        depths = sorted(
            set(branch_factors_by_depth)
            | set(requested_by_depth)
            | set(allocated_by_depth)
            | set(built_by_depth)
        )
        for depth in depths:
            branch_factors = branch_factors_by_depth.get(depth, [])
            allocated = int(allocated_by_depth.get(depth, 0))
            requested = int(requested_by_depth.get(depth, 0))
            built = int(built_by_depth.get(depth, 0))
            out[f"gear/depth_{depth}/budget_parent_nodes"] = int(
                parent_count_by_depth.get(depth, 0)
            )
            out[f"gear/depth_{depth}/allocated_branch_factor_mean"] = (
                float(sum(branch_factors) / len(branch_factors))
                if branch_factors
                else 0.0
            )
            out[f"gear/depth_{depth}/allocated_branch_factor_min"] = (
                float(min(branch_factors)) if branch_factors else 0.0
            )
            out[f"gear/depth_{depth}/allocated_branch_factor_max"] = (
                float(max(branch_factors)) if branch_factors else 0.0
            )
            out[f"gear/depth_{depth}/allocated_branch_factor_std"] = (
                float(pstdev(branch_factors)) if len(branch_factors) > 1 else 0.0
            )
            out[f"gear/depth_{depth}/built_nodes"] = built
            out[f"gear/depth_{depth}/allocated_node_budget"] = allocated
            out[f"gear/depth_{depth}/requested_node_budget"] = requested
            out[f"gear/depth_{depth}/built_to_allocated_ratio"] = (
                built / allocated if allocated > 0 else 0.0
            )
            out[f"gear/depth_{depth}/built_to_requested_ratio"] = (
                built / requested if requested > 0 else 0.0
            )
        return out

    per_depth_count: Dict[int, Counter] = {}
    stack = [tree]
    while stack:
        n = stack.pop()
        d = n.get("gear_depth")
        a = n.get("gear_action")
        if d is not None and a is not None:
            per_depth_count.setdefault(d, Counter())[a] += 1
        stack.extend(n.get("children") or [])

    out: Dict[str, float] = {}
    for d, c in sorted(per_depth_count.items()):
        total = sum(c.values())
        if total == 0:
            continue
        out[f"gear/depth_{d}/n"] = total
        out[f"gear/depth_{d}/expand_count"] = c.get("expand", 0)
        out[f"gear/depth_{d}/share_count"] = c.get("share", 0)
        out[f"gear/depth_{d}/prune_count"] = c.get("prune", 0)
        out[f"gear/depth_{d}/share_rate"] = c.get("share", 0) / total
        out[f"gear/depth_{d}/prune_rate"] = c.get("prune", 0) / total
    return out


def collect_demo_rows(
    tree,
    index_by_seg_id: Dict[str, Dict[str, Any]],
    question_id,
    n_each: int,
) -> Dict[str, List[List[Any]]]:
    """Return up to `n_each` SHARE demos and `n_each` PRUNE demos, each as a
    list of column-aligned cells matching `DEMO_COLUMNS`.
    """

    if _is_budget_allocation_tree(tree):
        budget_rows: List[List[Any]] = []
        stack = [tree]
        while stack:
            n = stack.pop()
            stack.extend(n.get("children") or [])
            allocated = n.get("gear_allocated_branch_factor")
            if allocated is None:
                continue
            depth = _node_depth(n)
            budget_rows.append(
                [
                    str(question_id),
                    "budget",
                    depth if depth is not None else "",
                    str(n.get("gear_segment_id", "")),
                    truncate(
                        index_by_seg_id.get(n.get("gear_parent_segment_id"), {}).get(
                            "text"
                        )
                        or index_by_seg_id.get(
                            n.get("gear_parent_segment_id"), {}
                        ).get("full_text")
                    ),
                    truncate(n.get("text") or n.get("full_text")),
                    float(n.get("gear_budget_weight") or 0.0),
                    (
                        float(n.get("gear_reward_variance") or 0.0)
                        if n.get("gear_reward_variance") is not None
                        else None
                    ),
                    int(allocated),
                    len(n.get("children") or []),
                    int(n.get("gear_discarded_budget_candidates") or 0),
                ]
            )
        return {"share": [], "prune": [], "budget": budget_rows[:n_each]}

    prune_rows: List[List[Any]] = []
    share_rows: List[List[Any]] = []

    stack = [tree]
    while stack:
        n = stack.pop()
        stack.extend(n.get("children") or [])
        action = n.get("gear_action")
        if action not in ("share", "prune"):
            continue

        seg_id = n.get("gear_segment_id", "")
        depth = n.get("gear_depth", "")
        parent_id = n.get("gear_parent_segment_id", "")
        parent = index_by_seg_id.get(parent_id, {})
        target_id = n.get("gear_share_target")
        target = index_by_seg_id.get(target_id, {}) if target_id else {}

        row = [
            str(question_id),
            action,
            int(depth) if isinstance(depth, int) else depth,
            str(seg_id),
            truncate(parent.get("text") or parent.get("full_text")),
            truncate(n.get("text") or n.get("full_text")),
            (
                truncate(target.get("text") or target.get("full_text"))
                if target_id
                else ""
            ),
            str(target_id or ""),
            float(n.get("gear_avg_lp_K") or 0.0),
            (
                float(n.get("gear_tv_m") or 0.0)
                if n.get("gear_tv_m") is not None
                else None
            ),
            (
                float(n.get("gear_gap_m") or 0.0)
                if n.get("gear_gap_m") is not None
                else None
            ),
            float(n.get("gear_eta") or 0.0),
            float(n.get("gear_tau") or 0.0),
        ]
        (prune_rows if action == "prune" else share_rows).append(row)

    return {
        "share": share_rows[:n_each],
        "prune": prune_rows[:n_each],
        "budget": [],
    }


def row_to_dict(row: List[Any], columns: Optional[List[str]] = None) -> Dict[str, Any]:
    return dict(zip(columns or DEMO_COLUMNS, row))


def render_md_section(
    tree_idx: int,
    question_id,
    stats: Dict[str, Any],
    demo_rows: Dict[str, List[List[Any]]],
) -> str:
    """One Markdown section for one tree, ready to append to demos.md."""

    out = [f"## Tree #{tree_idx}  (question_id={question_id})\n"]
    budget_rows = demo_rows.get("budget", [])
    if stats and "gear/budget/built_nodes" in stats:
        built = float(stats.get("gear/budget/built_nodes", 0.0) or 0.0)
        allocated = float(stats.get("gear/budget/allocated_node_budget", 0.0) or 0.0)
        requested = float(stats.get("gear/budget/requested_node_budget", 0.0) or 0.0)
        out.append(
            f"- budget built_nodes: **{built:.0f}** / allocated_node_budget: **{allocated:.0f}** "
            f"(requested_node_budget={requested:.0f})\n"
        )
    elif stats:
        share_rate = float(stats.get("gear/share_rate", 0.0) or 0.0)
        prune_rate = float(stats.get("gear/prune_rate", 0.0) or 0.0)
        out.append(
            f"- share_rate: **{share_rate:.3f}**, prune_rate: **{prune_rate:.3f}**, "
            f"#shared={stats.get('gear/shared_count', 0)}, "
            f"#pruned={stats.get('gear/pruned_count', 0)}, "
            f"#expanded={stats.get('gear/expanded_count', 0)}\n"
        )

    if budget_rows:
        out.append("### Budget allocation demos\n")
        for row in budget_rows:
            d = row_to_dict(row, BUDGET_DEMO_COLUMNS)
            out.append(
                f"- depth={d['depth']}  seg={d['seg_id']}  "
                f"weight={float(d['budget_weight']):.4f}  "
                f"sigma2={float(d['reward_variance'] or 0.0):.4f}  "
                f"allocated={int(d['allocated_branch_factor'])}  "
                f"built={int(d['built_children'])}  "
                f"discarded={int(d['discarded_candidates'])}\n"
            )
            out.append(f"  - node : `{d['node_text']}`\n")
        out.append("\n")

    for label, rows in (("SHARE", demo_rows["share"]), ("PRUNE", demo_rows["prune"])):
        if not rows:
            continue
        out.append(f"### {label} demos\n")
        for row in rows:
            d = row_to_dict(row)
            line = (
                f"- depth={d['depth']}  seg={d['seg_id']}  "
                f"AvgLP_K={float(d['avg_lp_K']):.3f}  "
            )
            if d.get("tv_m") is not None:
                line += f"TV_m={float(d['tv_m']):.3f}  "
            if d.get("gap_m") is not None:
                line += f"gap_m={float(d['gap_m']):.3f}  "
            line += f"(eta={float(d['eta']):.3f}, tau={float(d['tau']):.3f})\n"
            out.append(line)
            out.append(f"  - parent : `{d['parent_text']}`\n")
            out.append(f"  - child  : `{d['child_text']}`\n")
            if d.get("target_text"):
                out.append(
                    f"  - shared->`{d['target_seg_id']}` : `{d['target_text']}`\n"
                )
        out.append("\n")
    out.append("\n")
    return "".join(out)


def to_jsonl_record(
    tree_idx: int,
    question_id,
    stats: Dict[str, Any],
    per_depth: Dict[str, float],
    demo_rows: Dict[str, List[List[Any]]],
    tree_construction_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Pack one tree's metrics + demos into a JSONL-ready dict."""

    record: Dict[str, Any] = {
        "tree_idx": tree_idx,
        "question_id": question_id,
        "stats": stats,
        "per_depth": per_depth,
        "demos": {
            "share": [row_to_dict(r) for r in demo_rows.get("share", [])],
            "prune": [row_to_dict(r) for r in demo_rows.get("prune", [])],
            "budget": [
                row_to_dict(r, BUDGET_DEMO_COLUMNS) for r in demo_rows.get("budget", [])
            ],
        },
    }
    if tree_construction_seconds is not None:
        record["tree_construction_seconds"] = float(tree_construction_seconds)
    return record
