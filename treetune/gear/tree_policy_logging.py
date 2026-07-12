"""Tree-policy run manifests and full-tree logging helpers.

These helpers are deliberately independent from trainers.  They are used both
by normal GEAR runs and by the analysis-only pruning pipeline.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def branch_factors_from_shape(shape: Optional[str]) -> Dict[int, int]:
    if not shape:
        return {}
    if not str(shape).isdigit() or "0" in str(shape):
        raise ValueError(
            f"Invalid tree shape {shape!r}: expected digits 1-9, e.g. '666'."
        )
    return {idx: int(ch) for idx, ch in enumerate(str(shape))}


def branch_factors_from_strategy(strategy: Any) -> Dict[int, int]:
    branch_factors = getattr(strategy, "branch_factors", None)
    if not branch_factors:
        return {}
    out: Dict[int, int] = {}
    for item in branch_factors:
        try:
            out[int(item["depth"])] = int(item["branch_factor"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_run_manifest(
    *,
    algorithm_name: str,
    segmentation_type: str,
    allocation_type: str,
    pruning_enabled: bool,
    allocation_enabled: bool,
    k_algorithm: Optional[str] = None,
    tree_shape: Optional[str] = None,
    tree_m: Optional[int] = None,
    branch_factors: Optional[Mapping[int, int]] = None,
    use_residual_budget: Optional[bool] = None,
    root_allocation: Optional[bool] = None,
    n_min: Optional[int] = None,
    backend: Optional[str] = None,
    training: bool = True,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    branch_factors = dict(branch_factors or branch_factors_from_shape(tree_shape))
    manifest: Dict[str, Any] = {
        "mode": "training" if training else "analysis_only",
        "training": bool(training),
        "algorithm_name": algorithm_name,
        "segmentation_type": segmentation_type,
        "allocation_type": allocation_type,
        "allocation_enabled": bool(allocation_enabled),
        "pruning_enabled": bool(pruning_enabled),
        "k_algorithm": k_algorithm,
        "tree_shape": tree_shape,
        "tree_m": tree_m,
        "max_depth": len(branch_factors) if branch_factors else None,
        "branch_factors": {str(k): int(v) for k, v in sorted(branch_factors.items())},
        "use_residual_budget": use_residual_budget,
        "root_allocation": root_allocation,
        "n_min": n_min,
    }
    if backend is not None:
        manifest["backend"] = backend
    if extra:
        manifest.update(dict(extra))
    return manifest


def format_run_banner(
    manifest: Mapping[str, Any],
    *,
    prefix: str = "[tree-policy]",
) -> str:
    branch_factors = manifest.get("branch_factors") or {}
    branch_repr = "{" + ",".join(f"{k}:{v}" for k, v in branch_factors.items()) + "}"
    lines = [
        f"{prefix} mode={manifest.get('mode')} training={str(manifest.get('training')).lower()}",
    ]
    if manifest.get("backend") is not None:
        lines.append(f"{prefix} backend={manifest.get('backend')}")
    lines.extend(
        [
            f"{prefix} algorithm={manifest.get('algorithm_name')}",
            f"{prefix} tree_update_mode={manifest.get('tree_update_mode', 'spo')}",
            f"{prefix} segmentation={manifest.get('segmentation_type')}",
            f"{prefix} allocation={manifest.get('allocation_type')}",
            f"{prefix} pruning={str(manifest.get('pruning_enabled')).lower()}",
            (
                f"{prefix} tree_shape={manifest.get('tree_shape')} "
                f"tree_m={manifest.get('tree_m')} "
                f"depth={manifest.get('max_depth')} branch_factors={branch_repr}"
            ),
            (
                f"{prefix} k_algorithm={manifest.get('k_algorithm')} "
                f"residual_budget={manifest.get('use_residual_budget')} "
                f"root_allocation={manifest.get('root_allocation')}"
            ),
        ]
    )
    return "\n".join(lines)


def print_run_banner(manifest: Mapping[str, Any], *, prefix: str = "[tree-policy]") -> None:
    print(format_run_banner(manifest, prefix=prefix), flush=True)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def serialize_full_tree(tree: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe deep tree without truncating text fields."""

    def convert(node: Any) -> Any:
        if isinstance(node, Mapping):
            out: Dict[str, Any] = {}
            for key, value in node.items():
                if key == "_request_object":
                    out["request_object"] = _json_safe(value)
                elif key == "children":
                    out["children"] = [convert(child) for child in value or []]
                else:
                    out[str(key)] = convert(value)
            return out
        if isinstance(node, list):
            return [convert(item) for item in node]
        if isinstance(node, tuple):
            return [convert(item) for item in node]
        if isinstance(node, dict):
            return {str(k): convert(v) for k, v in node.items()}
        return _json_safe(node)

    return convert(copy.deepcopy(tree))


def iter_tree_nodes(tree: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.get("children") or []))


def should_log_full_tree(
    tree_idx: int,
    *,
    every_n_trees: int = 0,
    max_trees: int = 5,
) -> bool:
    tree_idx = int(tree_idx)
    every_n_trees = max(int(every_n_trees), 0)
    max_trees = max(int(max_trees), 0)
    if max_trees and tree_idx <= max_trees:
        return True
    return every_n_trees > 0 and tree_idx % every_n_trees == 0


def render_full_tree_markdown(tree: Mapping[str, Any], *, tree_idx: int, question_id) -> str:
    lines = [f"## Full Tree #{tree_idx}  (question_id={question_id})\n"]

    def render_node(node: Mapping[str, Any], indent: int = 0) -> None:
        pad = "  " * indent
        seg_id = node.get("gear_segment_id", node.get("segment_id", "root"))
        depth = node.get("gear_depth", node.get("depth"))
        action = node.get("gear_action", node.get("action", ""))
        reward = node.get("reward")
        predicted_k = node.get("gear_predicted_k")
        allocated = node.get("gear_allocated_branch_factor")
        default_b = node.get("gear_default_branch_factor")
        lines.append(
            f"{pad}- node={seg_id} depth={depth} action={action} "
            f"reward={reward} predicted_k={predicted_k} "
            f"allocated={allocated} default_b={default_b}\n"
        )
        lines.append(f"{pad}  text:\n\n{pad}  ```text\n{node.get('text', '')}\n{pad}  ```\n")
        lines.append(
            f"{pad}  full_text:\n\n{pad}  ```text\n{node.get('full_text', '')}\n{pad}  ```\n"
        )
        for child in node.get("children") or []:
            render_node(child, indent + 1)

    render_node(tree)
    lines.append("\n")
    return "".join(lines)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
