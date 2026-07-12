"""GEAR episode generator.

Subclasses `HybridEpisodeGenerator` from SPO and only overrides
`extract_edges_from_tree` so we can:

  * skip pruned children (no edge — segment contributes nothing to PPO),
  * for shared children, inherit `value` / `reward` from the share target
    instead of using the segment's own (NaN) reward,
  * carry through `gear_action`, `gear_share_target`, `gear_tv_m`,
    `gear_gap_m` so downstream metric loggers can quote them.

Everything else — replay buffer, _add_logprobs_to_edges, PPO collation —
is inherited unchanged so the trainer behaviour is identical to SPO.

The corresponding `*_strategy` registration lets configs use
`type: 'gear_episode_generator'`.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from treetune.episode_generators import EpisodeGenerator
from treetune.episode_generators.hybrid_episode_generator import (
    HybridEpisodeGenerator,
)
from treetune.episode_generators.tree_update_modes import compute_tree_update_values
from treetune.logging_utils import get_logger

logger = get_logger(__name__)


from treetune.gear.logging_helpers import (
    BUDGET_DEMO_COLUMNS,
    DEMO_COLUMNS,
    collect_demo_rows,
    per_depth_action_counts,
    render_md_section,
    to_jsonl_record,
)
from treetune.gear.tree_policy_logging import (
    build_run_manifest,
    branch_factors_from_shape,
    format_run_banner,
    render_full_tree_markdown,
    serialize_full_tree,
    should_log_full_tree,
    write_json,
)


@EpisodeGenerator.register("gear_episode_generator")
class GEAREpisodeGenerator(HybridEpisodeGenerator):
    """Tree → edges with online Share/Prune awareness."""

    def __init__(
        self,
        gear_zero_advantage_when_pruned: bool = True,
        gear_emit_pruned_edges: bool = False,
        gear_share_inherit: str = "value_and_reward",  # or "value_only"
        gear_demo_examples_per_tree: int = 2,  # how many SHARE / PRUNE demos to log per tree
        gear_demos_dir: Optional[
            str
        ] = None,  # absolute path; else exp_root/gear_demos
        gear_log_demos_to_wandb: bool = False,  # for offline servers, default off
        gear_log_reward_variance_nodes: bool = False,
        gear_full_tree_demo_every_n_trees: int = 0,
        gear_full_tree_demo_max_trees: int = 5,
        gear_tree_policy_algorithm_name: str = "gear_spo",
        gear_tree_policy_segmentation_type: str = "spo_step",
        gear_tree_policy_tree_shape: Optional[str] = None,
        gear_tree_policy_tree_m: Optional[int] = None,
        gear_print_run_manifest: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gear_zero_advantage_when_pruned = gear_zero_advantage_when_pruned
        self.gear_emit_pruned_edges = gear_emit_pruned_edges
        self.gear_share_inherit = gear_share_inherit
        self.gear_demo_examples_per_tree = int(gear_demo_examples_per_tree)
        self.gear_log_demos_to_wandb = bool(gear_log_demos_to_wandb)
        self.gear_log_reward_variance_nodes = bool(gear_log_reward_variance_nodes)
        self.gear_full_tree_demo_every_n_trees = int(gear_full_tree_demo_every_n_trees)
        self.gear_full_tree_demo_max_trees = int(gear_full_tree_demo_max_trees)
        self.gear_tree_policy_algorithm_name = gear_tree_policy_algorithm_name
        self.gear_tree_policy_segmentation_type = gear_tree_policy_segmentation_type
        self.gear_tree_policy_tree_shape = gear_tree_policy_tree_shape
        self.gear_tree_policy_tree_m = gear_tree_policy_tree_m
        self.gear_print_run_manifest = bool(gear_print_run_manifest)
        # Where to dump local-file demos. Resolves on first use because
        # exp_root is only set after super().__init__ on some SPO branches.
        self._gear_demos_dir_override = gear_demos_dir
        self._gear_demos_dir_resolved = None  # type: Optional[Any]
        self._gear_jsonl_handle = None
        self._gear_md_handle = None
        self._gear_full_tree_jsonl_handle = None
        self._gear_full_tree_md_handle = None
        self._gear_run_manifest_written = False
        self._gear_run_manifest_printed = False
        self._vdra_dispersion_C_jsonl_handle = None
        self._vdra_dispersion_C_csv_handle = None
        self._vdra_dispersion_C_csv_writer = None
        self._tree_seen = 0

    # ------------------------------------------------------------------
    # Override edge extraction
    # ------------------------------------------------------------------

    def extract_edges_from_tree(
        self,
        tree,
        adv_method: str = "rloo",
        only_adv_greater_than_zero: bool = True,
        use_hard_estimation: bool = False,
        tree_update_mode: str = "spo",
        treepo_global_weight: float = 0.5,
        treerl_gamma: float = 0.9,
    ) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        data_instance = tree["_request_object"]
        question_id = data_instance["_treetune__idx"]
        root_reward = tree.get("reward", 0.0) or 0.0

        # Index every node so SHARE children can dereference their target.
        index_by_seg_id: Dict[str, Dict[str, Any]] = {}

        def collect(node):
            seg_id = node.get("gear_segment_id")
            if seg_id is not None:
                index_by_seg_id[seg_id] = node
            for ch in node.get("children", []) or []:
                collect(ch)

        tree_copy = copy.deepcopy(tree)
        collect(tree_copy)

        gear_stats = tree_copy.get("gear_stats", {})
        per_depth = self._per_depth_action_counts(tree_copy)
        demo_rows = collect_demo_rows(
            tree_copy,
            index_by_seg_id,
            question_id=question_id,
            n_each=max(self.gear_demo_examples_per_tree, 0),
        )
        self._tree_seen += 1

        # ---- Local-file demo dump (works offline, no wandb required) -----
        self._dump_demos_to_disk(
            tree_idx=self._tree_seen,
            question_id=question_id,
            stats=gear_stats,
            per_depth=per_depth,
            demo_rows=demo_rows,
            tree_construction_seconds=tree_copy.get(
                "tree_construction_seconds",
                tree_copy.get("gear_tree_construction_seconds"),
            ),
        )
        self._dump_full_tree_to_disk(
            tree=tree_copy,
            tree_idx=self._tree_seen,
            question_id=question_id,
        )

        # ---- Optional wandb scalar+table logging --------------------------
        if gear_stats or per_depth:
            reward_variance_summary = self._summarize_reward_variance_nodes(tree_copy)
            log_entry = {
                **gear_stats,
                "gear/tree_idx": self._tree_seen,
                **reward_variance_summary,
                **per_depth,
                **(
                    {"gear/n_budget_demos_in_tree": len(demo_rows.get("budget", []))}
                    if tree_copy.get("gear_algorithm_mode") == "budget_allocation"
                    else {
                        "gear/n_share_demos_in_tree": len(demo_rows.get("share", [])),
                        "gear/n_prune_demos_in_tree": len(demo_rows.get("prune", [])),
                    }
                ),
            }
            if self.gear_log_demos_to_wandb:
                table = self._maybe_build_wandb_table(demo_rows)
                if table is not None:
                    log_entry["gear/demos"] = table
            self._cloud_log(log_entry)

        self._dump_reward_variance_nodes_to_disk(
            tree=tree_copy,
            tree_idx=self._tree_seen,
            question_id=question_id,
        )

        def BoK(value, bok=4):
            return 1 - (1 - value) ** bok

        def resolve_reward(node, fallback_parent_reward: float) -> Optional[float]:
            r = node.get("reward")
            if r is None:
                return None
            if isinstance(r, float) and np.isnan(r):
                action = node.get("gear_action")
                if action == "share":
                    target_id = node.get("gear_share_target")
                    if target_id is not None and target_id in index_by_seg_id:
                        target_r = index_by_seg_id[target_id].get("reward")
                        if target_r is not None and not (
                            isinstance(target_r, float) and np.isnan(target_r)
                        ):
                            return float(target_r)
                    return float(fallback_parent_reward)
                if action == "prune":
                    eps = node.get("gear_prune_value_eps")
                    if eps is not None:
                        key = str(node.get("gear_segment_id", node.get("text", "")))
                        digest = hashlib.sha256(key.encode("utf-8")).digest()
                        u = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
                        return float(fallback_parent_reward) + (2.0 * u - 1.0) * float(
                            eps
                        )
                    return float(fallback_parent_reward)
                return None
            return float(r)

        def dfs(node, parent=None):
            if parent is not None:
                query_text = parent["full_text"]
                response_text = node["text"]
                parent_reward = parent.get("reward", 0.0) or 0.0
                parent_reward_std = parent.get("reward_std", 0.0) or 0.0

                child_reward = resolve_reward(node, parent_reward)
                if child_reward is None:
                    # Unable to resolve; skip edge to avoid corrupting PPO.
                    pass
                else:
                    leaf = node.get("leaf", False)
                    gear_action = node.get("gear_action", "expand")
                    is_pruned = gear_action == "prune"
                    is_shared = gear_action == "share"

                    if is_pruned and not self.gear_emit_pruned_edges:
                        # Drop the edge entirely - PPO does not see it.
                        pass
                    else:
                        update_values = compute_tree_update_values(
                            child_reward=child_reward,
                            parent_reward=parent_reward,
                            root_reward=root_reward,
                            parent_reward_std=parent_reward_std,
                            adv_method=adv_method,
                            mode=tree_update_mode,
                            treepo_global_weight=treepo_global_weight,
                            treerl_gamma=treerl_gamma,
                        )
                        advantage = update_values["advantage"]
                        value = update_values["value"]

                        if is_pruned and self.gear_zero_advantage_when_pruned:
                            advantage = 0.0
                            update_values["advantage"] = 0.0

                        prover_advantage = BoK(child_reward) - BoK(parent_reward)
                        pav_advantage = advantage + prover_advantage

                        keep = True
                        if (
                            only_adv_greater_than_zero
                            and pav_advantage == 0
                            and not is_pruned
                        ):
                            keep = False

                        if keep and len(response_text) > 0:
                            edges.append(
                                {
                                    "question_id": question_id,
                                    "instance": data_instance,
                                    "query_text": query_text,
                                    "response_text": response_text,
                                    "advantage": advantage,
                                    "prover_advantage": prover_advantage,
                                    "value": value,
                                    "leaf": leaf,
                                    "reward": child_reward,
                                    **update_values,
                                    "gear_action": gear_action,
                                    "gear_share_target": node.get(
                                        "gear_share_target"
                                    ),
                                    "gear_tv_m": node.get("gear_tv_m"),
                                    "gear_gap_m": node.get("gear_gap_m"),
                                }
                            )

            for child in node.get("children", []) or []:
                dfs(child, node)
            node.pop("children", None)

        dfs(tree_copy)
        # Round-trip through json so downstream Datasets.from_list works
        # (matches SPO behaviour).
        edges = json.loads(json.dumps(edges, default=lambda o: None))
        return edges

    # ------------------------------------------------------------------
    # Logging helpers (thin wrappers around module-level pure helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _per_depth_action_counts(tree) -> Dict[str, float]:
        return per_depth_action_counts(tree)

    def _maybe_build_wandb_table(self, demo_rows):
        budget_rows = demo_rows.get("budget", [])
        share_prune_rows = demo_rows.get("share", []) + demo_rows.get("prune", [])
        if not (budget_rows or share_prune_rows):
            return None
        try:
            import wandb  # type: ignore
        except ImportError:
            return None
        if budget_rows:
            table = wandb.Table(columns=BUDGET_DEMO_COLUMNS)
            for r in budget_rows:
                table.add_data(*r)
            return table
        table = wandb.Table(columns=DEMO_COLUMNS)
        for r in share_prune_rows:
            table.add_data(*r)
        return table

    # ------------------------------------------------------------------
    # Offline-friendly file dump
    # ------------------------------------------------------------------

    def _resolve_demos_dir(self):
        if self._gear_demos_dir_resolved is not None:
            return self._gear_demos_dir_resolved

        from pathlib import Path

        if self._gear_demos_dir_override:
            base = Path(self._gear_demos_dir_override)
        elif getattr(self, "exp_root", None) is not None:
            base = Path(self.exp_root) / "gear_demos"
        else:
            base = Path.cwd() / "gear_demos"

        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"GEAR: could not create demos dir {base}: {exc}")
            self._gear_demos_dir_resolved = False
            return False
        self._gear_demos_dir_resolved = base
        return base

    def _open_demo_handles(self):
        base = self._resolve_demos_dir()
        if base is False:
            return None, None
        if self._gear_jsonl_handle is None:
            self._gear_jsonl_handle = (base / "demos.jsonl").open("a", buffering=1)
        if self._gear_md_handle is None:
            self._gear_md_handle = (base / "demos.md").open("a", buffering=1)
            if self._gear_md_handle.tell() == 0:
                self._gear_md_handle.write(
                    "# GEAR SHARE / PRUNE demos\n\n"
                    "One section per tree. `tail -F demos.md` to follow live.\n\n"
                )
        return self._gear_jsonl_handle, self._gear_md_handle

    def _open_reward_variance_handles(self):
        base = self._resolve_demos_dir()
        if base is False:
            return None, None
        if self._vdra_dispersion_C_jsonl_handle is None:
            self._vdra_dispersion_C_jsonl_handle = (
                base / "reward_variance_nodes.jsonl"
            ).open("a", buffering=1)
        if self._vdra_dispersion_C_csv_handle is None:
            csv_path = base / "reward_variance_nodes.csv"
            needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
            self._vdra_dispersion_C_csv_handle = csv_path.open(
                "a", buffering=1, newline=""
            )
            self._vdra_dispersion_C_csv_writer = csv.DictWriter(
                self._vdra_dispersion_C_csv_handle,
                fieldnames=self._reward_variance_fieldnames(),
                extrasaction="ignore",
            )
            if needs_header:
                self._vdra_dispersion_C_csv_writer.writeheader()
        return (
            self._vdra_dispersion_C_jsonl_handle,
            self._vdra_dispersion_C_csv_writer,
        )

    def _open_full_tree_handles(self):
        base = self._resolve_demos_dir()
        if base is False:
            return None, None
        if self._gear_full_tree_jsonl_handle is None:
            self._gear_full_tree_jsonl_handle = (
                base / "full_trees.jsonl"
            ).open("a", buffering=1)
        if self._gear_full_tree_md_handle is None:
            self._gear_full_tree_md_handle = (base / "full_trees.md").open(
                "a", buffering=1
            )
            if self._gear_full_tree_md_handle.tell() == 0:
                self._gear_full_tree_md_handle.write(
                    "# Full Tree Demos\n\n"
                    "Rate-limited complete tree snapshots. Text is not truncated.\n\n"
                )
        return self._gear_full_tree_jsonl_handle, self._gear_full_tree_md_handle

    def _build_run_manifest_from_tree(self, tree) -> Dict[str, Any]:
        tree_shape = self.gear_tree_policy_tree_shape
        branch_factors = {}
        if tree_shape:
            try:
                branch_factors = branch_factors_from_shape(tree_shape)
            except ValueError:
                branch_factors = {}
        if not branch_factors:
            branch_factors = {
                int(k): int(v)
                for k, v in (tree.get("gear_branch_factor_by_depth") or {}).items()
            }
        allocation_mode = tree.get("gear_allocation_mode", "budget_allocation")
        allocation_enabled = allocation_mode != "none"
        return build_run_manifest(
            algorithm_name=self.gear_tree_policy_algorithm_name,
            segmentation_type=self.gear_tree_policy_segmentation_type,
            allocation_type=allocation_mode,
            pruning_enabled=allocation_mode in {"budget_allocation", "prune_only"},
            allocation_enabled=allocation_enabled,
            k_algorithm=tree.get("gear_k_algorithm"),
            tree_shape=tree_shape,
            tree_m=self.gear_tree_policy_tree_m or tree.get("M"),
            branch_factors=branch_factors,
            use_residual_budget=tree.get("gear_use_residual_budget"),
            root_allocation=tree.get("gear_root_allocation"),
            n_min=tree.get("gear_n_min"),
            training=True,
            extra={
                "tree_update_mode": self.tree_update_mode,
                "treepo_global_weight": self.treepo_global_weight,
                "treerl_gamma": self.treerl_gamma,
            },
        )

    def _write_run_manifest_once(self, manifest: Dict[str, Any]) -> None:
        base = self._resolve_demos_dir()
        if base is False:
            return
        if self.gear_print_run_manifest and not self._gear_run_manifest_printed:
            print(format_run_banner(manifest, prefix="[tree-policy]"), flush=True)
            self._gear_run_manifest_printed = True
        if self._gear_run_manifest_written:
            return
        try:
            write_json(base / "run_manifest.json", manifest)
            self._gear_run_manifest_written = True
        except Exception as exc:
            logger.warning(f"GEAR: failed to write run_manifest.json: {exc}")

    def _dump_full_tree_to_disk(self, tree, tree_idx: int, question_id) -> None:
        manifest = self._build_run_manifest_from_tree(tree)
        self._write_run_manifest_once(manifest)
        if not should_log_full_tree(
            tree_idx,
            every_n_trees=self.gear_full_tree_demo_every_n_trees,
            max_trees=self.gear_full_tree_demo_max_trees,
        ):
            return
        jsonl, md = self._open_full_tree_handles()
        if jsonl is None or md is None:
            return
        full_tree = serialize_full_tree(tree)
        record = {
            "tree_idx": tree_idx,
            "question_id": question_id,
            "manifest": manifest,
            "tree": full_tree,
        }
        try:
            jsonl.write(json.dumps(record, default=str) + "\n")
            md.write(
                render_full_tree_markdown(
                    full_tree, tree_idx=tree_idx, question_id=question_id
                )
            )
        except Exception as exc:
            logger.warning(f"GEAR: failed to append full tree demo: {exc}")

    @staticmethod
    def _reward_variance_fieldnames() -> List[str]:
        return [
            "tree_idx",
            "question_id",
            "depth",
            "seg_id",
            "parent_seg_id",
            "action",
            "reward",
            "reward_std",
            "empirical_child_reward_variance",
            "vdra_dispersion_C",
            "vdra_dispersion_C_legacy_sigma2",
            "vdra_dispersion_C_legacy_sigma4",
            "gear_tv_pair_count",
            "gear_tv_support_size",
            "gear_allocated_branch_factor",
            "gear_budget_weight",
            "gear_discarded_budget_candidates",
            "n_children",
        ]

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _iter_reward_variance_rows(self, tree, tree_idx: int, question_id):
        stack = [tree]
        while stack:
            node = stack.pop()
            children = node.get("children") or []
            child_rewards = [
                self._safe_float(ch.get("reward"))
                for ch in children
                if self._safe_float(ch.get("reward")) is not None
            ]
            empirical_var = None
            if child_rewards:
                mean = sum(child_rewards) / len(child_rewards)
                empirical_var = sum((r - mean) ** 2 for r in child_rewards) / len(
                    child_rewards
                )
            row = {
                "tree_idx": tree_idx,
                "question_id": question_id,
                "depth": node.get("gear_depth", node.get("depth")),
                "seg_id": node.get("gear_segment_id"),
                "parent_seg_id": node.get("gear_parent_segment_id"),
                "action": node.get("gear_action"),
                "reward": self._safe_float(node.get("reward")),
                "reward_std": self._safe_float(node.get("reward_std")),
                "empirical_child_reward_variance": empirical_var,
                "vdra_dispersion_C": self._safe_float(
                    node.get("vdra_dispersion_C")
                ),
                "vdra_dispersion_C_legacy_sigma2": self._safe_float(node.get("vdra_dispersion_C_legacy_sigma2")),
                "vdra_dispersion_C_legacy_sigma4": self._safe_float(node.get("vdra_dispersion_C_legacy_sigma4")),
                "gear_tv_pair_count": node.get("gear_tv_pair_count"),
                "gear_tv_support_size": node.get("gear_tv_support_size"),
                "gear_allocated_branch_factor": node.get(
                    "gear_allocated_branch_factor"
                ),
                "gear_budget_weight": self._safe_float(
                    node.get("gear_budget_weight")
                ),
                "gear_discarded_budget_candidates": node.get(
                    "gear_discarded_budget_candidates"
                ),
                "n_children": len(children),
            }
            if row["vdra_dispersion_C"] is not None or row["reward"] is not None:
                yield row
            stack.extend(reversed(children))

    def _summarize_reward_variance_nodes(self, tree) -> Dict[str, float]:
        rows = list(
            self._iter_reward_variance_rows(
                tree=tree,
                tree_idx=self._tree_seen,
                question_id="",
            )
        )
        variances = [
            r["vdra_dispersion_C"]
            for r in rows
            if r["vdra_dispersion_C"] is not None
        ]
        rewards = [r["reward"] for r in rows if r["reward"] is not None]
        out: Dict[str, float] = {
            "gear/reward_variance_nodes/n": float(len(rows)),
            "gear/reward_variance_nodes/n_with_variance": float(len(variances)),
        }
        if variances:
            out["gear/reward_variance_nodes/mean_sigma2"] = float(
                sum(variances) / len(variances)
            )
            out["gear/reward_variance_nodes/max_sigma2"] = float(max(variances))
        if rewards:
            out["gear/reward_variance_nodes/mean_reward"] = float(
                sum(rewards) / len(rewards)
            )
        return out

    def _dump_reward_variance_nodes_to_disk(
        self,
        tree,
        tree_idx: int,
        question_id,
    ) -> None:
        if not self.gear_log_reward_variance_nodes:
            return

        jsonl, csv_writer = self._open_reward_variance_handles()
        if jsonl is None or csv_writer is None:
            return

        for row in self._iter_reward_variance_rows(tree, tree_idx, question_id):
            try:
                jsonl.write(json.dumps(row, default=str) + "\n")
                csv_writer.writerow(row)
            except Exception as exc:
                logger.warning(
                    f"GEAR: failed to append reward variance node record: {exc}"
                )
                return

    def _dump_demos_to_disk(
        self,
        tree_idx: int,
        question_id,
        stats: Dict[str, Any],
        per_depth: Dict[str, float],
        demo_rows: Dict[str, List[List[Any]]],
        tree_construction_seconds: Optional[float] = None,
    ) -> None:
        if max(self.gear_demo_examples_per_tree, 0) == 0 and not stats:
            return

        jsonl, md = self._open_demo_handles()
        if jsonl is None:
            return  # silent: never block training because logging failed

        # JSONL row: machine-readable. One line per tree.
        record = to_jsonl_record(
            tree_idx=tree_idx,
            question_id=question_id,
            stats=stats,
            per_depth=per_depth,
            demo_rows=demo_rows,
            tree_construction_seconds=tree_construction_seconds,
        )
        try:
            jsonl.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning(f"GEAR: failed to append demos.jsonl: {exc}")

        # Markdown: human-readable. One section per tree, only if there's
        # something interesting (rates or demos) to show.
        if not (
            demo_rows.get("share") or demo_rows.get("prune") or demo_rows.get("budget")
        ):
            return
        try:
            md.write(render_md_section(tree_idx, question_id, stats, demo_rows))
        except Exception as exc:
            logger.warning(f"GEAR: failed to append demos.md: {exc}")

    def __del__(self):
        for h in (
            self._gear_jsonl_handle,
            self._gear_md_handle,
            self._gear_full_tree_jsonl_handle,
            self._gear_full_tree_md_handle,
            self._vdra_dispersion_C_jsonl_handle,
            self._vdra_dispersion_C_csv_handle,
        ):
            try:
                if h is not None:
                    h.close()
            except Exception:
                pass
