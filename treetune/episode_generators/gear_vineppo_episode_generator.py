"""VinePPO episode generator with GEAR tree-policy logging.

This class intentionally keeps VinePPO's trajectory parsing and advantage
computation unchanged. It only adds the same run-manifest and full-tree demo
artifacts used by GEAR-tree runs, so GEAR-VinePPO experiments can be compared
against GEAR-SPO/SPO-chain without losing full text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset

from treetune.episode_generators import EpisodeGenerator
from treetune.episode_generators.vineppo_episode_generator import (
    VinePPOEpisodeGenerator,
)
from treetune.gear.tree_policy_logging import (
    branch_factors_from_shape,
    build_run_manifest,
    format_run_banner,
    render_full_tree_markdown,
    serialize_full_tree,
    should_log_full_tree,
    write_json,
)

logger = logging.getLogger(__name__)


@EpisodeGenerator.register("gear_vineppo_episode_generator")
class GEARVinePPOEpisodeGenerator(VinePPOEpisodeGenerator):
    """VinePPO with complete GEAR tree snapshots for paper demos."""

    def __init__(
        self,
        gear_full_tree_demo_every_n_trees: int = 0,
        gear_full_tree_demo_max_trees: int = 5,
        gear_tree_policy_algorithm_name: str = "gear_vineppo",
        gear_tree_policy_segmentation_type: str = "vineppo_step",
        gear_tree_policy_tree_shape: Optional[str] = None,
        gear_tree_policy_tree_m: Optional[int] = None,
        gear_demos_dir: Optional[str] = None,
        gear_print_run_manifest: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gear_full_tree_demo_every_n_trees = int(gear_full_tree_demo_every_n_trees)
        self.gear_full_tree_demo_max_trees = int(gear_full_tree_demo_max_trees)
        self.gear_tree_policy_algorithm_name = gear_tree_policy_algorithm_name
        self.gear_tree_policy_segmentation_type = gear_tree_policy_segmentation_type
        self.gear_tree_policy_tree_shape = gear_tree_policy_tree_shape
        self.gear_tree_policy_tree_m = gear_tree_policy_tree_m
        self.gear_print_run_manifest = bool(gear_print_run_manifest)
        self._gear_demos_dir_override = gear_demos_dir
        self._gear_demos_dir_resolved = None
        self._gear_full_tree_jsonl_handle = None
        self._gear_full_tree_md_handle = None
        self._gear_run_manifest_written = False
        self._gear_run_manifest_printed = False
        self._gear_tree_seen = 0

    def _resolve_demos_dir(self):
        if self._gear_demos_dir_resolved is not None:
            return self._gear_demos_dir_resolved
        if self._gear_demos_dir_override:
            base = Path(self._gear_demos_dir_override)
        else:
            exp_root = getattr(self, "exp_root", None) or getattr(
                self, "experiment_root", None
            )
            if exp_root is None:
                self._gear_demos_dir_resolved = False
                return False
            base = exp_root / "gear_demos"
        base.mkdir(parents=True, exist_ok=True)
        self._gear_demos_dir_resolved = base
        return base

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
                    "Rate-limited complete GEAR-VinePPO tree snapshots. "
                    "Text is not truncated.\n\n"
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
        return build_run_manifest(
            algorithm_name=self.gear_tree_policy_algorithm_name,
            segmentation_type=self.gear_tree_policy_segmentation_type,
            allocation_type=allocation_mode,
            pruning_enabled=allocation_mode in {"budget_allocation", "prune_only"},
            allocation_enabled=allocation_mode != "none",
            k_algorithm=tree.get("gear_k_algorithm"),
            tree_shape=tree_shape,
            tree_m=self.gear_tree_policy_tree_m or tree.get("M"),
            branch_factors=branch_factors,
            use_residual_budget=tree.get("gear_use_residual_budget"),
            root_allocation=tree.get("gear_root_allocation"),
            n_min=tree.get("gear_n_min"),
            training=True,
            extra={"tree_update_mode": "vineppo_original"},
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
            logger.warning("GEAR-VinePPO: failed to write run manifest: %s", exc)

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
            logger.warning("GEAR-VinePPO: failed to append full tree demo: %s", exc)

    def _create_trajectories(self, inference_results: Dataset, iteration: int):
        for instance in inference_results:
            tree = json.loads(instance["_treetune__reasoning_tree"])
            self._gear_tree_seen += 1
            self._dump_full_tree_to_disk(
                tree=tree,
                tree_idx=self._gear_tree_seen,
                question_id=instance.get("_treetune__idx", self._gear_tree_seen),
            )
        return super()._create_trajectories(inference_results, iteration)

    def __del__(self):
        for handle in (
            self._gear_full_tree_jsonl_handle,
            self._gear_full_tree_md_handle,
        ):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
