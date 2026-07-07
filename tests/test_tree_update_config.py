import json
from pathlib import Path

import _jsonnet

from treetune.gear.tree_policy_logging import format_run_banner


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _render_config(filename: str):
    return json.loads(
        _jsonnet.evaluate_file(
            str(CONFIGS / filename),
            ext_vars={
                "APP_SEED": "42",
                "APP_DISABLE_FLASH_ATTENTION": "0",
            },
        )
    )


def test_gear_tree_configs_default_to_spo_update_mode():
    for filename in (
        "polIter_qwen1_5b_base_gear_tree_MATH.jsonnet",
        "polIter_deepseekR1Qwen_gear_tree_MATH.jsonnet",
        "polIter_qwen1_5b_base_gear_spo_chain_MATH.jsonnet",
        "polIter_deepseekR1Qwen_gear_spo_chain_MATH.jsonnet",
    ):
        config = _render_config(filename)
        episode_generator = config["episode_generator"]

        assert episode_generator["tree_update_mode"] == "spo", filename
        assert episode_generator["treepo_global_weight"] == 0.5, filename
        assert episode_generator["treerl_gamma"] == 0.9, filename


def test_update_objective_ablation_configs_exist():
    assert (CONFIGS / "ablations/abl_treepo_original_update.jsonnet").exists()
    assert (CONFIGS / "ablations/abl_treerl_original_update.jsonnet").exists()


def test_training_banner_prints_tree_update_mode():
    banner = format_run_banner(
        {
            "mode": "training",
            "training": True,
            "algorithm_name": "gear_treepo",
            "tree_update_mode": "treepo_original",
            "segmentation_type": "treepo_fixed_step",
            "allocation_type": "budget_allocation",
            "pruning_enabled": True,
            "tree_shape": "666",
            "tree_m": 600,
            "max_depth": 3,
            "branch_factors": {"0": 6},
            "k_algorithm": "hierarchical",
            "use_residual_budget": True,
            "root_allocation": True,
        }
    )

    assert "[tree-policy] tree_update_mode=treepo_original" in banner
