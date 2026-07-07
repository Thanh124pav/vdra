import json
from pathlib import Path

import _jsonnet


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"


def test_gear_spo_chain_configs_apply_shared_gear_overlay():
    for model in ("qwen1_5b_base", "deepseekR1Qwen"):
        config = (CONFIGS / f"polIter_{model}_gear_spo_chain_MATH.jsonnet").read_text()

        assert "import 'gear_defaults.libsonnet'" in config
        assert "import 'gear_overlay.libsonnet'" in config
        assert "tree_policy_algorithm_name: 'gear_spo_chain'" in config
        assert "tree_policy_segmentation_type: 'spo_chain_step'" in config
        assert "skip_near_leaf_expand: false" in config


def test_gear_vineppo_configs_keep_vineppo_generator_behavior():
    overlay = (CONFIGS / "gear_vineppo_overlay.libsonnet").read_text()

    assert "type: 'gear_vineppo_episode_generator'" in overlay
    assert "type: 'gear'" in overlay
    assert "use_prob_mask: false" in overlay

    for model in ("qwen1_5b_base", "deepseekR1Qwen"):
        config = (CONFIGS / f"polIter_{model}_gear_vineppo_MATH.jsonnet").read_text()

        assert "import 'gear_defaults.libsonnet'" in config
        assert "import 'gear_vineppo_overlay.libsonnet'" in config
        assert "tree_policy_algorithm_name: 'gear_vineppo'" in config
        assert "tree_policy_segmentation_type: 'vineppo_step'" in config
        assert "skip_near_leaf_expand: false" in config


def test_gear_variant_launchers_default_to_chain_compatible_tree():
    for name, suffix in (
        ("train_gear_spo_chain_MATH.sh", "gear_spo_chain"),
        ("train_gear_vineppo_MATH.sh", "gear_vineppo"),
    ):
        script = (SCRIPTS / name).read_text()

        assert 'MODEL="${MODEL:-deepseekR1Qwen}"' in script
        assert 'TREE="${TREE:-${GEAR_TREE:-6}}"' in script
        assert f'resolve_math_config {suffix} "${{MODEL}}"' in script
        assert 'ensure_tree_config "${TREE}"' in script
    vineppo = (SCRIPTS / "train_gear_vineppo_MATH.sh").read_text()
    assert '[[ "${TREE}" =~ ^[1-9]$ ]]' in vineppo

GEAR_VARIANT_CONFIGS = [
    "polIter_qwen1_5b_base_gear_tree_MATH.jsonnet",
    "polIter_deepseekR1Qwen_gear_tree_MATH.jsonnet",
    "polIter_qwen1_5b_base_gear_spo_chain_MATH.jsonnet",
    "polIter_deepseekR1Qwen_gear_spo_chain_MATH.jsonnet",
    "polIter_qwen1_5b_base_gear_vineppo_MATH.jsonnet",
    "polIter_deepseekR1Qwen_gear_vineppo_MATH.jsonnet",
]


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


def test_all_gear_variants_enable_pruning_prediction_and_allocation():
    for filename in GEAR_VARIANT_CONFIGS:
        config = _render_config(filename)
        inference = config["episode_generator"]["inference_strategy"]

        assert inference["type"] == "gear", filename
        assert inference["gear_k_algorithm"] == "hierarchical", filename
        assert inference["gear_allocation_mode"] == "budget_allocation", filename
        assert inference["gear_use_residual_budget"] is True, filename
        assert inference["gear_root_allocation"] is True, filename
        assert inference["node_expander"]["program_kwargs"]["logprobs"] == 1, filename
