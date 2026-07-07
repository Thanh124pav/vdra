from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"


def test_requested_model_overrides_are_selectable_by_model_env():
    expected_models = {
        "qwen3_0_6b": "Qwen/Qwen3-0.6B",
        "qwen3_0_6b_base": "Qwen/Qwen3-0.6B-Base",
        "qwen3_1_7b": "Qwen/Qwen3-1.7B",
        "qwen3_1_7b_base": "Qwen/Qwen3-1.7B-Base",
        "qwen3_4b": "Qwen/Qwen3-4B",
        "qwen3_4b_base": "Qwen/Qwen3-4B-Base",
        "deepseekR1Qwen14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    }

    common_script = (SCRIPTS / "_common.sh").read_text()
    assert "configs/model_overrides/${model}.jsonnet" in common_script

    for alias, hf_model in expected_models.items():
        override = CONFIGS / "model_overrides" / f"{alias}.jsonnet"
        assert override.is_file()
        text = override.read_text()
        assert hf_model in text
        assert "initial_model_name_or_path: hf_model_name" in text
        assert "actor_model+: { hf_model_name: hf_model_name }" in text
        assert "reference_model+: { hf_model_name: hf_model_name }" in text
