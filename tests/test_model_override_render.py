from treetune.common.py_utils import load_config_object


BASE_GEAR_TREE_MATH_CONFIG = "configs/polIter_qwen1_5b_base_gear_tree_MATH.jsonnet"


def test_requested_model_overrides_render_to_expected_hf_models(monkeypatch):
    monkeypatch.setenv('APP_SEED', '123')
    monkeypatch.setenv('APP_DISABLE_FLASH_ATTENTION', '0')

    expected_models = {
        "qwen3_0_6b": "Qwen/Qwen3-0.6B",
        "qwen3_0_6b_base": "Qwen/Qwen3-0.6B-Base",
        "qwen3_1_7b": "Qwen/Qwen3-1.7B",
        "qwen3_1_7b_base": "Qwen/Qwen3-1.7B-Base",
        "qwen3_4b": "Qwen/Qwen3-4B",
        "qwen3_4b_base": "Qwen/Qwen3-4B-Base",
        "deepseekR1Qwen14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    }

    for alias, expected_hf_model in expected_models.items():
        config = load_config_object(
            [
                BASE_GEAR_TREE_MATH_CONFIG,
                f"configs/model_overrides/{alias}.jsonnet",
            ]
        )

        assert config["episode_generator"]["initial_model_name_or_path"] == expected_hf_model
        assert config["tokenizer"]["hf_model_name"] == expected_hf_model
        assert config["trainer"]["actor_model"]["hf_model_name"] == expected_hf_model
        assert config["trainer"]["reference_model"]["hf_model_name"] == expected_hf_model
        assert (
            config["episode_generator"]["inference_strategy"]["guidance_llm"]["model"]
            == expected_hf_model
        )
        assert (
            config["episode_generator"]["value_estimation_inference_strategy"][
                "guidance_llm"
            ]["model"]
            == expected_hf_model
        )
