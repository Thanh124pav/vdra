import pytest

from treetune.common.py_utils import apply_max_model_len_override


def _representative_config():
    return {
        "episode_generator": {
            "max_sequence_length": None,
            "vllm_server": {"max_num_seqs": 128},
            "inference_strategy": {
                "node_expander": {"model_context_size": 1024},
            },
            "value_estimation_inference_strategy": {
                "node_expander": {"model_context_size": 2047},
            },
        },
        "trainer": {
            "general_training_args": {"max_seq_len": 2048},
        },
        "evaluation_vllm_server": {},
        "inference_pipelines": [
            {
                "inference_strategy": {
                    "node_expander": {"model_context_size": 4095},
                },
            },
        ],
        "analyzers": [
            {
                "vllm_server": {"swap_space": 8},
                "inference_strategy": {
                    "node_expander": {"model_context_size": 1024},
                },
            },
        ],
    }


def test_unset_max_model_len_preserves_model_specific_defaults():
    config = _representative_config()

    apply_max_model_len_override(config, None)

    assert config["episode_generator"]["max_sequence_length"] is None
    assert config["episode_generator"]["inference_strategy"]["node_expander"][
        "model_context_size"
    ] == 1024
    assert "max_model_len" not in config["evaluation_vllm_server"]


def test_max_model_len_overrides_all_models_algorithms_and_vllm_servers():
    config = _representative_config()

    apply_max_model_len_override(config, "8192")

    episode_generator = config["episode_generator"]
    assert episode_generator["max_sequence_length"] == 8192
    assert episode_generator["vllm_server"]["max_model_len"] == 8192
    assert episode_generator["inference_strategy"]["node_expander"][
        "model_context_size"
    ] == 8192
    assert episode_generator["value_estimation_inference_strategy"]["node_expander"][
        "model_context_size"
    ] == 8192
    assert config["trainer"]["general_training_args"]["max_seq_len"] == 8192
    assert config["evaluation_vllm_server"]["max_model_len"] == 8192
    assert config["inference_pipelines"][0]["inference_strategy"]["node_expander"][
        "model_context_size"
    ] == 8192
    assert config["analyzers"][0]["vllm_server"]["max_model_len"] == 8192
    assert config["analyzers"][0]["inference_strategy"]["node_expander"][
        "model_context_size"
    ] == 8192


def test_max_model_len_adds_missing_trainer_limit():
    config = _representative_config()
    del config["trainer"]["general_training_args"]["max_seq_len"]

    apply_max_model_len_override(config, "4096")

    assert config["trainer"]["general_training_args"]["max_seq_len"] == 4096


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5"])
def test_max_model_len_rejects_invalid_values(value):
    with pytest.raises(
        ValueError, match="APP_MAX_MODEL_LEN must be a positive integer"
    ):
        apply_max_model_len_override(_representative_config(), value)
