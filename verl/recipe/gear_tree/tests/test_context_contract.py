
import pytest

from recipe.gear_tree.context_contract import (
    normalize_tree_shape,
    validate_context_contract,
    worst_case_edge_prompt_length,
)


def test_tree_shape_hydra_string_uses_list_depth():
    assert normalize_tree_shape("[6,6,6]") == [6, 6, 6]
    assert normalize_tree_shape("'[6,6,6]'") == [6, 6, 6]
    assert worst_case_edge_prompt_length(
        max_original=512, tree_shape="[6,6,6]", segment_length=600
    ) == 1712


def test_context_contract_reports_depth_from_parsed_tree_shape():
    with pytest.raises(ValueError) as excinfo:
        validate_context_contract(
            data_cfg={
                "max_prompt_length": 512,
                "max_edge_prompt_length": 512,
                "max_response_length": 2048,
            },
            tree_shape="[6,6,6]",
            segment_length=600,
            model_context_length=4096,
        )

    message = str(excinfo.value)
    assert "depth=3" in message
    assert "query length=1712" in message
    assert "depth=7" not in message


def test_context_contract_accepts_edge_prompt_headroom_for_hydra_string_shape():
    validate_context_contract(
        data_cfg={
            "max_prompt_length": 512,
            "max_original_prompt_length": 512,
            "max_edge_prompt_length": 1712,
            "max_response_length": 2048,
        },
        tree_shape="[6,6,6]",
        segment_length=600,
        model_context_length=3760,
    )


def test_context_contract_allows_dynamic_response_remaining_context():
    validate_context_contract(
        data_cfg={
            "max_prompt_length": 512,
            "max_original_prompt_length": 512,
            "max_edge_prompt_length": 1456,
            "max_response_length": 2048,
        },
        tree_shape="[6,6,6]",
        segment_length=472,
        model_context_length=2560,
    )


def test_context_contract_rejects_edge_prompt_over_model_context():
    with pytest.raises(ValueError) as excinfo:
        validate_context_contract(
            data_cfg={
                "max_prompt_length": 512,
                "max_original_prompt_length": 512,
                "max_edge_prompt_length": 1456,
                "max_response_length": 2048,
            },
            tree_shape="[6,6,6]",
            segment_length=472,
            model_context_length=1024,
        )

    message = str(excinfo.value)
    assert "max_edge_prompt_length=1456 exceeds resolved model context length 1024" in message
    assert "max_response_length=2048" not in message
