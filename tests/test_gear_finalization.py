from treetune.inference_strategies.gear_inference_strategy import (
    _finalize_empty_expansion_node,
)


def test_finalize_empty_expansion_scores_current_node_as_leaf():
    calls = []

    def reward_function(*, query, response, dataset_instance):
        calls.append((query, response, dataset_instance))
        return 0.75, {"source": "test"}

    node = {
        "text": " partial answer",
        "full_text": "question partial answer",
        "leaf": False,
    }
    data_instance = {"answer": "expected"}

    _finalize_empty_expansion_node(
        node,
        initial_prompt="question",
        data_instance=data_instance,
        reward_function=reward_function,
    )

    assert calls == [("question", "question partial answer", data_instance)]
    assert node["children"] == []
    assert node["reward"] == 0.75
    assert node["reward_std"] == 0.0
    assert node["leaf"] is True
