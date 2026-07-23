import torch

from verl import DataProto
from verl.workers.reward_manager import get_reward_manager_cls
from recipe.gear_tree import reward  # noqa: F401 - registers gear_math
from recipe.gear_tree.reward import compute_gear_math_score
from recipe.gear_tree.gear_core.reward_function import MathRewardFunction


class TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        _ = skip_special_tokens
        return " ".join(str(int(x)) for x in ids)


def test_compute_gear_math_score_uses_predicted_answer_metadata():
    score = compute_gear_math_score(
        solution_str="ignored decoded response",
        ground_truth="2",
        extra_info={"predicted_answer": "2"},
    )
    assert score["score"] == 1.0
    assert score["correct"] is True


def test_gear_math_reward_manager_registers_and_scores_last_token():
    cls = get_reward_manager_cls("gear_math")
    manager = cls(TinyTokenizer(), num_examine=0)
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[101, 102]]),
            "responses": torch.tensor([[11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        },
        non_tensors={
            "data_source": ["math"],
            "reward_model": [{"ground_truth": "2"}],
            "extra_info": [{"predicted_answer": "2"}],
        },
    )

    reward_tensor = manager(data)
    assert torch.equal(reward_tensor, torch.tensor([[0.0, 1.0, 0.0]]))


class MappingTokenizer:
    def __init__(self, mapping):
        self.mapping = {tuple(k): v for k, v in mapping.items()}

    def decode(self, ids, skip_special_tokens=True):
        _ = skip_special_tokens
        key = tuple(int(x) for x in ids)
        return self.mapping.get(key, " ".join(str(x) for x in key))


def _math_instance(answer="2"):
    return {"problem": "What is 1+1?", "answer": answer, "solution": f"We get \\boxed{{{answer}}}."}


def test_math_reward_auto_detects_boxed_prompt_and_counts_parse_success():
    reward_fn = MathRewardFunction(answer_prefix=None)
    reward, unfinished = reward_fn(
        "What is 1+1? Output the final answer within \\boxed{}.",
        "Reasoning. Therefore \\boxed{2}.",
        _math_instance(),
    )

    assert reward == 1.0
    assert unfinished is False
    assert reward_fn.parse_metrics()["reward/answer_parse_failures"] == 0.0
    assert reward_fn.parse_metrics()["reward/answer_parse_mode_boxed"] == 1.0


def test_math_reward_explicit_answer_prefix_mode():
    reward_fn = MathRewardFunction(answer_prefix="answer")
    reward, unfinished = reward_fn(
        "What is 1+1? Put final answer after # Answer.",
        "Reasoning.\n# Answer\n2",
        _math_instance(),
    )

    assert reward == 1.0
    assert unfinished is False
    assert reward_fn.parse_metrics()["reward/answer_parse_mode_answer"] == 1.0


def test_math_reward_explicit_boxed_counts_parse_failure():
    reward_fn = MathRewardFunction(answer_prefix="boxed")
    reward, unfinished = reward_fn(
        "What is 1+1? Output the final answer within \\boxed{}.",
        "Reasoning. The answer is 2.",
        _math_instance(),
    )

    assert reward == 0.0
    assert unfinished is True
    assert reward_fn.parse_metrics()["reward/answer_parse_failures"] == 1.0


def test_gear_math_reward_manager_logs_parse_failure_metrics():
    cls = get_reward_manager_cls("gear_math")
    manager = cls(
        MappingTokenizer(
            {
                (1,): "What is 1+1? Output the final answer within \\boxed{}.",
                (2,): "Reasoning. Therefore \\boxed{2}.",
                (3,): "Reasoning. The answer is 2.",
            }
        ),
        num_examine=0,
        answer_prefix=None,
    )
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1], [1]]),
            "responses": torch.tensor([[2], [3]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        },
        non_tensors={
            "data_source": ["math", "math"],
            "reward_model": [{"ground_truth": "2"}, {"ground_truth": "2"}],
            "extra_info": [{"problem": "What is 1+1?"}, {"problem": "What is 1+1?"}],
        },
    )

    result = manager(data, return_dict=True)

    assert result["reward_extra_info"]["answer_parse_failed"] == [0.0, 1.0]
    assert result["reward_extra_info"]["answer_parse_mode_boxed"] == [1.0, 1.0]
    assert torch.equal(result["reward_tensor"], torch.tensor([[1.0], [0.0]]))
