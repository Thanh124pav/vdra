import torch

from verl import DataProto
from verl.workers.reward_manager import get_reward_manager_cls
from recipe.gear_tree import reward  # noqa: F401 - registers gear_math
from recipe.gear_tree.reward import compute_gear_math_score


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