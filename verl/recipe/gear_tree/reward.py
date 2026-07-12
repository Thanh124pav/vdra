"""Math reward manager for the GEAR/Tree recipe."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from recipe.gear_tree.gear_core.grading.math_grader import grade_answer


def compute_gear_math_score(
    *,
    data_source: str | None = None,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a math response with the vendored treetune grader.

    Tree rollout can pass an already extracted answer in ``extra_info`` under
    ``predicted_answer`` or ``answer``. Without one, this falls back to grading
    the decoded response string directly.
    """
    _ = data_source
    extra_info = extra_info or {}
    given_answer = extra_info.get("predicted_answer", extra_info.get("answer", solution_str))
    correct = bool(grade_answer(given_answer=given_answer, ground_truth=ground_truth))
    return {"score": float(correct), "correct": correct, "given_answer": given_answer}


@register("gear_math")
class GearMathRewardManager(AbstractRewardManager):
    """verl reward manager that mirrors treetune math grading semantics."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        _ = kwargs
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or compute_gear_math_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": {key: data.non_tensor_batch[key] for key in reward_extra_keys},
                }
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        printed_by_source: dict[str, int] = {}

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(item.batch["attention_mask"][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = item.batch["responses"]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_model = item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth", item.non_tensor_batch.get("ground_truth"))
            data_source = item.non_tensor_batch.get(self.reward_fn_key, "gear_math")
            extra_info = dict(item.non_tensor_batch.get("extra_info", {}))

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward = score["score"] if isinstance(score, dict) else score
            if isinstance(score, dict):
                for key, value in score.items():
                    reward_extra_info[key].append(value)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = float(reward)

            printed_by_source.setdefault(data_source, 0)
            if printed_by_source[data_source] < self.num_examine:
                printed_by_source[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", reward)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor