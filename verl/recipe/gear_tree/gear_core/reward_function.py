"""Faithful port of treetune's ``MATHRewardFunction`` grading path.

Mirrors ``MATHRewardFunction.__call__`` and ``MATH.grade_answer`` /
``MATH.extract_predicted_answer_from_text`` from
``treetune/episode_generators/math_episode_generator.py`` and
``treetune/tasks/math.py`` **exactly** (same extraction, same unfinished/
multi-#### penalties, same grader selection). Only the class/registry glue is
dropped; the math is unchanged and delegates to the vendored grading modules.

Used by the tree rollout to grade leaf segments *during* tree construction, so
the segment mean-reward (and therefore the SPO/GEAR advantage) is byte-identical
to treetune.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from .grading.math_answer_exctraction import (
    extract_math_answer,
    extract_math_minerva_few_shot_cot_answer,
)
from .grading.math_grader import grade_answer
from .grading.math_grader_minerva import eval_math


class MathRewardFunction:
    """Treetune-faithful MATH reward.

    Parameters mirror the ``MATH`` task + ``MATHRewardFunction`` config knobs
    that affect grading. Defaults match the common MATH tree configs
    (``answer_prefix='# Answer\\n'`` extraction, dataset answer used for the
    minerva grader).
    """

    def __init__(
        self,
        *,
        answer_prefix: Optional[str] = "# Answer\n",
        use_minerva_few_shot_prompt: bool = False,
        use_dataset_answer: bool = True,
        penalize_unfinished_response: bool = True,
        unfinished_response_penalty: float = 0.0,
    ) -> None:
        self.answer_prefix = answer_prefix
        self.use_minerva_few_shot_prompt = use_minerva_few_shot_prompt
        self.use_dataset_answer = use_dataset_answer
        self.penalize_unfinished_response = penalize_unfinished_response
        self.unfinished_response_penalty = unfinished_response_penalty

    def get_unfinished_response_penalty(self) -> float:
        return float(self.unfinished_response_penalty)

    # --- MATH.extract_predicted_answer_from_text (math.py:364-392) -----------
    def extract_predicted_answer_from_text(
        self, text: str, problem: Optional[str] = None
    ) -> Optional[str]:
        if self.use_minerva_few_shot_prompt or self.answer_prefix is None:
            return extract_math_minerva_few_shot_cot_answer(problem, text)

        splits = text.split("# Answer\n")
        if len(splits) != 2:
            return None
        return splits[1].strip()

    # --- MATH.grade_answer (math.py:494-544) --------------------------------
    def _grade_answer(
        self,
        *,
        given_answer: Optional[str],
        ground_truth: Optional[str],
        item: Optional[Dict[str, Any]],
    ) -> bool:
        if self.use_minerva_few_shot_prompt or self.answer_prefix is None:
            item = copy.deepcopy(item)
            if self.use_dataset_answer:
                answer = item["answer"]
            else:
                answer = extract_math_answer(item["problem"], item["solution"])
            item["answer"] = answer
            item["prediction"] = given_answer
            return eval_math(item)
        return grade_answer(given_answer=given_answer, ground_truth=ground_truth)

    # --- MATHRewardFunction.__call__ (math_episode_generator.py:45-69) ------
    def __call__(
        self, query: str, response: str, dataset_instance: Dict[str, Any]
    ) -> Tuple[float, bool]:
        pred_answer = self.extract_predicted_answer_from_text(
            response, dataset_instance["problem"]
        )
        is_unfinished_response = pred_answer is None

        # GSK8K multiple-#### guard (math_episode_generator.py:54-56).
        parts = response.split("####")
        if len(parts) > 2:
            return -2.0, False

        if is_unfinished_response and self.penalize_unfinished_response:
            return float(self.unfinished_response_penalty), is_unfinished_response

        gold_answer = dataset_instance["answer"]
        reward = self._grade_answer(
            given_answer=pred_answer,
            ground_truth=gold_answer,
            item=dataset_instance,
        )
        return float(reward), is_unfinished_response
