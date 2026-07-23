"""Faithful port of treetune's ``MATHRewardFunction`` grading path.

Mirrors ``MATHRewardFunction.__call__`` and ``MATH.grade_answer`` /
``MATH.extract_predicted_answer_from_text`` from
``treetune/episode_generators/math_episode_generator.py`` and
``treetune/tasks/math.py`` for grading, with a VDRA-facing parser-mode
resolver layered on top so configs can select ``boxed`` / ``answer`` / auto
detection. The math grading itself delegates to the vendored grading modules.

Used by the tree rollout to grade leaf segments *during* tree construction, so
the segment mean-reward (and therefore the SPO/GEAR advantage) is byte-identical
to treetune.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from .grading.math_answer_exctraction import (
    extract_boxed_answers,
    extract_math_answer,
    extract_math_minerva_few_shot_cot_answer,
    strip_string,
)
from .grading.math_grader import grade_answer
from .grading.math_grader_minerva import eval_math


ANSWER_PREFIX = "# Answer\n"


def normalize_answer_parse_mode(answer_prefix: Optional[str]) -> tuple[str, Optional[str]]:
    """Resolve the user-facing answer_prefix knob to a strict parse mode.

    ``None`` means auto-detect from the prompt text. ``boxed`` means require a
    ``\\boxed{...}`` final answer. ``answer`` means require the treetune
    ``# Answer\n`` delimiter. The historical literal ``# Answer\n`` remains
    accepted as answer-prefix mode for backward compatibility.
    """
    if answer_prefix is None:
        return "auto", None
    raw = str(answer_prefix)
    key = raw.strip().lower()
    if key in {"boxed", "box", "\\boxed", "\\boxed{}"}:
        return "boxed", None
    if key in {"answer", "answer-prefix", "answer_prefix", "# answer"} or raw == ANSWER_PREFIX:
        return "answer", ANSWER_PREFIX
    return "answer", raw


def detect_answer_parse_mode_from_prompt(prompt: Optional[str]) -> str:
    text = str(prompt or "")
    lowered = text.lower()
    if "# answer" in lowered:
        return "answer"
    if "\\boxed" in text or "boxed{}" in lowered or "boxed" in lowered:
        return "boxed"
    return "boxed"


def extract_boxed_answer_from_text(text: str) -> Optional[str]:
    answers = extract_boxed_answers(text)
    if not answers:
        return None
    answer = strip_string(answers[-1])
    return answer if answer != "" else None


def extract_answer_prefix_answer_from_text(text: str, delimiter: str = ANSWER_PREFIX) -> Optional[str]:
    splits = text.split(delimiter)
    if len(splits) != 2:
        return None
    answer = splits[1].strip()
    return answer if answer != "" else None


def extract_predicted_answer(
    *,
    text: str,
    problem: Optional[str],
    prompt: Optional[str],
    answer_prefix: Optional[str],
    use_minerva_few_shot_prompt: bool = False,
) -> tuple[Optional[str], str]:
    configured_mode, delimiter = normalize_answer_parse_mode(answer_prefix)
    mode = detect_answer_parse_mode_from_prompt(prompt) if configured_mode == "auto" else configured_mode
    if use_minerva_few_shot_prompt:
        return extract_math_minerva_few_shot_cot_answer(problem, text), mode
    if mode == "boxed":
        return extract_boxed_answer_from_text(text), mode
    return extract_answer_prefix_answer_from_text(text, delimiter or ANSWER_PREFIX), mode


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
        self.reset_parse_metrics()

    def get_unfinished_response_penalty(self) -> float:
        return float(self.unfinished_response_penalty)

    def reset_parse_metrics(self) -> None:
        self.answer_parse_attempts = 0
        self.answer_parse_failures = 0
        self.answer_parse_mode_counts = {"boxed": 0, "answer": 0}
        self._last_answer_parse_mode = None

    def parse_metrics(self) -> Dict[str, float]:
        attempts = int(self.answer_parse_attempts)
        failures = int(self.answer_parse_failures)
        return {
            "reward/answer_parse_attempts": float(attempts),
            "reward/answer_parse_failures": float(failures),
            "reward/answer_parse_failure_rate": float(failures / attempts) if attempts else 0.0,
            "reward/answer_parse_mode_boxed": float(self.answer_parse_mode_counts.get("boxed", 0)),
            "reward/answer_parse_mode_answer": float(self.answer_parse_mode_counts.get("answer", 0)),
        }

    # --- MATH.extract_predicted_answer_from_text (math.py:364-392) -----------
    def extract_predicted_answer_from_text(
        self, text: str, problem: Optional[str] = None, prompt: Optional[str] = None
    ) -> Optional[str]:
        pred_answer, mode = extract_predicted_answer(
            text=text,
            problem=problem,
            prompt=prompt,
            answer_prefix=self.answer_prefix,
            use_minerva_few_shot_prompt=self.use_minerva_few_shot_prompt,
        )
        self._last_answer_parse_mode = mode
        self.answer_parse_attempts += 1
        self.answer_parse_mode_counts[mode] = self.answer_parse_mode_counts.get(mode, 0) + 1
        if pred_answer is None:
            self.answer_parse_failures += 1
        return pred_answer

    # --- MATH.grade_answer (math.py:494-544) --------------------------------
    def _grade_answer(
        self,
        *,
        given_answer: Optional[str],
        ground_truth: Optional[str],
        item: Optional[Dict[str, Any]],
    ) -> bool:
        configured_mode, _ = normalize_answer_parse_mode(self.answer_prefix)
        if self.use_minerva_few_shot_prompt or configured_mode == "boxed" or self._last_answer_parse_mode == "boxed":
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
            response, dataset_instance["problem"], query
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
