"""VinePPO Monte-Carlo value + step advantages (for GEAR-VinePPO).

Ports ``VinePPOEpisodeGenerator._compute_mc_value`` and
``_compute_step_advantages`` (vineppo_episode_generator.py:377-432) **exactly**:

  * ``mc_value`` = mean reward of ``K`` independent rollouts from a cut-point,
    with the unfinished-response penalty for ``finish_reason == "length"``.
  * step advantages via the TD residual
    ``A[i] = step_rewards[i] + values[i+1] - values[i]``, with ``values[-1] = 0``
    and missing values back-filled by ``values[i] = step_rewards[i] + values[i+1]``.

The MC rollouts are supplied by a ``rollout_fn(prefix_token_ids, K) -> samples``
callable so this is CPU-testable with a mock; in production it wraps the same
vLLM engine used by the tree rollout.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from recipe.gear_tree.tree_rollout import SegmentSample


# rollout_fn(prefix_token_ids, K) -> list[SegmentSample] (K completions).
RolloutFn = Callable[[Sequence[int], int], List[SegmentSample]]
GradeFn = Callable[[str, str, Dict[str, Any]], float]


def mc_value(
    prefix_text: str,
    prefix_token_ids: Sequence[int],
    data_instance: Dict[str, Any],
    *,
    rollout_fn: RolloutFn,
    grade_fn: GradeFn,
    K: int,
    unfinished_penalty: float = 0.0,
    return_rewards: bool = False,
):
    """MC value estimate = mean reward over K rollouts (VinePPO ``_compute_mc_value``)."""
    samples = rollout_fn(prefix_token_ids, K)
    rewards: List[float] = []
    for s in samples:
        if s.finish_reason != "length":
            rewards.append(float(grade_fn(prefix_text, prefix_text + s.text, data_instance)))
        else:
            rewards.append(float(unfinished_penalty))
    value = sum(rewards) / len(rewards) if rewards else 0.0
    if return_rewards:
        return value, rewards
    return value


def step_advantages(step_rewards: List[float], values: List[Optional[float]]) -> List[float]:
    """VinePPO ``_compute_step_advantages`` (byte-faithful).

    ``values`` has one more entry than ``step_rewards``; the terminal value
    (``values[-1]``) must be ``None`` and is set to 0. Missing (``None``) values
    are back-filled, then ``A[i] = step_rewards[i] + values[i+1] - values[i]``.
    """
    values = list(values)
    assert values[-1] is None
    values[-1] = 0.0
    for i in range(len(values) - 2, -1, -1):
        if values[i] is not None:
            break
        values[i] = step_rewards[i] + values[i + 1]
    assert all(v is not None for v in values)

    advantages: List[float] = [0.0] * len(step_rewards)
    assert len(advantages) == len(values) - 1
    for i in range(len(advantages)):
        advantages[i] = step_rewards[i] + values[i + 1] - values[i]
    return advantages


def annotate_tree_with_mc_values(
    tree: Dict[str, Any],
    data_instance: Dict[str, Any],
    *,
    rollout_fn: RolloutFn,
    grade_fn: GradeFn,
    K: int,
    unfinished_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Replace each internal node's reward with a VinePPO MC value estimate.

    After this, ``extract_edges_from_tree`` with ``adv_method='rloo'`` yields the
    VinePPO TD advantage ``value(child) - value(parent)`` (step reward = 0 for
    intermediate segments), matching treetune's step advantage on the tree.
    Leaf rewards are left as the graded terminal reward.
    """

    def visit(node: Dict[str, Any]) -> None:
        children = node.get("children") or []
        for child in children:
            visit(child)
        if children:  # internal node: MC value from K rollouts of its trajectory
            node["reward"] = mc_value(
                node.get("full_text", ""),
                node.get("full_token_ids", []),
                data_instance,
                rollout_fn=rollout_fn,
                grade_fn=grade_fn,
                K=K,
                unfinished_penalty=unfinished_penalty,
            )

    visit(tree)
    return tree
