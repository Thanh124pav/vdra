"""Tree policy update objectives shared by SPO/GEAR-style generators."""

from __future__ import annotations

from typing import Any, Dict


SUPPORTED_TREE_UPDATE_MODES = {
    "spo",
    "treepo_original",
    "treerl_original",
}


def validate_tree_update_mode(mode: str) -> str:
    if mode not in SUPPORTED_TREE_UPDATE_MODES:
        supported = ", ".join(sorted(SUPPORTED_TREE_UPDATE_MODES))
        raise ValueError(f"Unknown tree_update_mode={mode!r}; supported: {supported}")
    return mode


def compute_tree_update_values(
    *,
    child_reward: float,
    parent_reward: float,
    root_reward: float,
    parent_reward_std: float = 0.0,
    adv_method: str = "rloo",
    mode: str = "spo",
    treepo_global_weight: float = 0.5,
    treerl_gamma: float = 0.9,
) -> Dict[str, Any]:
    """Return edge-level advantage/value fields for tree policy training.

    `spo` preserves the current local parent-child advantage.
    `treepo_original` scalarizes TreePO's local and global segment objectives
    into the single advantage tensor consumed by this PPO trainer.
    `treerl_original` uses a TD-style dense process target for each tree edge.
    """

    mode = validate_tree_update_mode(mode)
    parent_reward_std = float(parent_reward_std or 0.0)
    child_reward = float(child_reward)
    parent_reward = float(parent_reward)
    root_reward = float(root_reward)

    if adv_method == "rloo":
        local_advantage = child_reward - parent_reward
    elif adv_method == "grpo":
        local_advantage = (child_reward - parent_reward) / (parent_reward_std + 1e-8)
    else:
        raise ValueError(f"adv_method {adv_method} is not supported")

    global_advantage = child_reward - root_reward

    if mode == "spo":
        advantage = local_advantage
        value = child_reward
    elif mode == "treepo_original":
        global_weight = min(max(float(treepo_global_weight), 0.0), 1.0)
        advantage = (
            (1.0 - global_weight) * local_advantage
            + global_weight * global_advantage
        )
        value = child_reward
    else:
        immediate_process_reward = child_reward - parent_reward
        value = immediate_process_reward + float(treerl_gamma) * child_reward
        advantage = value - parent_reward

    return {
        "advantage": float(advantage),
        "value": float(value),
        "tree_update_mode": mode,
        "tree_update_local_advantage": float(local_advantage),
        "tree_update_global_advantage": float(global_advantage),
        "tree_update_root_reward": float(root_reward),
        "tree_update_parent_reward": float(parent_reward),
        "tree_update_child_reward": float(child_reward),
    }
