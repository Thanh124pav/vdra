"""Tree policy update objectives shared by SPO/GEAR-style generators.

PLAN.md P0.N3: ``treepo_original`` and ``treerl_original`` scalarise a single
per-edge advantage and are NOT byte-faithful reproductions of the official
TreePO or TreeRL algorithms (those use different credit assignment and
aggregation). They are retained here only as scalar-objective ablations. The
canonical names are ``treepo_style_ablation`` and ``treerl_style_ablation``;
the ``_original`` names are kept as deprecated aliases only so vendor parity
tests against upstream ``treetune`` (which still uses those strings) keep
passing. Do not use the ``_original`` names in main VDRA configs.
"""

from __future__ import annotations

from typing import Any, Dict


# Canonical ablation names.
_TREEPO_ABLATION = "treepo_style_ablation"
_TREERL_ABLATION = "treerl_style_ablation"

# Deprecated aliases kept for treetune vendor parity only.
_LEGACY_ALIASES = {
    "treepo_original": _TREEPO_ABLATION,
    "treerl_original": _TREERL_ABLATION,
}


SUPPORTED_TREE_UPDATE_MODES = {
    "spo",
    _TREEPO_ABLATION,
    _TREERL_ABLATION,
    # Deprecated aliases — retained for vendor parity, not for main runs.
    "treepo_original",
    "treerl_original",
}


def _canonicalise(mode: str) -> str:
    return _LEGACY_ALIASES.get(mode, mode)


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

    ``spo`` preserves the current local parent-child advantage — the main VDRA
    setting until another estimator is implemented faithfully.

    ``treepo_style_ablation`` (alias: ``treepo_original``) scalarises TreePO's
    local and global segment objectives into the single advantage tensor
    consumed by this PPO trainer. This is a scalar-objective ablation, not a
    faithful TreePO reproduction.

    ``treerl_style_ablation`` (alias: ``treerl_original``) uses a TD-style
    dense process target for each tree edge. This is a scalar-objective
    ablation, not a faithful TreeRL reproduction.
    """

    mode = validate_tree_update_mode(mode)
    canonical = _canonicalise(mode)
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

    if canonical == "spo":
        advantage = local_advantage
        value = child_reward
    elif canonical == _TREEPO_ABLATION:
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
