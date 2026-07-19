"""PLAN.md M5: canonical cross-level config validation.

Extracted verbatim from ``RayGearTreeTrainer._validate_replay_startup`` so the
pre-GPU Hydra-composition gate can run the exact validation the trainer runs,
instead of re-implementing it. This module stays engine-free (no torch / verl
/ ray imports) so it is importable from a bare script.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _to_plain_gear_tree_cfg(config: Any) -> Mapping[str, Any]:
    """Mirror ``RayGearTreeTrainer._gear_tree_config``: the ``gear_tree``
    block as a plain dict (resolved when it is an OmegaConf node)."""
    raw = config.get("gear_tree") or {}
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(raw, DictConfig):
            return OmegaConf.to_container(raw, resolve=True)
    except ImportError:  # pragma: no cover - omegaconf always present in prod
        pass
    return dict(raw)


def validate_policy_loss_consistency(
    config: Any,
    *,
    gear_tree_cfg: Optional[Mapping[str, Any]] = None,
) -> None:
    """PLAN.md P0.1 / P1.R7: tree_policy <-> actor.policy_loss consistency.

    Raises ``ValueError`` on any violation; returns ``None`` when the config
    is consistent. ``gear_tree_cfg`` may be passed pre-converted (the trainer
    does); otherwise it is derived from ``config.gear_tree``.
    """
    gt = gear_tree_cfg if gear_tree_cfg is not None else _to_plain_gear_tree_cfg(config)
    # PLAN.md P1.R7: refuse the deprecated ablation `_original` names in
    # strict main runs, and refuse to combine the *_style_ablation modes
    # with the canonical policy aggregation (they are ablations, not
    # main-paper losses). The gate's own strict checks cover
    # pilot_execution_mode and allocation_runtime.
    gear_cfg = gt.get("gear") or {}
    strict = bool(gear_cfg.get("strict_vdra", True))
    tree_update_mode = str(gt.get("tree_update_mode", "spo"))
    tree_policy = config.get("tree_policy") or {}
    policy_agg = str(tree_policy.get("policy_aggregation", "legacy_token_mean"))
    actor_loss_mode = str(
        config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
    )
    segment_reduction = str(
        tree_policy.get("segment_token_reduction", "mean")
    ).strip().lower()
    # PLAN.md P0.1: segment_token_reduction must be exactly `mean` or `sum`.
    if segment_reduction not in ("mean", "sum"):
        raise ValueError(
            "tree_policy.segment_token_reduction must be exactly 'mean' or "
            f"'sum' (PLAN.md P0.1); got {segment_reduction!r}."
        )
    # PLAN.md P0.1: `tree_policy.segment_token_reduction` and
    # `actor.policy_loss.segment_token_reduction` are duplicates that MUST
    # agree, otherwise the actor loss silently reads a different reduction
    # than the manifest/logs advertise.
    actor_reduction = str(
        config.actor_rollout_ref.actor.policy_loss.get(
            "segment_token_reduction", "mean"
        )
    ).strip().lower()
    if actor_reduction not in ("mean", "sum"):
        raise ValueError(
            "actor_rollout_ref.actor.policy_loss.segment_token_reduction "
            f"must be exactly 'mean' or 'sum' (PLAN.md P0.1); got "
            f"{actor_reduction!r}."
        )
    if actor_reduction != segment_reduction:
        raise ValueError(
            "tree_policy.segment_token_reduction "
            f"({segment_reduction!r}) must equal "
            "actor_rollout_ref.actor.policy_loss.segment_token_reduction "
            f"({actor_reduction!r}) (PLAN.md P0.1)."
        )
    if strict:
        if tree_update_mode in {"treepo_original", "treerl_original"}:
            raise ValueError(
                "strict VDRA main runs must not use the deprecated "
                "tree_update_mode aliases (PLAN.md P1.R7); rename to "
                "'*_style_ablation' or set strict_vdra=false."
            )
        if (
            policy_agg == "vdra_node_balanced"
            and tree_update_mode
            in {"treepo_style_ablation", "treerl_style_ablation"}
        ):
            raise ValueError(
                "The style-ablation tree_update_modes are not main-paper "
                "advantage estimators (PLAN.md P1.R7)."
            )
        # PLAN.md P0.1: the canonical main policy_aggregation is
        # global_segment_mean, which requires vdra_segment_mean_ppo. The
        # legacy vdra_node_balanced_ppo must not appear in the strict main
        # config; keep it available only when strict_vdra=false (ablation).
        if policy_agg == "global_segment_mean" and actor_loss_mode != "vdra_segment_mean_ppo":
            raise ValueError(
                "tree_policy.policy_aggregation=global_segment_mean requires "
                "actor_rollout_ref.actor.policy_loss.loss_mode=vdra_segment_mean_ppo "
                f"(PLAN.md P0.1); got {actor_loss_mode!r}."
            )
        if actor_loss_mode == "vdra_node_balanced_ppo" and policy_agg != "vdra_node_balanced":
            raise ValueError(
                "loss_mode=vdra_node_balanced_ppo is only valid when "
                "tree_policy.policy_aggregation=vdra_node_balanced "
                "(labeled ablation, PLAN.md P0.1)."
            )
