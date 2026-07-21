"""PLAN.md M5: canonical cross-level config validation.

Extracted verbatim from ``RayGearTreeTrainer._validate_replay_startup`` so the
pre-GPU Hydra-composition gate can run the exact validation the trainer runs,
instead of re-implementing it. This module stays engine-free (no torch / verl
/ ray imports) so it is importable from a bare script.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional


def _assign(container: Any, key: str, value: Any) -> None:
    """Write a canonicalized value back into a config container.

    Works for OmegaConf DictConfig, plain dicts, and typed dataclasses whose
    mutability guards require ``object.__setattr__`` (PLAN.md §11).
    """
    try:
        container[key] = value
        return
    except Exception:
        pass
    try:
        setattr(container, key, value)
        return
    except Exception:
        object.__setattr__(container, key, value)


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
    # PLAN.md §1.3 (2026-07-21): `global_segment_mean` was RENAMED to
    # `tree_balanced_segment_mean` and its semantics stay the historical
    # tree-balanced objective. Strict mode fails fast; non-strict legacy
    # loading maps it with a deprecation warning. It NEVER maps to the new
    # uniform `segment_mean` — that would silently change the objective of
    # existing configs and checkpoints.
    # PLAN.md §11: canonicalize the retired name HERE (the single
    # strictness-aware stage) and write the result back to BOTH duplicated
    # fields, so runtime production code only ever sees a canonical value.
    actor_pl = config.actor_rollout_ref.actor.policy_loss
    actor_agg_raw = str(actor_pl.get("policy_aggregation", "segment_mean")).strip().lower()
    if policy_agg == "global_segment_mean" or actor_agg_raw == "global_segment_mean":
        if strict:
            raise ValueError(
                "`global_segment_mean` was renamed to "
                "`tree_balanced_segment_mean`. Use `segment_mean` only if "
                "uniform weighting over all original segment slots is "
                "intended (PLAN.md §1.3)."
            )
        warnings.warn(
            "policy_aggregation='global_segment_mean' is deprecated; mapping "
            "to the tree_balanced_segment_mean ablation (PLAN.md §1.3). "
            "Update the config to the new name.",
            DeprecationWarning,
            stacklevel=2,
        )
        if policy_agg == "global_segment_mean":
            policy_agg = "tree_balanced_segment_mean"
            _assign(tree_policy, "policy_aggregation", policy_agg)
        if actor_agg_raw == "global_segment_mean":
            actor_agg_raw = "tree_balanced_segment_mean"
            _assign(actor_pl, "policy_aggregation", actor_agg_raw)
    _VALID_POLICY_AGGS = (
        "token_mean",
        "segment_mean",
        "tree_balanced_segment_mean",
        "legacy_token_mean",
        "vdra_node_balanced",
    )
    if policy_agg not in _VALID_POLICY_AGGS:
        raise ValueError(
            f"tree_policy.policy_aggregation={policy_agg!r} is invalid; must "
            f"be one of {_VALID_POLICY_AGGS} (PLAN.md §1.3)."
        )
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

    # PLAN.md M5: the node-balanced aggregation and its loss are a matched
    # pair — enforce the mapping in BOTH directions regardless of strict mode.
    # The canonical global_segment_mean aggregation always needs its
    # segment-mean loss; check the aggregation->loss direction first so a
    # misconfigured canonical run reports the segment-mean requirement.
    if policy_agg == "vdra_node_balanced" and actor_loss_mode != "vdra_node_balanced_ppo":
        raise ValueError(
            "tree_policy.policy_aggregation=vdra_node_balanced requires "
            "actor_rollout_ref.actor.policy_loss.loss_mode=vdra_node_balanced_ppo "
            f"(PLAN.md M5); got {actor_loss_mode!r}."
        )
    _SEGMENT_LOSS_AGGS = (
        "token_mean",
        "segment_mean",
        "tree_balanced_segment_mean",
    )
    if policy_agg in _SEGMENT_LOSS_AGGS and actor_loss_mode != "vdra_segment_mean_ppo":
        raise ValueError(
            f"tree_policy.policy_aggregation={policy_agg} requires "
            "actor_rollout_ref.actor.policy_loss.loss_mode=vdra_segment_mean_ppo "
            f"(PLAN.md §1.3/M5); got {actor_loss_mode!r}."
        )
    if actor_loss_mode == "vdra_node_balanced_ppo" and policy_agg != "vdra_node_balanced":
        raise ValueError(
            "loss_mode=vdra_node_balanced_ppo is only valid when "
            "tree_policy.policy_aggregation=vdra_node_balanced "
            "(labeled ablation, PLAN.md M5)."
        )
    # PLAN.md §1.3: `tree_policy.policy_aggregation` and
    # `actor.policy_loss.policy_aggregation` are duplicates that MUST agree
    # (same pattern as segment_token_reduction above), otherwise the actor
    # loss silently optimizes a different objective than the manifest/logs
    # advertise. The retired name is rejected at the actor level too.
    if actor_loss_mode == "vdra_segment_mean_ppo":
        # Already canonicalized above; the legacy token can no longer appear.
        actor_agg = actor_agg_raw
        if policy_agg in _SEGMENT_LOSS_AGGS and actor_agg != policy_agg:
            raise ValueError(
                f"tree_policy.policy_aggregation ({policy_agg!r}) must equal "
                "actor_rollout_ref.actor.policy_loss.policy_aggregation "
                f"({actor_agg!r}) (PLAN.md §1.3)."
            )
        # PLAN.md §1: the probability-mask threshold is authoritative and must
        # be usable; both use_prob_mask values are first-class.
        _thr = float(actor_pl.get("probability_mask_threshold", 0.9))
        if not (0.0 < _thr <= 1.0):
            raise ValueError(
                "actor_rollout_ref.actor.policy_loss."
                f"probability_mask_threshold={_thr!r} is invalid; must "
                "satisfy 0 < threshold <= 1 (PLAN.md §1)."
            )
        # PLAN.md §3: canonical logical-batch VDRA supports the
        # POLICY-GRADIENT objective only. Entropy/KL are per-token auxiliary
        # reductions whose normalization is not yet guaranteed to be
        # invariant to the micro-batch partitioning of a logical batch, so
        # they are refused REGARDLESS of strict_vdra rather than silently
        # producing a partition-dependent objective.
        if policy_agg in ("segment_mean", "token_mean"):
            actor_cfg = config.actor_rollout_ref.actor
            _entropy = float(actor_cfg.get("entropy_coeff", 0.0) or 0.0)
            _use_kl = bool(actor_cfg.get("use_kl_loss", False))
            _kl_coef = float(actor_cfg.get("kl_loss_coef", 0.0) or 0.0)
            if _entropy != 0.0 or _use_kl or _kl_coef != 0.0:
                raise ValueError(
                    "Canonical logical-batch VDRA currently supports the "
                    "policy-gradient objective only. Entropy/KL auxiliary "
                    "reductions are not yet guaranteed to preserve the same "
                    "normalization across microbatches. Disable them or use "
                    "an explicitly supported non-canonical loss path. "
                    f"(policy_aggregation={policy_agg!r}, "
                    f"entropy_coeff={_entropy}, use_kl_loss={_use_kl}, "
                    f"kl_loss_coef={_kl_coef}; PLAN.md §3)"
                )
        # PLAN.md §5: canonical logical batching partitions the reservation
        # into EXACT ppo_mini_batch_size batches before tensor filtering; a
        # tail logical batch is not implemented, so an indivisible reservation
        # would fail at tensorization instead of being gated up front.
        if policy_agg in ("segment_mean", "token_mean"):
            _replay = (gt.get("replay_buffer") or {})
            _underfilled = str(
                _replay.get(
                    "underfilled_update_policy", "postpone_until_divisible"
                )
            )
            if _underfilled == "use_available":
                raise ValueError(
                    "policy_aggregation="
                    f"{policy_agg!r} requires "
                    "gear_tree.replay_buffer.underfilled_update_policy="
                    "'postpone_until_divisible'. Canonical logical batching "
                    "partitions the reservation into exact "
                    "ppo_mini_batch_size logical batches BEFORE tensor "
                    "filtering, and tail logical batches are not implemented "
                    "(PLAN.md §5); 'use_available' would produce an "
                    "indivisible reservation."
                )

    if strict:
        # PLAN.md §1.3/M5: the strict canonical main path must be the exact
        # triple spo / {segment_mean|token_mean} / vdra_segment_mean_ppo.
        # This rejects the deprecated `_original` aliases AND the
        # `*_style_ablation` tree_update_modes on the canonical main path,
        # because neither is `spo`.
        if tree_update_mode != "spo":
            raise ValueError(
                "strict VDRA main runs require tree_update_mode='spo' "
                f"(PLAN.md M5); got {tree_update_mode!r}. The "
                "treepo_style_ablation / treerl_style_ablation / *_original "
                "modes are ablations — set strict_vdra=false to run them."
            )
        if policy_agg not in ("segment_mean", "token_mean"):
            raise ValueError(
                "strict VDRA main runs require "
                "tree_policy.policy_aggregation in ('segment_mean', "
                f"'token_mean') (PLAN.md §1.3); got {policy_agg!r}. Set "
                "strict_vdra=false for the tree_balanced_segment_mean / "
                "legacy_token_mean / vdra_node_balanced ablations."
            )
        if actor_loss_mode != "vdra_segment_mean_ppo":
            raise ValueError(
                "strict VDRA main runs require "
                "actor_rollout_ref.actor.policy_loss.loss_mode="
                f"'vdra_segment_mean_ppo' (PLAN.md M5); got {actor_loss_mode!r}."
            )
        # PLAN.md §1.2 strict sparse-execution contract: exact-zero rows are
        # filtered from TENSOR EXECUTION only, over the logical-slot ledger.
        if not bool(gt.get("only_adv_greater_than_zero", False)):
            raise ValueError(
                "strict VDRA main runs require "
                "gear_tree.only_adv_greater_than_zero=true — the sparse "
                "tensor-execution policy over the logical-slot ledger "
                "(corrected meaning, PLAN.md §1.2). Dense execution of zero "
                "rows is not the canonical strict path."
            )
