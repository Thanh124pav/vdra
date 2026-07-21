"""Policy losses for the tree-family algorithms.

Three losses are registered here:

* ``treetune_ppo`` — the byte-faithful port of treetune's
  ``PPOTrainer._compute_actor_loss`` used by SPO/TreeRL/TreePO/GEAR-*. Kept as
  the SPO baseline and legacy aggregation; see the module docstring below.
* ``vdra_segment_mean_ppo`` — canonical main-run loss family
  (PLAN.md §1.3, 2026-07-21). ``policy_loss.policy_aggregation`` selects:
  ``segment_mean`` (canonical DEFAULT — every original logical segment slot
  weighs ``1/M_B``), ``token_mean`` (every original valid token weighs
  ``1/T_B``), or ``tree_balanced_segment_mean`` (labeled ABLATION — the
  historical ``w_s = 1 / (N_T * N_seg(T))`` tree-balanced objective). The
  canonical denominators ``M_B``/``T_B`` are the PRE-FILTER logical-batch
  counts stamped by the trainer (zero-advantage slots included), so sparse
  tensor execution never changes the objective. No float
  ``segment_objective_weights`` tensor is needed on any canonical path;
  that tensor belongs only to the explicit node-balanced ablation.
* ``vdra_node_balanced_ppo`` — legacy parent-balanced ablation. NOT the main
  VDRA path (PLAN.md P0.1). Kept for controlled comparison runs; it must not
  be selected by the main config.

All three are registered on module import; select via
``actor_rollout_ref.actor.policy_loss.loss_mode=<name>``.

--- treetune_ppo notes (unchanged) ---
Ports ``PPOTrainer._compute_actor_loss`` from
``treetune/trainers/ppo_trainer.py`` (lines 1069-1166) **exactly**, so that the
tree-family algorithms (SPO / TreeRL / TreePO / GEAR-*) keep identical PPO
numerics after moving off DeepSpeed.

Differences vs verl's built-in ``vanilla`` loss that this preserves:
  * ``use_prob_mask``: tokens whose *old* prob >= 0.9 are dropped from the loss
    and from the mean (treetune ppo_trainer.py:1072-1074).
  * log-ratio is masked then clamped to **±10** before ``exp`` (not ±20), and
    there is **no dual-clip** (ppo_trainer.py:1113-1125).
  * ``ratio_threshold`` skip: if mean(ratio) over the action mask exceeds the
    threshold (default 10), the whole batch loss is zeroed (ppo_trainer.py:1155-1160).
  * mean is ``masked_mean`` over the *prob-masked* action mask, i.e. token-mean
    over kept tokens (treetune utils.py:239-245).
"""

from typing import TYPE_CHECKING, Optional

import torch

# PLAN.md §1: the probability-mask predicate is defined ONCE in a
# dependency-light module and shared with extraction-time counting; these
# re-exports keep the historical `policy_loss.<helper>` import paths working.
from recipe.gear_tree.prob_mask import (  # noqa: F401
    DEFAULT_PROBABILITY_MASK_THRESHOLD,
    count_prob_mask_active_tokens,
    effective_action_mask,
    probability_mask_active,
)
from verl.trainer.ppo.core_algos import register_policy_loss

if TYPE_CHECKING:  # ActorConfig is only used as a type hint; avoid a hard import.
    from verl.workers.config import ActorConfig


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean with a finite differentiable zero for empty masks."""

    mask = mask.to(dtype=values.dtype)
    numerator = (values * mask).sum()
    denominator = mask.sum()
    if denominator.item() == 0:
        return numerator * 0.0
    return numerator / denominator


def _weighted_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over valid tokens, optionally normalized by positive edge weights."""

    mask = mask.to(dtype=values.dtype)
    if edge_weights is None:
        return _masked_mean(values, mask)
    weights = edge_weights.to(dtype=values.dtype, device=values.device)
    if weights.shape != values.shape:
        raise ValueError(
            f"edge_weights shape {tuple(weights.shape)} must match values shape {tuple(values.shape)}"
        )
    valid_weights = weights[mask.bool()]
    if valid_weights.numel() and (
        not torch.isfinite(valid_weights).all().item()
        or torch.any(valid_weights <= 0).item()
    ):
        raise ValueError("edge_weights must be finite and strictly positive on valid tokens")
    weighted_mask = mask * weights
    numerator = (values * weighted_mask).sum()
    denominator = weighted_mask.sum()
    if denominator.item() == 0:
        return numerator * 0.0
    return numerator / denominator


def _ppo_clipped_token_surrogate(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    cliprange: float,
    use_prob_mask: bool,
    rollout_is_weights: Optional[torch.Tensor],
    probability_mask_threshold: float = DEFAULT_PROBABILITY_MASK_THRESHOLD,
    is_dummy: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-token PPO-clip surrogate loss and diagnostics.

    Returns ``(pg_losses, action_mask, ratio, pg_losses1, pg_losses2)``. The
    caller decides how to reduce ``pg_losses`` over ``action_mask``, which is
    already the dummy-safe effective mask.
    """

    action_mask = effective_action_mask(
        response_mask,
        old_log_prob,
        use_prob_mask=use_prob_mask,
        probability_mask_threshold=probability_mask_threshold,
        is_dummy=is_dummy,
    )
    action_mask = action_mask.to(dtype=advantages.dtype)

    log_ratio = (log_prob - old_log_prob) * action_mask
    log_ratio_clamped = torch.clamp(log_ratio, -10.0, 10.0)
    ratio = torch.exp(log_ratio_clamped)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_losses = torch.max(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    return pg_losses, action_mask, ratio, pg_losses1, pg_losses2


@register_policy_loss("treetune_ppo")
def compute_policy_loss_treetune(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: "Optional[ActorConfig]" = None,
    rollout_is_weights: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO-clip loss identical to treetune ``_compute_actor_loss`` (PPO-clip core).

    Shapes: all of ``old_log_prob``, ``log_prob``, ``advantages``,
    ``response_mask`` are ``(batch_size, response_length)``.
    """
    assert config is not None

    # treetune hyper-params (defaults match PPOHParams in ppo_trainer.py).
    # PLAN.md P0.F: read PolicyLossConfig fields from config.policy_loss.*.
    cliprange = float(config.clip_ratio)              # PPOHParams.cliprange = 0.2
    use_prob_mask = bool(_resolve_policy_loss_field(config, "use_prob_mask", True))
    # PLAN.md §1: authoritative threshold from the config — never hard-coded.
    probability_mask_threshold = float(
        _resolve_policy_loss_field(
            config, "probability_mask_threshold",
            DEFAULT_PROBABILITY_MASK_THRESHOLD,
        )
    )
    ratio_threshold = float(
        _resolve_policy_loss_field(config, "ratio_threshold", 10.0)
    )

    pg_losses, action_mask, ratio, pg_losses1, pg_losses2 = _ppo_clipped_token_surrogate(
        old_log_prob, log_prob, advantages, response_mask,
        cliprange=cliprange, use_prob_mask=use_prob_mask,
        rollout_is_weights=rollout_is_weights,
        probability_mask_threshold=probability_mask_threshold,
    )

    pg_loss = _weighted_masked_mean(pg_losses, action_mask, edge_weights=edge_weights)

    # --- ratio-threshold skip : ppo_trainer.py:1153-1160 ---
    avg_ratio = _masked_mean(ratio, action_mask)
    is_skipped = False
    if avg_ratio.item() > ratio_threshold:
        pg_loss = pg_loss * 0.0
        is_skipped = True

    # --- metrics : ppo_trainer.py:1162-1172 ---
    pg_clipfrac = _masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)
    approx_kl = 0.5 * _masked_mean((log_prob - old_log_prob) ** 2, action_mask)
    policy_kl = _masked_mean(old_log_prob - log_prob, action_mask)

    _ = policy_kl
    _ = is_skipped
    pg_clipfrac_lower = torch.zeros((), dtype=pg_loss.dtype, device=pg_loss.device)
    return pg_loss, pg_clipfrac, approx_kl, pg_clipfrac_lower


def _reduce_child_to_parent(
    child_losses: torch.Tensor,
    parent_ids: torch.Tensor,
    multiplicities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean of child losses per unique parent id.

    Returns ``(parent_losses, unique_parent_ids)``. Both fresh_iid (all
    multiplicities == 1) and weighted_reuse groups reduce through the same
    formula: L_p = sum_j m_{p,j} L_{p,j} / sum_j m_{p,j}. Under fresh_iid this
    collapses to a plain arithmetic mean over the group's realised children.
    """
    unique_parents, inverse = torch.unique(parent_ids, return_inverse=True)
    num_parents = int(unique_parents.numel())
    parent_num = torch.zeros(num_parents, dtype=child_losses.dtype, device=child_losses.device)
    parent_den = torch.zeros(num_parents, dtype=child_losses.dtype, device=child_losses.device)
    parent_num.index_add_(0, inverse, child_losses * multiplicities)
    parent_den.index_add_(0, inverse, multiplicities)
    # Guard against a zero-weight parent group (shouldn't happen for valid
    # groups; still, keep it differentiable).
    safe_den = torch.where(
        parent_den > 0, parent_den, torch.ones_like(parent_den)
    )
    parent_losses = parent_num / safe_den
    return parent_losses, unique_parents


def _reduce_parent_to_tree(
    parent_losses: torch.Tensor,
    parent_tree_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arithmetic mean of parent losses per unique tree id."""
    unique_trees, inverse = torch.unique(parent_tree_ids, return_inverse=True)
    num_trees = int(unique_trees.numel())
    tree_sum = torch.zeros(num_trees, dtype=parent_losses.dtype, device=parent_losses.device)
    tree_count = torch.zeros(num_trees, dtype=parent_losses.dtype, device=parent_losses.device)
    tree_sum.index_add_(0, inverse, parent_losses)
    tree_count.index_add_(0, inverse, torch.ones_like(parent_losses))
    safe_count = torch.where(
        tree_count > 0, tree_count, torch.ones_like(tree_count)
    )
    return tree_sum / safe_count, unique_trees


def hierarchical_reference_reduction(
    child_losses: torch.Tensor,
    parent_ids: torch.Tensor,
    tree_ids: torch.Tensor,
    multiplicities: torch.Tensor,
) -> torch.Tensor:
    """PLAN.md Section 4.3 reference reduction (used by tests).

    Computes ``mean_T ( mean_p in T ( sum_j m L_{p,j} / sum_j m ) )``. The
    production loss below must match this on the same inputs.
    """
    parent_losses, unique_parents = _reduce_child_to_parent(
        child_losses, parent_ids, multiplicities
    )
    # Each unique parent belongs to exactly one tree by group integrity.
    parent_to_tree = torch.zeros(
        unique_parents.numel(), dtype=tree_ids.dtype, device=tree_ids.device
    )
    for i, pid in enumerate(unique_parents.tolist()):
        matches = (parent_ids == pid).nonzero(as_tuple=True)[0]
        parent_to_tree[i] = tree_ids[matches[0]]
    tree_losses, _ = _reduce_parent_to_tree(parent_losses, parent_to_tree)
    return tree_losses.mean()


_VALID_SEGMENT_TOKEN_REDUCTIONS = ("mean", "sum")


def _resolve_policy_loss_field(config, name: str, default):
    """PLAN.md P0.F: canonical read for fields declared on ``PolicyLossConfig``.

    Prefer ``config.policy_loss.<name>`` (the typed ActorConfig level). Fall
    back to a top-level lookup ONLY when the caller passed the policy-loss
    subconfig (or a bare dict) directly — a duplicate left at the ActorConfig
    top level is never read once ``policy_loss`` carries the field, which
    effectively removes wrong-level overrides.
    """

    if config is None:
        return default
    policy_loss_cfg = None
    try:
        policy_loss_cfg = getattr(config, "policy_loss", None)
    except Exception:  # pragma: no cover — omegaconf edge cases
        policy_loss_cfg = None
    if policy_loss_cfg is None and hasattr(config, "get"):
        policy_loss_cfg = config.get("policy_loss", None)
    if policy_loss_cfg is not None:
        if hasattr(policy_loss_cfg, "get"):
            value = policy_loss_cfg.get(name, None)
        else:
            value = getattr(policy_loss_cfg, name, None)
        if value is not None:
            return value
    # The caller passed the policy-loss subconfig directly (bare
    # PolicyLossConfig / dict), or a legacy pre-migration config.
    if hasattr(config, "get"):
        value = config.get(name, None)
    else:
        value = getattr(config, name, None)
    return default if value is None else value


def _resolve_segment_token_reduction(config) -> str:
    """PLAN.md P0.1: `segment_token_reduction` must be exactly ``mean`` or
    ``sum``. Read from the authoritative source
    ``config.policy_loss.segment_token_reduction`` (typed field on
    ``PolicyLossConfig``); fall back to a legacy top-level lookup only if the
    caller passed a raw ``PolicyLossConfig`` directly. An invalid value is an
    actionable startup error, not a silent fallback.
    """

    raw = None
    if config is not None:
        # Prefer the actor.policy_loss level (canonical, PLAN.md P0.1). Falls
        # through to config.get(...) when the caller passed the policy-loss
        # subconfig directly (e.g. tests that build PolicyLossConfig alone).
        policy_loss_cfg = None
        try:
            policy_loss_cfg = getattr(config, "policy_loss", None)
        except Exception:  # pragma: no cover — omegaconf edge cases
            policy_loss_cfg = None
        if policy_loss_cfg is None and hasattr(config, "get"):
            policy_loss_cfg = config.get("policy_loss", None)
        if policy_loss_cfg is not None and hasattr(policy_loss_cfg, "get"):
            raw = policy_loss_cfg.get("segment_token_reduction", None)
        if raw is None and hasattr(config, "get"):
            # Legacy shape: field lived on ActorConfig top-level. Kept so
            # existing overrides do not silently regress before migration.
            raw = config.get("segment_token_reduction", None)
    if raw is None:
        raw = "mean"
    reduction = str(raw).strip().lower()
    if reduction not in _VALID_SEGMENT_TOKEN_REDUCTIONS:
        raise ValueError(
            f"segment_token_reduction={raw!r} is invalid; must be one of "
            f"{_VALID_SEGMENT_TOKEN_REDUCTIONS} (PLAN.md P0.1)."
        )
    return reduction


def _segment_row_losses(
    pg_losses: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    """PLAN.md P0.1 / P0.4: reduce active token losses per segment row.

    ``mean``: ``L_s = sum_t M_st * ell_st / sum_t M_st``, with an empty-mask
    fallback of zero (still counted in ``N_seg(T)`` upstream).
    ``sum``:  ``L_s = sum_t M_st * ell_st``.
    """

    numerator = (pg_losses * action_mask).sum(dim=-1)
    if reduction == "sum":
        return numerator
    denominator = action_mask.sum(dim=-1)
    empty = denominator <= 0
    safe = torch.where(empty, torch.ones_like(denominator), denominator)
    row_losses = numerator / safe
    return torch.where(empty, torch.zeros_like(row_losses), row_losses)


@register_policy_loss("vdra_segment_mean_ppo")
def compute_policy_loss_vdra_segment_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: "Optional[ActorConfig]" = None,
    rollout_is_weights: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
    segment_objective_weights: torch.Tensor | None = None,
    tree_group_ids: torch.Tensor | None = None,
    tree_total_segment_count: torch.Tensor | None = None,
    original_optimizer_batch_slot_count: int | None = None,
    original_optimizer_batch_tree_count: int | None = None,
    original_logical_segment_count: float | None = None,
    original_logical_response_token_count: float | None = None,
    original_logical_prob_mask_token_count: float | None = None,
    is_dummy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VDRA main-run loss family (PLAN.md §1.3, user decision 2026-07-21).

    Canonical paper objectives (``policy_loss.policy_aggregation``)
    ---------------------------------------------------------------
    Both use the PRE-FILTER denominators of the LOGICAL optimizer batch the
    rows were reserved into (stamped by the trainer, passed by
    ``dp_actor.update_policy``; zero-advantage slots count in the
    denominator even though they carry no tensor rows):

        segment_mean:  L_B = sum_{u in rows} L_u / M_B,
                       L_u = sum_t A_ut * ell_ut / sum_t A_ut
                       (A = the dummy-safe EFFECTIVE action mask; an empty
                        A_u yields a differentiable zero segment loss while
                        the slot still counts in M_B),
                       M_B = original_logical_segment_count

        token_mean:    L_B = sum_{u,t} A_ut * ell_ut / T_B, where the
                       denominator MATCHES the numerator's mask:
                       use_prob_mask=false -> T_B =
                           original_logical_response_token_count
                       use_prob_mask=true  -> T_B =
                           original_logical_prob_mask_token_count
                       A probability-masked numerator is NEVER divided by
                       the unmasked response-token count (PLAN.md §5).

    Example: advantages [pos, neg, 0, 0] -> tensor rows [L1, L2],
    M_B = 4, segment_mean loss = (L1 + L2) / 4.

    These paths return the LOCAL numerator over the fixed logical
    denominator; the distributed dp-size compensation for the averaging
    reducer lives in ``dp_actor`` (``loss_scale_factor``), never here.

    Tree-balanced ablation (``policy_aggregation=tree_balanced_segment_mean``)
    --------------------------------------------------------------------------
    The historical objective, per-row weights over the ORIGINAL optimizer
    batch:

        L_B = sum_{s in retained(B)} w_s * L_s^r,
        w_s = 1 / (N_T * N_seg(T)),

    where ``N_T`` is the number of unique trees in the original optimizer
    batch (passed by ``dp_actor.update_policy`` as
    ``original_optimizer_batch_tree_count`` so microbatch splits preserve
    it) and ``N_seg(T)`` is the PRE-FILTER ``tree_total_segment_count`` of
    the row's tree. Measured NON-parity under multi-rank FSDP dispatch
    (docs/h1_fsdp_parity_report.md) — labeled ablation only.

    ``L_s^r`` is either ``TokenMean(pg_row)`` (``r=mean``) or the raw token
    sum (``r=sum``).

    Retired ``batch_slot_mean_ablation`` flag
    -----------------------------------------
    The flag's mathematics IS the canonical ``segment_mean``; configs that
    still set it fail fast with a migration message (PLAN.md §1.3).

    Legacy tree-average fallbacks
    -----------------------------
    When the tree-balanced ablation inputs are absent,
    ``segment_objective_weights`` (precomputed ``w_s``) or
    ``(tree_group_ids, tree_total_segment_count)`` reproduce the same
    tree-average identity for unit tests; the derived fallback computes
    ``N_T`` locally and is therefore not microbatch-split invariant.
    ``edge_weights`` is a hard error in every path.
    """

    assert config is not None
    if edge_weights is not None:
        raise ValueError(
            "vdra_segment_mean_ppo does not accept edge_weights; the main "
            "VDRA path uses only the segment-average weight (PLAN.md P0.1)."
        )

    reduction = _resolve_segment_token_reduction(config)
    cliprange = float(config.clip_ratio)
    # PLAN.md P0.F: PolicyLossConfig fields read from config.policy_loss.*.
    use_prob_mask = bool(_resolve_policy_loss_field(config, "use_prob_mask", True))
    # PLAN.md §1: authoritative threshold from the config — never hard-coded.
    probability_mask_threshold = float(
        _resolve_policy_loss_field(
            config, "probability_mask_threshold",
            DEFAULT_PROBABILITY_MASK_THRESHOLD,
        )
    )
    # PLAN.md P0.4: report the ratio as a metric; do NOT skip microbatches on
    # the canonical VDRA path.
    ratio_threshold = float(
        _resolve_policy_loss_field(config, "ratio_threshold", float("inf"))
    )

    pg_losses, action_mask, ratio, pg_losses1, pg_losses2 = _ppo_clipped_token_surrogate(
        old_log_prob, log_prob, advantages, response_mask,
        cliprange=cliprange, use_prob_mask=use_prob_mask,
        rollout_is_weights=rollout_is_weights,
        probability_mask_threshold=probability_mask_threshold,
        # PLAN.md §9: dummy rows are masked EXPLICITLY, never left to the
        # probability mask to remove by accident.
        is_dummy=is_dummy,
    )

    row_losses = _segment_row_losses(pg_losses, action_mask, reduction=reduction)

    policy_aggregation = str(
        _resolve_policy_loss_field(config, "policy_aggregation", "segment_mean")
    ).strip().lower()
    if policy_aggregation == "global_segment_mean":
        # PLAN.md §11: cross-level validation canonicalizes this legacy token
        # BEFORE runtime. Seeing it here means validation was bypassed.
        raise RuntimeError(
            "internal configuration error: policy_aggregation="
            "'global_segment_mean' reached the actor loss. Cross-level "
            "validation must canonicalize it to 'tree_balanced_segment_mean' "
            "(non-strict) or reject it (strict); it must NEVER mean "
            "'segment_mean' (PLAN.md §1.3/§11)."
        )
    if policy_aggregation not in (
        "token_mean",
        "segment_mean",
        "tree_balanced_segment_mean",
    ):
        raise ValueError(
            f"policy_loss.policy_aggregation={policy_aggregation!r} is invalid; "
            "must be one of {'token_mean', 'segment_mean', "
            "'tree_balanced_segment_mean'} (PLAN.md §1.3)."
        )

    # PLAN.md §1.3: the batch_slot_mean_ablation flag was RETIRED — its
    # mathematics IS the canonical segment_mean. Reject configs that still
    # set it instead of silently ignoring them.
    if bool(_resolve_policy_loss_field(config, "batch_slot_mean_ablation", False)):
        raise ValueError(
            "policy_loss.batch_slot_mean_ablation was retired (PLAN.md §1.3): "
            "its mathematics is exactly policy_aggregation='segment_mean'. "
            "Remove the flag and set policy_aggregation explicitly."
        )
    if policy_aggregation in ("segment_mean", "token_mean"):
        # Canonical paper objectives — pre-filter LOGICAL batch denominators
        # only. No retained-row fallback, no tree-count approximation
        # (PLAN.md §1.3).
        if policy_aggregation == "segment_mean":
            if original_logical_segment_count is None:
                raise ValueError(
                    "policy_aggregation='segment_mean' requires the "
                    "pre-filter logical-batch slot count "
                    "(original_logical_segment_count stamped by the trainer); "
                    "refusing to fall back to the retained tensor-row count "
                    "(PLAN.md §1.3)."
                )
            m_b = float(original_logical_segment_count)
            if m_b <= 0:
                raise ValueError(
                    "original_logical_segment_count must be > 0 for "
                    "segment_mean; got "
                    f"{original_logical_segment_count!r}."
                )
            pg_loss = row_losses.sum() / m_b
        else:  # token_mean
            if reduction == "sum":
                raise ValueError(
                    "policy_aggregation='token_mean' has no within-segment "
                    "reduction; segment_token_reduction='sum' would be "
                    "silently ignored, which strict VDRA forbids "
                    "(PLAN.md §1.3)."
                )
            # PLAN.md §5: the denominator MUST match the numerator's mask.
            if use_prob_mask:
                raw_t_b = original_logical_prob_mask_token_count
                field = "original_logical_prob_mask_token_count"
                why = (
                    "use_prob_mask=true masks the numerator, so it must be "
                    "normalized by the pre-filter PROBABILITY-MASK-ACTIVE "
                    "token count — never by the unmasked response-token count"
                )
            else:
                raw_t_b = original_logical_response_token_count
                field = "original_logical_response_token_count"
                why = (
                    "use_prob_mask=false keeps every valid response token, so "
                    "the pre-filter RESPONSE-token count is the denominator"
                )
            if raw_t_b is None:
                raise ValueError(
                    f"policy_aggregation='token_mean' requires {field} "
                    "stamped by the trainer; refusing to fall back to the "
                    f"retained-token count (PLAN.md §5). {why}."
                )
            t_b = float(raw_t_b)
            if t_b <= 0:
                raise ValueError(
                    f"{field} must be > 0 for token_mean; got {raw_t_b!r}."
                )
            token_numerator = (pg_losses * action_mask).sum()
            pg_loss = token_numerator / t_b

        avg_ratio = _masked_mean(ratio, action_mask)
        _ = avg_ratio
        _ = ratio_threshold
        pg_clipfrac = _masked_mean(
            torch.gt(pg_losses2, pg_losses1).float(), action_mask
        )
        approx_kl = 0.5 * _masked_mean((log_prob - old_log_prob) ** 2, action_mask)
        pg_clipfrac_lower = torch.zeros((), dtype=pg_loss.dtype, device=pg_loss.device)
        return pg_loss, pg_clipfrac, approx_kl, pg_clipfrac_lower

    if (
        original_optimizer_batch_tree_count is not None
        and tree_total_segment_count is not None
    ):
        # tree_balanced_segment_mean ABLATION: w_s = 1 / (N_T * N_seg(T))
        # with N_T fixed from the ORIGINAL optimizer batch (microbatch-split
        # invariant) and N_seg(T) the pre-filter tree_total_segment_count per
        # row. Measured NON-parity under multi-rank FSDP dispatch
        # (docs/h1_fsdp_parity_report.md) — never a main-run objective.
        n_tree = int(original_optimizer_batch_tree_count)
        if n_tree <= 0:
            raise ValueError(
                "original_optimizer_batch_tree_count must be > 0 for the "
                "VDRA tree-balanced segment-mean ablation."
            )
        counts = tree_total_segment_count.to(
            dtype=row_losses.dtype, device=row_losses.device
        )
        safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
        w = 1.0 / (float(n_tree) * safe_counts)
        pg_loss = (w * row_losses).sum()
    elif segment_objective_weights is not None:
        # Legacy tree-average path. Retained for unit tests and the
        # vdra_node_balanced_ppo ablation; NOT the main path.
        w = segment_objective_weights.to(
            dtype=row_losses.dtype, device=row_losses.device
        )
        if w.shape != row_losses.shape:
            raise ValueError(
                "segment_objective_weights shape must match [batch]; got "
                f"{tuple(w.shape)}"
            )
        pg_loss = (w * row_losses).sum()
    elif tree_group_ids is not None and tree_total_segment_count is not None:
        # Legacy tree-average derived path — kept so pre-P0.4 tests without
        # precomputed weights still exercise the analytic identity. N_T is
        # computed on the local rows, so this is NOT split invariant.
        tids = tree_group_ids.to(dtype=torch.long, device=row_losses.device)
        unique_trees = torch.unique(tids)
        n_tree = float(unique_trees.numel())
        counts = tree_total_segment_count.to(
            dtype=row_losses.dtype, device=row_losses.device
        )
        safe_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
        w = 1.0 / (n_tree * safe_counts)
        pg_loss = (w * row_losses).sum()
    else:
        raise ValueError(
            "vdra_segment_mean_ppo with "
            "policy_aggregation='tree_balanced_segment_mean' requires either "
            "(original_optimizer_batch_tree_count, tree_total_segment_count) "
            "for the split-invariant ablation path, or "
            "segment_objective_weights / (tree_group_ids, "
            "tree_total_segment_count) for the legacy tree-average fallback. "
            "The canonical paper objectives use "
            "policy_aggregation='segment_mean'/'token_mean' with the "
            "trainer-stamped logical denominators (PLAN.md §1.3)."
        )

    # Report ratio as a metric.
    avg_ratio = _masked_mean(ratio, action_mask)
    _ = avg_ratio
    _ = ratio_threshold

    pg_clipfrac = _masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)
    approx_kl = 0.5 * _masked_mean((log_prob - old_log_prob) ** 2, action_mask)
    pg_clipfrac_lower = torch.zeros((), dtype=pg_loss.dtype, device=pg_loss.device)
    return pg_loss, pg_clipfrac, approx_kl, pg_clipfrac_lower


def segment_average_reference(
    row_losses: torch.Tensor,
    tree_ids: torch.Tensor,
    tree_total_segment_count: torch.Tensor,
) -> torch.Tensor:
    """PLAN.md P0.4 test reference: direct evaluation of

        L = (1/N_T) sum_T (1/N_seg(T)) sum_{s in retained(T)} L_s.

    Used by the parity tests. ``tree_total_segment_count`` must be the
    pre-filter ``N_seg(T)`` (not the retained row count).
    """
    tree_ids = tree_ids.to(dtype=torch.long)
    unique_trees = torch.unique(tree_ids)
    total = row_losses.new_zeros(())
    n_tree = float(unique_trees.numel())
    for tid in unique_trees.tolist():
        mask = (tree_ids == tid)
        n_seg = float(tree_total_segment_count[mask][0].item())
        total = total + row_losses[mask].sum() / max(n_seg, 1.0)
    return total / max(n_tree, 1.0)


@register_policy_loss("vdra_node_balanced_ppo")
def compute_policy_loss_vdra_node_balanced(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: "Optional[ActorConfig]" = None,
    rollout_is_weights: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
    parent_group_ids: torch.Tensor | None = None,
    tree_group_ids: torch.Tensor | None = None,
    sample_multiplicity: torch.Tensor | None = None,
    allocated_k: torch.Tensor | None = None,
    objective_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PLAN.md Sections 4.3 & P0.4 hierarchical policy loss.

    The precomputed row-level ``objective_weights`` (P0.3) already encode the
    hierarchical reduction:

        w_{p,j} = (1 / N_tree) * (1 / |P(T)|) * (m_{p,j} / sum_j' m_{p,j'})

    so the loss is a single weighted sum:

        L = sum_row w_row * TokenMean(pg_loss_row).

    Mini/microbatch splits partition ``w``; summing across microbatches yields
    the exact full-batch weighted sum, matching the hierarchical reference.

    Callers must pass ``objective_weights`` (via ``tree_data.edges_to_dataproto``);
    for backward compatibility the loss still accepts ``parent_group_ids`` +
    ``tree_group_ids`` and derives weights from those on-the-fly when
    ``objective_weights`` is missing. Passing ``edge_weights`` remains a
    configuration error.
    """
    assert config is not None
    if edge_weights is not None:
        raise ValueError(
            "vdra_node_balanced_ppo does not accept edge_weights; use "
            "sample_multiplicity + objective_weights (PLAN.md P0.3). Passing "
            "edge_weights would double-count against the hierarchical reduction."
        )

    cliprange = float(config.clip_ratio)
    # PLAN.md P0.F: PolicyLossConfig fields read from config.policy_loss.*.
    use_prob_mask = bool(_resolve_policy_loss_field(config, "use_prob_mask", True))
    # PLAN.md §1: authoritative threshold from the config — never hard-coded.
    probability_mask_threshold = float(
        _resolve_policy_loss_field(
            config, "probability_mask_threshold",
            DEFAULT_PROBABILITY_MASK_THRESHOLD,
        )
    )
    # PLAN.md P0.4: do NOT apply ratio_threshold as a per-microbatch skip on
    # the canonical VDRA path; only report the diagnostic. Legacy skip lives
    # in treetune_ppo.
    ratio_threshold = float(
        _resolve_policy_loss_field(config, "ratio_threshold", float("inf"))
    )

    pg_losses, action_mask, ratio, pg_losses1, pg_losses2 = _ppo_clipped_token_surrogate(
        old_log_prob, log_prob, advantages, response_mask,
        cliprange=cliprange, use_prob_mask=use_prob_mask,
        rollout_is_weights=rollout_is_weights,
        probability_mask_threshold=probability_mask_threshold,
    )

    # Stage 1: token mean per child segment (one row per child).
    # Empty-mask children contribute a finite zero and stay in the parent
    # denominator so the group weight does not silently shift.
    token_num = (pg_losses * action_mask).sum(dim=-1)
    token_den = action_mask.sum(dim=-1)
    empty_child_mask = token_den <= 0
    child_losses = token_num / torch.where(
        empty_child_mask, torch.ones_like(token_den), token_den
    )
    child_losses = torch.where(
        empty_child_mask, torch.zeros_like(child_losses), child_losses
    )

    if objective_weights is not None:
        # P0.4 fast path: single weighted sum. Weights already encode the
        # hierarchical reduction, and mini/microbatch splits preserve the
        # invariant sum_row w_row * child_loss_row.
        w = objective_weights.to(dtype=child_losses.dtype, device=child_losses.device)
        if w.shape != child_losses.shape:
            raise ValueError(
                "objective_weights shape must match [batch]; got "
                f"{tuple(w.shape)}"
            )
        pg_loss = (w * child_losses).sum()
    else:
        # Legacy: derive the hierarchy from the group tensors.
        if parent_group_ids is None or tree_group_ids is None:
            raise ValueError(
                "vdra_node_balanced_ppo requires objective_weights OR "
                "(parent_group_ids, tree_group_ids). Wire them via "
                "tree_data.edges_to_dataproto and add them to model_inputs "
                "in dp_actor."
            )

        # Multiplicities: default to 1 when not provided (== fresh_iid).
        if sample_multiplicity is None:
            multiplicities = torch.ones_like(child_losses)
        else:
            multiplicities = sample_multiplicity.to(
                dtype=child_losses.dtype, device=child_losses.device
            )
            if multiplicities.shape != child_losses.shape:
                raise ValueError(
                    "sample_multiplicity must be a 1-D tensor of shape [batch]; got "
                    f"{tuple(multiplicities.shape)}"
                )

        parent_ids = parent_group_ids.to(dtype=torch.long, device=child_losses.device)
        tree_ids = tree_group_ids.to(dtype=torch.long, device=child_losses.device)

        # Stage 2: child -> parent group.
        parent_losses, unique_parents = _reduce_child_to_parent(
            child_losses, parent_ids, multiplicities
        )

        # Map each parent group to its tree. Group integrity guarantees a single
        # tree per parent, so pick the first row of each unique parent.
        parent_tree_ids = torch.zeros_like(unique_parents)
        for i in range(unique_parents.numel()):
            row = (parent_ids == unique_parents[i]).nonzero(as_tuple=True)[0][0]
            parent_tree_ids[i] = tree_ids[row]

        # Stage 3: parent -> tree.
        tree_losses, _ = _reduce_parent_to_tree(parent_losses, parent_tree_ids)

        # Stage 4: tree -> batch.
        pg_loss = tree_losses.mean()

    # PLAN.md P0.4: report the ratio as a metric; do not skip.
    avg_ratio = _masked_mean(ratio, action_mask)
    _ = avg_ratio
    _ = ratio_threshold

    # Diagnostics.
    pg_clipfrac = _masked_mean(torch.gt(pg_losses2, pg_losses1).float(), action_mask)
    approx_kl = 0.5 * _masked_mean((log_prob - old_log_prob) ** 2, action_mask)
    pg_clipfrac_lower = torch.zeros((), dtype=pg_loss.dtype, device=pg_loss.device)
    return pg_loss, pg_clipfrac, approx_kl, pg_clipfrac_lower
