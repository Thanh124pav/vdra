"""PLAN.md P0.2 — one context-length contract, resolved in one place.

Kept in its own module (no verl/transformers imports) so CPU unit tests can
exercise the resolver + validator without the full training stack.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence


def _dict_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Support both plain dicts and dot-attr / OmegaConf-style configs."""

    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            # OmegaConf DictConfig.get takes a single arg pre-2.3.
            try:
                return getter(key) or default
            except Exception:
                return default
    return getattr(cfg, key, default)


def resolve_max_edge_prompt_length(data_cfg: Any) -> int:
    """Return L_edge,max in a single place.

    Precedence: ``max_edge_prompt_length`` → ``max_prompt_length``. This value
    MUST be used by both startup validation and every actor-tensorization site
    so a config that clears startup validation cannot then fail training.
    """

    edge_cap = int(_dict_get(data_cfg, "max_edge_prompt_length", 0) or 0)
    if edge_cap > 0:
        return edge_cap
    return int(_dict_get(data_cfg, "max_prompt_length", 0) or 0)


def resolve_max_original_prompt_length(data_cfg: Any) -> int:
    original = int(_dict_get(data_cfg, "max_original_prompt_length", 0) or 0)
    if original > 0:
        return original
    return int(_dict_get(data_cfg, "max_prompt_length", 0) or 0)


def worst_case_edge_prompt_length(
    *, max_original: int, tree_shape: Sequence[int], segment_length: int
) -> int:
    """Length of the deepest legal edge query: L_original + (D-1) * M."""

    max_depth = max(len(list(tree_shape)), 1)
    return int(max_original) + max(max_depth - 1, 0) * int(segment_length or 0)


def validate_context_contract(
    *,
    data_cfg: Any,
    tree_shape: Sequence[int],
    segment_length: int,
    model_context_length: Optional[int] = None,
) -> None:
    """Enforce PLAN.md P0.2: L_original + (D-1) * M <= L_edge,max <= L_model_ctx.

    Raises ``ValueError`` with a diagnostic message on any overflow.
    """

    max_original = resolve_max_original_prompt_length(data_cfg)
    max_edge = resolve_max_edge_prompt_length(data_cfg)
    max_response = int(_dict_get(data_cfg, "max_response_length", 0) or 0)
    worst_case = worst_case_edge_prompt_length(
        max_original=max_original,
        tree_shape=tree_shape,
        segment_length=segment_length,
    )
    if max_edge > 0 and worst_case > max_edge:
        raise ValueError(
            "context-length bound overflow (PLAN.md P0.2/P0.5): "
            f"max_original_prompt_length={max_original}, deepest edge "
            f"depth={max(len(list(tree_shape)), 1)}, "
            f"segment_length={segment_length}, worst-case edge "
            f"query length={worst_case} > max_edge_prompt_length={max_edge}. "
            "Either reduce M or tree_shape depth, pre-filter prompts to "
            "reserve segment headroom, or raise data.max_edge_prompt_length "
            "(within the model context length)."
        )
    if (
        max_edge > 0
        and max_response > 0
        and model_context_length is not None
        and int(model_context_length) > 0
    ):
        model_ctx = int(model_context_length)
        if max_edge + max_response > model_ctx:
            raise ValueError(
                "context-length bound overflow (PLAN.md P0.2): "
                f"max_edge_prompt_length={max_edge} + "
                f"max_response_length={max_response} = "
                f"{max_edge + max_response} exceeds resolved model context "
                f"length {model_ctx}. Lower one of the two limits or raise "
                "the model context."
            )
