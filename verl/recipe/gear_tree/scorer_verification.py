"""PLAN.md P0.5: two-mode rollout/scorer weight-version verification.

Extracted from :mod:`recipe.gear_tree.gear_ray_trainer` so the two-mode
contract can be unit-tested on CPU without importing the full Ray/torch-data
trainer stack.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple


def resolve_endpoints(gear_cfg: Mapping[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Return ``(uses_rollout_server, rollout_api_base, scorer_api_base)``.

    Raises ``ValueError`` in strict main runs if the config does not clearly
    pick one of the two supported modes.
    """
    strict = bool(gear_cfg.get("strict_vdra", True))
    uses_rollout_server = bool(gear_cfg.get("scorer_uses_rollout_server", False))
    rollout_api_base = gear_cfg.get("rollout_api_base")
    scorer_api_base = gear_cfg.get("scorer_api_base")

    if uses_rollout_server:
        if rollout_api_base and str(rollout_api_base) != str(scorer_api_base):
            raise ValueError(
                "scorer_uses_rollout_server=true forbids a distinct "
                "rollout_api_base (PLAN.md P0.5); leave rollout_api_base "
                "null in same-server mode."
            )
    else:
        if strict and not rollout_api_base:
            raise ValueError(
                "strict VDRA main runs require an explicit rollout_api_base "
                "when scorer_uses_rollout_server is false (PLAN.md P0.5)."
            )
        if strict and not scorer_api_base:
            raise ValueError(
                "strict VDRA main runs require an explicit scorer_api_base "
                "(PLAN.md P0.5)."
            )
    return uses_rollout_server, rollout_api_base, scorer_api_base


def fetch_rollout_weight_version(
    gear_cfg: Mapping[str, Any],
    *,
    fetch_fn: Callable[..., Optional[str]],
) -> Optional[str]:
    """Fetch the rollout replica's server-reported weight version.

    ``fetch_fn(api_base, api_key=..., timeout=...)`` is the injectable
    probe (``recipe.gear_tree.gear_core.gear.vllm_scorer.fetch_server_weight_version``
    in production). Strict main runs must obtain a non-empty fingerprint —
    a ``None`` return raises so the trainer stops before allocation.
    """
    strict = bool(gear_cfg.get("strict_vdra", True))
    uses_rollout_server, rollout_api_base, scorer_api_base = resolve_endpoints(gear_cfg)
    api_base = scorer_api_base if uses_rollout_server else (rollout_api_base or scorer_api_base)
    if not api_base:
        return None
    try:
        version = fetch_fn(
            str(api_base),
            api_key=str(gear_cfg.get("scorer_api_key", "EMPTY")),
            timeout=float(gear_cfg.get("scorer_version_timeout", 5.0)),
        )
    except Exception:
        if strict:
            raise
        return None
    if strict and version is None:
        raise RuntimeError(
            f"strict VDRA main runs require a server-reported weight "
            f"version from rollout endpoint {api_base!r} (PLAN.md P0.5)."
        )
    return version
