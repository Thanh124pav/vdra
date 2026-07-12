"""Tail-calibration artifact loading and compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_tail_calibration(
    path: str,
    *,
    model: Optional[str] = None,
    pilot_branch_factor: Optional[int] = None,
    likelihood_samples_per_distribution: Optional[int] = None,
    short_horizon: Optional[int] = None,
    quantile: float = 0.99,
) -> Dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"VDRA tail calibration not found: {path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = artifact.get("metadata") or {}
    checks = {
        "model": model,
        "pilot_branch_factor": pilot_branch_factor,
        "likelihood_samples_per_distribution": likelihood_samples_per_distribution,
    }
    for key, expected in checks.items():
        actual = metadata.get(key)
        if expected is not None and actual not in {None, expected, "*"}:
            raise ValueError(
                f"Incompatible VDRA calibration {key}: expected {expected!r}, got {actual!r}"
            )
    horizons = artifact.get("summary", {}).get("per_horizon", {})
    horizon_key = str(short_horizon) if short_horizon is not None else None
    if horizon_key not in horizons:
        if not horizons:
            raise ValueError("VDRA calibration has no per_horizon estimates")
        horizon_key = sorted(horizons, key=int)[-1]
    quantiles = horizons[horizon_key].get("eps_tail_quantiles", {})
    quantile_key = str(quantile)
    if quantile_key not in quantiles:
        raise ValueError(
            f"VDRA calibration horizon {horizon_key} has no quantile {quantile_key}"
        )
    eps_tail = float(quantiles[quantile_key])
    depth_table = {
        int(depth): float(values[quantile_key])
        for depth, values in horizons[horizon_key].get("eps_tail_by_depth", {}).items()
        if quantile_key in values
    }
    return {
        "eps_tail": eps_tail,
        "eps_tail_by_depth": depth_table or None,
        "metadata": metadata,
        "path": str(artifact_path),
    }
