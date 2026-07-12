"""Resolve and validate VDRA tail calibration for recipe workers."""

from typing import Any, Dict

from vdra_core.calibration import load_tail_calibration


def resolve_gear_calibration(gear: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(gear)
    if not resolved.get("strict_vdra", True):
        return resolved
    path = resolved.get("eps_tail_calibration_path")
    if not path:
        raise ValueError("strict VDRA requires eps_tail_calibration_path")
    calibration = load_tail_calibration(
        str(path),
        pilot_branch_factor=resolved.get("pilot_branch_factor"),
        likelihood_samples_per_distribution=resolved.get(
            "likelihood_samples_per_distribution", 2
        ),
        short_horizon=resolved.get("tv_second_phase_tokens", 60),
    )
    resolved["eps_tail"] = calibration["eps_tail"]
    resolved["eps_tail_by_depth"] = calibration["eps_tail_by_depth"]
    resolved["eps_tail_source"] = calibration["path"]
    return resolved
