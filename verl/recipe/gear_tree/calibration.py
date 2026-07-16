"""Resolve and validate VDRA tail calibration for recipe workers."""

from typing import Any, Dict

from vdra_core.calibration import load_tail_calibration


def resolve_gear_calibration(gear: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(gear)
    if not resolved.get("strict_vdra", True):
        return resolved
    path = resolved.get("eps_tail_calibration_path")
    if not path:
        raise ValueError(
            "strict VDRA requires eps_tail_calibration_path. Produce one with:\n"
            "  python scripts/calibrate_tail_divergence.py \\\n"
            "    --api-base <vllm>/v1 --model <served-model> \\\n"
            "    --prompts-file <train.jsonl> --k0 8 --r 2 \\\n"
            "    --horizons 8,16,32,60 --full-tokens 512 \\\n"
            "    --out results/tail_calibration.json\n"
            "then set gear_tree.gear.eps_tail_calibration_path to that file "
            "(or set strict_vdra: false to use the raw eps_tail value)."
        )
    calibration = load_tail_calibration(
        str(path),
        model=resolved.get("model") or resolved.get("scorer_model"),
        checkpoint=resolved.get("checkpoint"),
        dataset=resolved.get("dataset"),
        pilot_branch_factor=resolved.get("pilot_branch_factor"),
        likelihood_samples_per_distribution=resolved.get(
            "likelihood_samples_per_distribution", 2
        ),
        short_horizon=resolved.get("tv_second_phase_tokens", 60),
        quantile=resolved.get("eps_tail_quantile", 0.99),
        strict_metadata=resolved.get("strict_vdra", True),
    )
    resolved["eps_tail"] = calibration["eps_tail"]
    resolved["eps_tail_by_depth"] = calibration["eps_tail_by_depth"]
    resolved["eps_tail_source"] = calibration["path"]
    resolved["eps_tail_calibration_metadata"] = calibration["metadata"]
    return resolved
