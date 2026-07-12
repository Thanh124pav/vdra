"""Budget-preserving bounded rounding strategies for VDRA ablations."""

import math
import random
from typing import Dict, Mapping


def round_bounded(
    raw: Mapping[str, float],
    *,
    lower: Mapping[str, int],
    upper: Mapping[str, int],
    target: int,
    strategy: str = "largest_remainder",
    seed: int = 0,
) -> Dict[str, int]:
    if strategy not in {"largest_remainder", "nearest_repair", "stochastic"}:
        raise ValueError(f"Unsupported VDRA rounding strategy: {strategy}")
    rng = random.Random(seed)
    if strategy == "nearest_repair":
        out = {
            key: min(max(int(math.floor(value + 0.5)), lower[key]), upper[key])
            for key, value in raw.items()
        }
    elif strategy == "stochastic":
        out = {}
        for key, value in raw.items():
            floor_value = math.floor(value)
            rounded = floor_value + int(rng.random() < value - floor_value)
            out[key] = min(max(rounded, lower[key]), upper[key])
    else:
        out = {
            key: min(max(int(math.floor(value)), lower[key]), upper[key])
            for key, value in raw.items()
        }

    while sum(out.values()) < target:
        eligible = [key for key in raw if out[key] < upper[key]]
        if not eligible:
            break
        key = min(eligible, key=lambda item: (-(raw[item] - out[item]), item))
        out[key] += 1
    while sum(out.values()) > target:
        eligible = [key for key in raw if out[key] > lower[key]]
        if not eligible:
            break
        key = min(eligible, key=lambda item: (raw[item] - out[item], item))
        out[key] -= 1
    return out
