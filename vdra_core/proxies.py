"""Dispersion proxies used by VDRA structural ablations."""

from __future__ import annotations

import random
from typing import Any, Mapping, Tuple

from .logging_schema import node_id

PairKey = Tuple[int, int]


def select_dispersion_proxy(
    method: str,
    *,
    vdra_dispersion_C: float,
    pair_tvs: Mapping[PairKey, float],
    pilot_count: int,
    node: Mapping[str, Any],
) -> float:
    if method == "vdra":
        return max(float(vdra_dispersion_C), 0.0)
    if method == "uniform":
        return 1.0
    if method == "random":
        # RQ1 baseline "random non-uniform allocation": a seeded uniform draw
        # in (0, 1] per node, so allocation is non-uniform but carries no
        # dispersion signal. Deterministic per node id for reproducibility.
        rng = random.Random(f"vdra-random-proxy:{node_id(node)}")
        return 1.0 - rng.random()
    if method == "direct_tv":
        n = max(int(pilot_count), 1)
        return sum(float(tv) ** 2 for tv in pair_tvs.values()) / float(n * n)
    field_by_method = {
        "empirical_variance": "vdra_empirical_reward_variance",
        "external_score": "vdra_external_dispersion_C",
        "oracle": "vdra_oracle_value_dispersion",
    }
    if method not in field_by_method:
        raise ValueError(f"Unsupported VDRA allocation proxy: {method}")
    field = field_by_method[method]
    if node.get(field) is None:
        raise ValueError(f"VDRA allocation proxy {method!r} requires node field {field!r}")
    return max(float(node[field]), 0.0)
