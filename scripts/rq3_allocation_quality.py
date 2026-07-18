"""PLAN.md §9 RQ3 — allocation-quality comparison scaffolding.

Compares five allocators on a synthetic queue whose oracle dispersions are
known:

    * ``fixed``               — every prefix gets the queue's mean budget.
    * ``random``              — sampled uniformly from [n_min, u_p].
    * ``uncertainty``         — pick proportional to the model's own naive
                                perplexity-driven uncertainty (a stand-in
                                for a simple heuristic proxy).
    * ``empirical_variance``  — measure variance of a few free rollouts and
                                allocate proportional to sqrt(variance).
    * ``vdra``                — the canonical bounded integer marginal
                                allocator with the oracle dispersion.

For each allocator the script computes the queue objective
    Obj = sum_p C_p / k_p
which is the very quantity VDRA minimises. Lower is better. The synthetic
setup replaces the real rollout server with a callable, so this script runs
on CPU without loading verl or vLLM. Under a real cluster the callables can
be swapped for the async rollout paths.

Run:
    PYTHONPATH=verl:. python scripts/rq3_allocation_quality.py \
        --num-prefixes 8 --budget 32 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

sys.path.insert(0, "verl")

from vdra_core import allocate_branch_factors


# --- synthetic oracle -------------------------------------------------------
def sample_oracle_prefixes(num_prefixes: int, seed: int) -> List[Dict[str, Any]]:
    """Draw a random queue of prefixes with known dispersion C_p in [0, 1]."""

    rng = random.Random(seed)
    prefixes: List[Dict[str, Any]] = []
    for i in range(num_prefixes):
        c_true = round(rng.random(), 4)
        # A naive uncertainty proxy for the "uncertainty" allocator: perplexity
        # in [1.0, 3.0].
        naive = round(1.0 + 2.0 * rng.random(), 3)
        prefixes.append({"id": f"n{i}", "C_true": c_true, "naive_uncertainty": naive})
    return prefixes


# --- allocator plug-ins -----------------------------------------------------
Allocation = Dict[str, int]


def _clip(alloc: Dict[str, int], *, n_min: int, u: int) -> Dict[str, int]:
    return {k: max(n_min, min(v, u)) for k, v in alloc.items()}


def _scale_to_budget(alloc: Dict[str, int], budget: int, *, n_min: int, u: int) -> Dict[str, int]:
    """Deterministically clip + rescale integer allocations to fit budget."""
    keys = sorted(alloc)
    result = _clip(alloc, n_min=n_min, u=u)
    total = sum(result.values())
    # Adjust downwards or upwards by unit increments to hit exactly budget.
    while total > budget:
        for k in keys:
            if total <= budget:
                break
            if result[k] > n_min:
                result[k] -= 1
                total -= 1
    while total < budget:
        for k in keys:
            if total >= budget:
                break
            if result[k] < u:
                result[k] += 1
                total += 1
    return result


def allocate_fixed(
    prefixes: Sequence[Dict[str, Any]], budget: int, *, n_min: int, u: int
) -> Allocation:
    n = len(prefixes)
    per = max(min(budget // n, u), n_min)
    return _scale_to_budget({p["id"]: per for p in prefixes}, budget, n_min=n_min, u=u)


def allocate_random(
    prefixes: Sequence[Dict[str, Any]], budget: int, *, n_min: int, u: int, seed: int
) -> Allocation:
    rng = random.Random(seed)
    raw = {p["id"]: rng.randint(n_min, u) for p in prefixes}
    return _scale_to_budget(raw, budget, n_min=n_min, u=u)


def allocate_uncertainty(
    prefixes: Sequence[Dict[str, Any]], budget: int, *, n_min: int, u: int
) -> Allocation:
    weights = {p["id"]: max(float(p.get("naive_uncertainty", 1.0)), 1e-9) for p in prefixes}
    total_w = sum(weights.values())
    raw = {k: max(int(round(budget * w / total_w)), n_min) for k, w in weights.items()}
    return _scale_to_budget(raw, budget, n_min=n_min, u=u)


def allocate_empirical_variance(
    prefixes: Sequence[Dict[str, Any]],
    budget: int,
    *,
    n_min: int,
    u: int,
    rollout_fn: Callable[[Dict[str, Any], int], List[float]],
    probe_rollouts: int = 4,
) -> Allocation:
    """Estimate variance from a few free rollouts, allocate proportional to sqrt(var)."""
    weights: Dict[str, float] = {}
    for p in prefixes:
        samples = rollout_fn(p, probe_rollouts)
        if len(samples) < 2:
            weights[p["id"]] = 1e-3
            continue
        mu = sum(samples) / len(samples)
        var = sum((s - mu) ** 2 for s in samples) / len(samples)
        weights[p["id"]] = math.sqrt(max(var, 1e-12))
    total_w = sum(weights.values()) or 1.0
    raw = {k: max(int(round(budget * w / total_w)), n_min) for k, w in weights.items()}
    return _scale_to_budget(raw, budget, n_min=n_min, u=u)


def allocate_vdra(
    prefixes: Sequence[Dict[str, Any]], budget: int, *, n_min: int, u: int
) -> Allocation:
    nodes = [
        {"id": p["id"], "vdra_dispersion_C": float(p["C_true"]), "vdra_default_k": u}
        for p in prefixes
    ]
    summary = allocate_branch_factors(
        nodes,
        total_budget=budget,
        n_min=n_min,
        max_k_per_node=u,
        predicted_k_cap_mode="configured_max_for_all_nodes",
        infeasible_upper_policy="expand_nonredundant_caps",
    )
    return dict(summary.allocations)


# --- objective + reporter --------------------------------------------------
def queue_objective(prefixes: Sequence[Dict[str, Any]], allocation: Allocation) -> float:
    total = 0.0
    for p in prefixes:
        k = max(int(allocation.get(p["id"], 1)), 1)
        total += float(p["C_true"]) / k
    return total


@dataclass
class AllocatorResult:
    name: str
    allocation: Allocation
    objective: float


def run_all(
    prefixes: Sequence[Dict[str, Any]],
    *,
    budget: int,
    n_min: int,
    u: int,
    rollout_fn: Callable[[Dict[str, Any], int], List[float]],
    seed: int,
) -> List[AllocatorResult]:
    results = [
        AllocatorResult(
            "fixed",
            alloc := allocate_fixed(prefixes, budget, n_min=n_min, u=u),
            queue_objective(prefixes, alloc),
        ),
        AllocatorResult(
            "random",
            alloc := allocate_random(prefixes, budget, n_min=n_min, u=u, seed=seed),
            queue_objective(prefixes, alloc),
        ),
        AllocatorResult(
            "uncertainty",
            alloc := allocate_uncertainty(prefixes, budget, n_min=n_min, u=u),
            queue_objective(prefixes, alloc),
        ),
        AllocatorResult(
            "empirical_variance",
            alloc := allocate_empirical_variance(
                prefixes, budget, n_min=n_min, u=u, rollout_fn=rollout_fn
            ),
            queue_objective(prefixes, alloc),
        ),
        AllocatorResult(
            "vdra",
            alloc := allocate_vdra(prefixes, budget, n_min=n_min, u=u),
            queue_objective(prefixes, alloc),
        ),
    ]
    return results


def _synthetic_rollout(prefix: Dict[str, Any], count: int) -> List[float]:
    """Draw ``count`` rewards from a Bernoulli whose variance == C_true."""
    rng = random.Random(hash(prefix["id"]) & 0xFFFF)
    p_success = 0.5 + math.sqrt(max(0.25 - prefix["C_true"], 0.0))
    p_success = min(max(p_success, 0.0), 1.0)
    return [1.0 if rng.random() < p_success else 0.0 for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--u", type=int, default=8)
    args = parser.parse_args()

    prefixes = sample_oracle_prefixes(args.num_prefixes, args.seed)
    results = run_all(
        prefixes,
        budget=args.budget,
        n_min=args.n_min,
        u=args.u,
        rollout_fn=_synthetic_rollout,
        seed=args.seed,
    )
    print(json.dumps(
        {
            "prefixes": prefixes,
            "budget": args.budget,
            "results": [
                {"name": r.name, "allocation": r.allocation, "objective": r.objective}
                for r in results
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":  # pragma: no cover
    main()
