"""PLAN.md §9 RQ4 — proxy / estimation validation scaffolding.

Compares the VDRA proxy dispersion ``C_p`` against three references:

  * ``empirical_dispersion``  — sample variance of ``num_free_rollouts``
    free continuations, per prefix.
  * ``value_mse_vs_budget``   — MSE of the pilot-driven parent value
    estimate at multiple budgets K, so the curve MSE(K) reveals whether
    the estimator collapses as K -> infty.
  * ``high_budget_reference`` — objective at a very large K used as a
    proxy for the truth; the VDRA-vs-reference gap is the value-induced
    gradient error surrogate.

The script is CPU-mockable: pass a callable ``score_prefix(prefix, k)``
that returns ``k`` continuation rewards, and the harness measures the
proxy quality in a real experiment.

Run:
    PYTHONPATH=verl:. python scripts/rq4_proxy_mse.py \\
        --num-prefixes 6 --seed 0 --budgets 2,4,8,16
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from typing import Callable, Dict, List, Sequence

sys.path.insert(0, "verl")


def _synthetic_prefix(idx: int, seed: int) -> Dict[str, float]:
    """Prefix with a hidden true reward mean p and true variance p(1-p)."""
    rng = random.Random(f"prefix:{idx}:{seed}")
    p = round(rng.random(), 3)
    return {"id": f"n{idx}", "p_true": p, "var_true": p * (1.0 - p)}


def synthetic_scorer(prefix: Dict[str, float], k: int, seed: int = 0) -> List[float]:
    rng = random.Random(f"score:{prefix['id']}:{seed}")
    return [1.0 if rng.random() < prefix["p_true"] else 0.0 for _ in range(k)]


def empirical_dispersion(samples: Sequence[float]) -> float:
    if len(samples) < 2:
        return 0.0
    mu = sum(samples) / len(samples)
    return sum((s - mu) ** 2 for s in samples) / len(samples)


def value_estimate(samples: Sequence[float]) -> float:
    return sum(samples) / max(len(samples), 1)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def run_evaluation(
    prefixes: Sequence[Dict[str, float]],
    *,
    budgets: Sequence[int],
    reference_budget: int,
    scorer: Callable[[Dict[str, float], int], List[float]],
    proxy_C: Callable[[Dict[str, float]], float] = lambda p: p["var_true"],
) -> Dict[str, object]:
    """Produce the RQ4 diagnostic table.

    ``proxy_C`` defaults to the ground-truth variance (a stand-in). Plug the
    real VDRA proxy — e.g., a wrapper over ``compute_tanh_tv_bound`` — for
    the paper.
    """
    per_prefix: List[Dict[str, float]] = []
    for prefix in prefixes:
        entry: Dict[str, float] = {"id": prefix["id"], "C_proxy": proxy_C(prefix)}
        # Empirical dispersion at every budget.
        for K in budgets:
            samples = scorer(prefix, K)
            entry[f"C_empirical_K{K}"] = empirical_dispersion(samples)
            entry[f"value_K{K}"] = value_estimate(samples)
        ref_samples = scorer(prefix, reference_budget)
        entry["value_reference"] = value_estimate(ref_samples)
        per_prefix.append(entry)

    # Aggregate: correlation between C_proxy and largest-budget C_empirical.
    biggest_K = max(budgets)
    corr = _pearson(
        [p["C_proxy"] for p in per_prefix],
        [p[f"C_empirical_K{biggest_K}"] for p in per_prefix],
    )

    # Value MSE curve: mean over prefixes of (value_K - value_ref)^2.
    mse_by_K = {
        K: statistics.mean(
            (p[f"value_K{K}"] - p["value_reference"]) ** 2 for p in per_prefix
        )
        for K in budgets
    }

    return {
        "per_prefix": per_prefix,
        "C_proxy_vs_empirical_pearson": corr,
        "value_mse_by_budget": mse_by_K,
        "reference_budget": reference_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-prefixes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", type=str, default="2,4,8,16")
    parser.add_argument("--reference-budget", type=int, default=256)
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    prefixes = [_synthetic_prefix(i, args.seed) for i in range(args.num_prefixes)]
    result = run_evaluation(
        prefixes,
        budgets=budgets,
        reference_budget=args.reference_budget,
        scorer=synthetic_scorer,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
