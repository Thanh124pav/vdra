#!/usr/bin/env python3
"""CPU-only microbenchmark for the exact VDRA integer allocation solver."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vdra_core.core import allocate_branch_factors_integer


def make_nodes(queue_size: int, target_budget: int) -> list[dict[str, float | int | str]]:
    default_k = max(target_budget // max(queue_size, 1), 1)
    nodes = []
    for idx in range(queue_size):
        predicted = max(default_k // 2, 1) if idx % 5 == 0 else default_k
        nodes.append(
            {
                "gear_segment_id": f"node-{idx:03d}",
                "vdra_default_k": default_k,
                "vdra_predicted_k": predicted,
                "vdra_dispersion_C": 0.01 + ((idx * 37) % 101) / 100.0,
            }
        )
    return nodes


def run_benchmark(
    *,
    queue_size: int = 32,
    target_budget: int = 512,
    rounds: int = 200,
    n_min: int = 1,
    max_k_per_node: int = 32,
) -> dict[str, float]:
    nodes = make_nodes(queue_size, target_budget)
    samples = []
    for _ in range(rounds):
        fresh_nodes = [dict(node) for node in nodes]
        summary = allocate_branch_factors_integer(
            fresh_nodes,
            total_budget=target_budget,
            n_min=n_min,
            max_k_per_node=max_k_per_node,
            max_repair_k_per_node=max_k_per_node,
        )
        samples.append(float(summary.solver_time_ms))
        if summary.allocated_budget != target_budget:
            raise RuntimeError("allocation budget was not preserved")
    samples_sorted = sorted(samples)
    p99_index = min(len(samples_sorted) - 1, int(0.99 * (len(samples_sorted) - 1)))
    return {
        "allocation/solver_time_ms_median": float(statistics.median(samples_sorted)),
        "allocation/solver_time_ms_p99": float(samples_sorted[p99_index]),
        "allocation/solver_time_ms_max": float(max(samples_sorted)),
        "allocation/queue_size": float(queue_size),
        "allocation/target_budget": float(target_budget),
        "allocation/increment_steps": float(target_budget - queue_size * n_min),
        "allocation/rounds": float(rounds),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--target-budget", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--median-threshold-ms", type=float, default=1.0)
    parser.add_argument("--p99-threshold-ms", type=float, default=5.0)
    parser.add_argument("--no-enforce-targets", action="store_true")
    args = parser.parse_args(argv)

    metrics = run_benchmark(
        queue_size=args.queue_size,
        target_budget=args.target_budget,
        rounds=args.rounds,
    )
    print(json.dumps(metrics, sort_keys=True))
    if not args.no_enforce_targets:
        if metrics["allocation/solver_time_ms_median"] >= args.median_threshold_ms:
            raise SystemExit(
                f"median solver time {metrics['allocation/solver_time_ms_median']:.6g} ms "
                f">= {args.median_threshold_ms:.6g} ms"
            )
        if metrics["allocation/solver_time_ms_p99"] >= args.p99_threshold_ms:
            raise SystemExit(
                f"p99 solver time {metrics['allocation/solver_time_ms_p99']:.6g} ms "
                f">= {args.p99_threshold_ms:.6g} ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
