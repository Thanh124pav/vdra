#!/usr/bin/env python
"""RQ5: does VDRA allocation reduce node-value estimation error?

For a sample of nodes this script:

  1. generates ``--n-ref`` full continuations per node and grades them; their
     mean is the high-budget reference value V_ref(s) and the graded pool is
     the sampling distribution of child values;
  2. generates ``k0`` pilot children + ``r`` short continuations per node and
     computes the VDRA dispersion bound C_s from the §9 tanh TV estimate at
     the runtime horizon;
  3. allocates one shared branch budget (``default_bf x num_nodes``) across
     the nodes with each method (uniform / vdra / random / empirical_variance
     / oracle), simulates the value estimate V_hat(s) by subsampling k_s
     rewards from the node's pool over many seeds, and reports

         MSE_V = E[(V_hat(s) - V_ref(s))^2]

     per method (Summary.md RQ5).

Needs a running OpenAI-compatible vLLM server (``scripts/start_vllm_server.sh``).

Example:
    python scripts/eval_value_mse.py \
        --api-base http://127.0.0.1:8000/v1 --model <served-model> \
        --prompts-file data/math_train.jsonl --num-prompts 16 \
        --n-ref 32 --k0 8 --r 2 --default-bf 6 --seeds 64 \
        --out results/value_mse.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdra_core import allocate_branch_factors, value_gap_bound  # noqa: E402

_CAL_SPEC = importlib.util.spec_from_file_location(
    "_vdra_calibrate", Path(__file__).resolve().parent / "calibrate_tail_divergence.py"
)
_cal = importlib.util.module_from_spec(_CAL_SPEC)
sys.modules.setdefault("_vdra_calibrate", _cal)
_CAL_SPEC.loader.exec_module(_cal)

METHODS = ("uniform", "vdra", "random", "empirical_variance", "oracle")


# --------------------------------------------------------------------------- #
# Pure aggregation (CPU-testable).
# --------------------------------------------------------------------------- #
def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def method_weight(method: str, record: Dict[str, Any], *, k0: int) -> float:
    """Allocation priority weight w_s (k_s ~ w_s) for one node record."""

    rewards = [float(r) for r in record["rewards"]]
    if method == "uniform":
        return 1.0
    if method == "vdra":
        return max(float(record.get("c_s", 0.0)), 0.0) ** 0.5
    if method == "random":
        return 1.0 - random.Random(f"rq5-random:{record['node_id']}").random()
    if method == "empirical_variance":
        # Variance estimate from a pilot-sized sample of graded rollouts.
        return _std(rewards[: max(int(k0), 2)])
    if method == "oracle":
        return _std(rewards)
    raise ValueError(f"Unknown RQ5 method: {method}")


def allocate_by_weights(
    weights: Dict[str, float], *, budget: int, n_min: int
) -> Dict[str, int]:
    nodes = [
        {
            "vdra_node_id": node_id,
            "vdra_dispersion_C": float(w) * float(w),
            # base = n_min floor for every node; cap = the whole budget so the
            # capped water-filling distributes strictly by sqrt(C_s).
            "vdra_default_k": int(n_min),
            "vdra_predicted_k": int(budget),
        }
        for node_id, w in sorted(weights.items())
    ]
    summary = allocate_branch_factors(nodes, total_budget=int(budget), n_min=int(n_min))
    return dict(summary.allocations)


def evaluate_value_mse(
    records: List[Dict[str, Any]],
    *,
    default_bf: int,
    n_min: int,
    seeds: int,
    k0: int,
    methods: tuple = METHODS,
) -> Dict[str, Any]:
    """Simulate V_hat under each allocation and report MSE_V vs V_ref."""

    budget = int(default_bf) * len(records)
    v_ref = {
        rec["node_id"]: sum(rec["rewards"]) / len(rec["rewards"]) for rec in records
    }
    out: Dict[str, Any] = {
        "num_nodes": len(records),
        "budget": budget,
        "default_bf": int(default_bf),
        "n_min": int(n_min),
        "seeds": int(seeds),
        "per_method": {},
    }
    for method in methods:
        weights = {
            rec["node_id"]: method_weight(method, rec, k0=k0) for rec in records
        }
        allocations = allocate_by_weights(weights, budget=budget, n_min=n_min)
        sq_errors: List[float] = []
        for seed in range(int(seeds)):
            rng = random.Random(f"rq5:{method}:{seed}")
            for rec in records:
                pool = [float(r) for r in rec["rewards"]]
                k = max(int(allocations[rec["node_id"]]), 1)
                if k <= len(pool):
                    sample = rng.sample(pool, k)
                else:
                    sample = [rng.choice(pool) for _ in range(k)]
                v_hat = sum(sample) / len(sample)
                sq_errors.append((v_hat - v_ref[rec["node_id"]]) ** 2)
        out["per_method"][method] = {
            "mse_v": sum(sq_errors) / len(sq_errors) if sq_errors else None,
            "mean_k": sum(allocations.values()) / len(allocations),
            "allocations": allocations,
        }
    uniform_mse = out["per_method"]["uniform"]["mse_v"]
    for method in methods:
        mse = out["per_method"][method]["mse_v"]
        out["per_method"][method]["mse_ratio_vs_uniform"] = (
            mse / uniform_mse if uniform_mse else None
        )
    return out


# --------------------------------------------------------------------------- #
# Online data collection (vLLM server).
# --------------------------------------------------------------------------- #
async def build_value_record(
    sampler: Any,
    *,
    node_id: str,
    prompt: str,
    answer: str,
    depth: int,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    node_prefix = prompt
    for _ in range(depth):
        node_prefix += await sampler.generate(node_prefix, args.segment_tokens)

    # Reference pool: graded full continuations.
    rewards = []
    for _ in range(args.n_ref):
        text = await sampler.generate(node_prefix, args.full_tokens)
        rewards.append(
            _cal.simple_math_grade(node_prefix + text, answer, args.answer_prefix)
        )
    if not rewards:
        return None

    # VDRA proxy: pilots + short continuations -> pairwise tanh TV -> C_s.
    pilots = [
        node_prefix + await sampler.generate(node_prefix, args.first_phase_tokens)
        for _ in range(args.k0)
    ]
    conts: List[tuple] = []
    for i, pilot in enumerate(pilots):
        for _ in range(args.r):
            conts.append((i, await sampler.generate(pilot, args.short_horizon)))
    total = 0.0
    pair_count = 0
    for i in range(len(pilots)):
        for j in range(i + 1, len(pilots)):
            cols = [c for c, (origin, _) in enumerate(conts) if origin in (i, j)]
            if not cols:
                continue
            lp_i = [
                sum(await sampler.continuation_token_logps(pilots[i], conts[c][1]))
                for c in cols
            ]
            lp_j = [
                sum(await sampler.continuation_token_logps(pilots[j], conts[c][1]))
                for c in cols
            ]
            tv = _cal.tanh_tv(lp_i, lp_j)
            gap = value_gap_bound(tv, r_max=args.r_max, eps_tail=args.assumed_eps_tail)
            total += gap * gap
            pair_count += 1
    c_s = total / (args.k0 * args.k0) if pair_count else 0.0

    return {"node_id": node_id, "depth": depth, "rewards": rewards, "c_s": c_s}


async def amain(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows = []
    with open(args.prompts_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng.shuffle(rows)
    rows = rows[: args.num_prompts]

    from treetune.gear.vllm_scorer import VLLMLogprobClient

    client = VLLMLogprobClient(api_base=args.api_base, model=args.model)
    sampler = _cal.Sampler(client, temperature=args.temperature)
    records: List[Dict[str, Any]] = []
    try:
        for row_idx, row in enumerate(rows):
            prompt = row.get("prompt") or row.get("problem") or row.get("question")
            answer = row.get("answer") or row.get("solution")
            if not prompt or answer is None:
                continue
            for depth in args.depths:
                record = await build_value_record(
                    sampler,
                    node_id=f"{row_idx}/d{depth}",
                    prompt=str(prompt),
                    answer=str(answer),
                    depth=depth,
                    args=args,
                )
                if record is not None:
                    records.append(record)
                    print(
                        f"[rq5] node={record['node_id']} v_ref="
                        f"{sum(record['rewards']) / len(record['rewards']):.3f} "
                        f"C_s={record['c_s']:.4f}",
                        flush=True,
                    )
    finally:
        close = getattr(client, "aclose", None)
        if close is not None:
            await close()

    summary = evaluate_value_mse(
        records,
        default_bf=args.default_bf,
        n_min=args.n_min,
        seeds=args.seeds,
        k0=args.k0,
    )
    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "dataset": args.dataset,
            "num_prompts": args.num_prompts,
            "n_ref": args.n_ref,
            "k0": args.k0,
            "r": args.r,
            "short_horizon": args.short_horizon,
            "assumed_eps_tail": args.assumed_eps_tail,
            "seed": args.seed,
        },
        "summary": summary,
        "records": records,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[rq5] wrote {args.out}")
    else:
        print(text)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--answer-prefix", default="# Answer\n")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--n-ref", type=int, default=32, help="reference rollouts per node")
    ap.add_argument("--k0", type=int, default=8)
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--short-horizon", type=int, default=60)
    ap.add_argument("--first-phase-tokens", type=int, default=60)
    ap.add_argument("--full-tokens", type=int, default=512)
    ap.add_argument("--segment-tokens", type=int, default=100)
    ap.add_argument("--depths", default="0")
    ap.add_argument("--default-bf", type=int, default=6)
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=64, help="subsampling repetitions")
    ap.add_argument("--assumed-eps-tail", type=float, default=0.0)
    ap.add_argument("--r-max", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.depths = [int(x) for x in str(args.depths).split(",") if x != ""]
    return args


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
