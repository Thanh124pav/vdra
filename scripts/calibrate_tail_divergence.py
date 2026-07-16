#!/usr/bin/env python
"""Validate the VDRA short-horizon divergence assumptions (Summary.md RQ2-RQ4, Direction A/C/D).

For a sample of nodes this script:

  1. generates ``k0`` pilot children per node and ``r`` continuations per child;
  2. scores every continuation under every pilot prefix ONCE (per-token prompt
     logprobs), so cumulative sums give the §9 tanh TV estimate ``D_m`` at every
     horizon ``m`` simultaneously, plus the full-horizon estimate ``D_L``;
  3. reports, per horizon: Spearman / Pearson correlation between ``D_m`` and
     ``D_L`` (RQ2), the tail-ratio distribution

         r_ij = (D_L - D_m)_+ / (1 - D_m + delta)

     and its high quantiles -> calibrated ``eps_tail`` (RQ3), globally and per
     tree depth (for the depth-dependent variant);
  4. optionally (``--grade``) estimates oracle child values from full
     continuations, giving oracle node dispersion sigma^2_s, its correlation
     with the proxy ``C_s`` and the bound ratio (RQ4), and the allocation
     regret J(k_uniform) / J(k_VDRA) / J(k_oracle) (Direction D).

Needs only a running OpenAI-compatible vLLM server (``scripts/start_vllm_server.sh``).

Example:
    python scripts/calibrate_tail_divergence.py \
        --api-base http://127.0.0.1:8000/v1 --model <served-model> \
        --prompts-file data/math_train.jsonl --num-prompts 16 \
        --k0 4 --r 2 --horizons 8,16,32,64 --full-tokens 512 \
        --depths 0,1 --grade --out results/tail_calibration.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from treetune.gear.budget_allocation import (  # noqa: E402
    allocate_branch_factors,
    value_gap_bound,
)


# --------------------------------------------------------------------------- #
# Small self-contained statistics helpers (no scipy dependency).
# --------------------------------------------------------------------------- #
def _rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return pearson(_rank(xs), _rank(ys))


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    pos = min(max(q, 0.0), 1.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def tanh_tv(logps_i: Sequence[float], logps_j: Sequence[float]) -> float:
    """Summary.md §9: mean_z |tanh((log P_i(z) - log P_j(z)) / 2)|."""
    vals = [
        abs(math.tanh((a - b) / 2.0))
        for a, b in zip(logps_i, logps_j)
        if math.isfinite(a) and math.isfinite(b)
    ]
    return sum(vals) / len(vals) if vals else 0.0


def simple_math_grade(response: str, answer: str, answer_prefix: str) -> float:
    """Exact-match grade on the text after ``answer_prefix`` (fallback grader)."""
    if answer_prefix in response:
        predicted = response.split(answer_prefix, 1)[1].strip().split("\n")[0].strip()
        return 1.0 if predicted == str(answer).strip() else 0.0
    return 0.0


# --------------------------------------------------------------------------- #
# Generation / scoring against the vLLM server.
# --------------------------------------------------------------------------- #
class Sampler:
    def __init__(self, client: Any, temperature: float):
        self.client = client
        self.temperature = temperature

    async def generate(self, prompt: str, max_tokens: int) -> str:
        text, _, _ = await self.client.completion_with_token_entropies(
            prompt, max_tokens=max_tokens, temperature=self.temperature, top_logprobs=1
        )
        return text

    async def continuation_token_logps(self, prefix: str, continuation: str) -> List[float]:
        """Per-token log P(continuation | prefix) via echo prompt_logprobs.

        The continuation's token logprobs are the tail of the full-prompt
        logprob list; cumulative sums over this tail give log P(z_{1:m}|prefix)
        for every horizon m at once.
        """
        full_lps = await self.client.prompt_logprobs(prefix + continuation)
        prefix_lps = await self.client.prompt_logprobs(prefix)
        tail = full_lps[len(prefix_lps):]
        return [float(lp) for lp in tail if lp is not None]


async def build_node_record(
    sampler: Sampler,
    *,
    prompt: str,
    answer: Optional[str],
    depth: int,
    args: argparse.Namespace,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    # Walk `depth` segments down from the prompt to sample a deeper tree node.
    node_prefix = prompt
    for _ in range(depth):
        node_prefix += await sampler.generate(node_prefix, args.segment_tokens)

    pilots = [
        node_prefix + await sampler.generate(node_prefix, args.first_phase_tokens)
        for _ in range(args.k0)
    ]
    # r continuations per pilot; each is scored under EVERY pilot prefix.
    conts: List[Tuple[int, str]] = []  # (origin pilot index, continuation text)
    for i, pilot in enumerate(pilots):
        for _ in range(args.r):
            conts.append((i, await sampler.generate(pilot, args.full_tokens)))
    if len(conts) < 2:
        return None

    # logps[i][c] = per-token log P(cont_c | pilot_i).
    logps: List[List[List[float]]] = []
    for pilot in pilots:
        row = await asyncio.gather(
            *[sampler.continuation_token_logps(pilot, text) for _, text in conts]
        )
        logps.append([list(lps) for lps in row])

    horizons = sorted(set(args.horizons))
    pair_records = []
    for i in range(args.k0):
        for j in range(i + 1, args.k0):
            cols = [c for c, (origin, _) in enumerate(conts) if origin in (i, j)]
            if not cols:
                continue

            def _sums(row: List[List[float]], m: Optional[int]) -> List[float]:
                return [
                    sum(row[c][:m] if m is not None else row[c]) for c in cols
                ]

            d_l = tanh_tv(_sums(logps[i], None), _sums(logps[j], None))
            d_by_m = {
                m: tanh_tv(_sums(logps[i], m), _sums(logps[j], m)) for m in horizons
            }
            pair_records.append({"pair": (i, j), "d_l": d_l, "d_m": d_by_m})

    record: Dict[str, Any] = {
        "depth": depth,
        "k0": args.k0,
        "pairs": pair_records,
    }

    if args.grade and answer is not None:
        # Oracle child values: mean grade over each pilot's own continuations.
        values = []
        for i, pilot in enumerate(pilots):
            grades = [
                simple_math_grade(pilot + text, answer, args.answer_prefix)
                for origin, text in conts
                if origin == i
            ]
            values.append(sum(grades) / len(grades) if grades else 0.0)
        mean_v = sum(values) / len(values)
        record["oracle_child_values"] = values
        record["sigma2_oracle"] = sum((v - mean_v) ** 2 for v in values) / len(values)
    return record


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def summarize(records: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    horizons = sorted(set(args.horizons))
    summary: Dict[str, Any] = {"num_nodes": len(records), "horizons": horizons}

    per_horizon: Dict[int, Dict[str, Any]] = {}
    for m in horizons:
        d_ms: List[float] = []
        d_ls: List[float] = []
        ratios_global: List[float] = []
        ratios_by_depth: Dict[int, List[float]] = {}
        for rec in records:
            for pr in rec["pairs"]:
                d_m = pr["d_m"][m]
                d_l = pr["d_l"]
                d_ms.append(d_m)
                d_ls.append(d_l)
                ratio = max(d_l - d_m, 0.0) / (1.0 - d_m + args.delta)
                ratios_global.append(ratio)
                ratios_by_depth.setdefault(rec["depth"], []).append(ratio)
        per_horizon[m] = {
            "spearman_dm_dl": spearman(d_ms, d_ls),
            "pearson_dm_dl": pearson(d_ms, d_ls),
            "mean_gap_dl_minus_dm": (
                sum(l - s for l, s in zip(d_ls, d_ms)) / len(d_ms) if d_ms else None
            ),
            # RQ3: eps_tail = Q_{1-alpha}({r_ij}) global and per depth.
            "eps_tail_quantiles": {
                str(q): quantile(ratios_global, q) for q in args.quantiles
            },
            "eps_tail_by_depth": {
                str(d): {str(q): quantile(rs, q) for q in args.quantiles}
                for d, rs in sorted(ratios_by_depth.items())
            },
            # Empirical coverage of D_L <= D_m + (1-D_m)*eps_tail at the main quantile.
            "coverage_at_main_quantile": _coverage(
                d_ms, d_ls, quantile(ratios_global, args.quantiles[0]), args.delta
            ),
        }
    summary["per_horizon"] = {str(m): v for m, v in per_horizon.items()}

    graded = [r for r in records if "sigma2_oracle" in r]
    if graded:
        m_star = max(horizons)
        cs_list, sig_list = [], []
        for rec in graded:
            pair_tvs = {tuple(pr["pair"]): pr["d_m"][m_star] for pr in rec["pairs"]}
            n = rec["k0"]
            eps_tail = args.assumed_eps_tail
            total = 0.0
            for tv in pair_tvs.values():
                gap = value_gap_bound(tv, r_max=args.r_max, eps_tail=eps_tail)
                total += gap * gap
            cs = total / (n * n)
            rec["c_s"] = cs
            cs_list.append(cs)
            sig_list.append(rec["sigma2_oracle"])
        summary["rq4"] = {
            "horizon_used": m_star,
            "num_nodes": len(graded),
            "spearman_cs_sigma2": spearman(cs_list, sig_list),
            "pearson_cs_sigma2": pearson(cs_list, sig_list),
            "bound_ratio_mean": (
                sum(c / (s + args.delta) for c, s in zip(cs_list, sig_list)) / len(cs_list)
            ),
            "bound_violation_rate": (
                sum(1 for c, s in zip(cs_list, sig_list) if c < s) / len(cs_list)
            ),
        }
        summary["direction_d"] = _allocation_regret(graded, args)
    summary["direction_b"] = _adaptive_lookahead(records, horizons, args)
    return summary


def _adaptive_lookahead(
    records: List[Dict[str, Any]], horizons: List[int], args: argparse.Namespace
) -> Dict[str, Any]:
    """Direction B: pick the first horizon where |D_next - D_m| <= eta.

    Reported as an approximation/ablation only (Summary.md Direction B) — a
    stabilized short-horizon estimate cannot exclude very late divergence, so
    the residual D_L - D_m at the adaptive horizon is reported alongside.
    """

    eta = float(args.stabilize_eta)
    adaptive_ms: List[float] = []
    residuals: List[float] = []
    histogram: Dict[str, int] = {}
    stabilized = 0
    total = 0
    for rec in records:
        for pr in rec["pairs"]:
            total += 1
            chosen = None
            for m, m_next in zip(horizons, horizons[1:]):
                if abs(pr["d_m"][m_next] - pr["d_m"][m]) <= eta:
                    chosen = m
                    stabilized += 1
                    break
            horizon = chosen if chosen is not None else horizons[-1]
            adaptive_ms.append(float(horizon))
            histogram[str(horizon)] = histogram.get(str(horizon), 0) + 1
            residuals.append(max(pr["d_l"] - pr["d_m"][horizon], 0.0))
    return {
        "eta": eta,
        "num_pairs": total,
        "stabilized_fraction": stabilized / total if total else None,
        "adaptive_horizon_mean": (
            sum(adaptive_ms) / len(adaptive_ms) if adaptive_ms else None
        ),
        "adaptive_horizon_histogram": histogram,
        "residual_dl_minus_dm_mean": (
            sum(residuals) / len(residuals) if residuals else None
        ),
        "residual_dl_minus_dm_quantiles": {
            str(q): quantile(residuals, q) for q in args.quantiles
        },
        "note": (
            "approximation/ablation only: stabilization across horizons cannot "
            "exclude very late divergence (Summary.md Direction B)"
        ),
    }


def _coverage(
    d_ms: List[float], d_ls: List[float], eps_tail: Optional[float], delta: float
) -> Optional[float]:
    if eps_tail is None or not d_ms:
        return None
    covered = sum(
        1 for m, l in zip(d_ms, d_ls) if l <= m + (1.0 - m) * eps_tail + delta
    )
    return covered / len(d_ms)


def _allocation_regret(graded: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """Direction D: J(k) = sum_s sigma2_oracle_s / k_s under one shared budget."""

    budget = args.default_bf * len(graded)

    def j_of(allocs: Dict[str, int]) -> float:
        total = 0.0
        for idx, rec in enumerate(graded):
            k = max(allocs.get(f"node_{idx}", 0), 1e-9)
            total += rec["sigma2_oracle"] / k
        return total

    def alloc(weight_field: Optional[str]) -> Dict[str, int]:
        nodes = []
        for idx, rec in enumerate(graded):
            node = {
                "id": f"node_{idx}",
                "vdra_default_k": args.default_bf,
                "vdra_predicted_k": budget,
            }
            if weight_field is None:
                node["w"] = 1.0
            else:
                # sqrt handled by allocate for vdra_dispersion_C; explicit for oracle.
                node["w"] = math.sqrt(max(rec[weight_field], 0.0))
            nodes.append(node)
        summary = allocate_branch_factors(
            nodes,
            total_budget=budget,
            n_min=args.n_min,
            weight_key="w",
        )
        return summary.allocations

    j_uniform = j_of(alloc(None))
    j_vdra = j_of(alloc("c_s"))
    j_oracle = j_of(alloc("sigma2_oracle"))
    return {
        "budget": budget,
        "J_uniform": j_uniform,
        "J_vdra": j_vdra,
        "J_oracle": j_oracle,
        "regret_vdra": j_vdra - j_oracle,
        "regret_uniform": j_uniform - j_oracle,
    }


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def build_metadata(args: argparse.Namespace, selected_runtime_horizon: int) -> Dict[str, Any]:
    return {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "pilot_branch_factor": args.k0,
        "likelihood_samples_per_distribution": args.r,
        "first_phase_tokens": args.first_phase_tokens,
        "short_horizon": selected_runtime_horizon,
        "full_horizon": args.full_tokens,
        "quantile": args.quantile,
        "seed": args.seed,
    }


def selected_runtime_horizon(args: argparse.Namespace) -> int:
    short_horizon = int(args.short_horizon if args.short_horizon is not None else args.first_phase_tokens)
    if short_horizon not in args.horizons:
        args.horizons = sorted(set(list(args.horizons) + [short_horizon]))
    return short_horizon


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-file", required=True, help="jsonl with prompt/answer fields")
    ap.add_argument("--prompt-field", default="problem")
    ap.add_argument("--answer-field", default="answer")
    ap.add_argument("--answer-prefix", default="# Answer\n")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--k0", type=int, default=4, help="pilot children per node")
    ap.add_argument("--r", type=int, default=2, help="continuations per pilot child")
    ap.add_argument("--horizons", default="8,16,32,60")
    ap.add_argument("--full-tokens", type=int, default=512, help="full-continuation length for D_L")
    ap.add_argument("--first-phase-tokens", type=int, default=60)
    ap.add_argument("--short-horizon", type=int, default=None, help="runtime TV second-phase horizon to validate/load")
    ap.add_argument("--segment-tokens", type=int, default=100, help="segment length when walking depths")
    ap.add_argument("--depths", default="0", help="comma list of node depths to calibrate")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--quantiles", default="0.9,0.95,0.99")
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument(
        "--stabilize-eta", type=float, default=0.02,
        help="Direction B: |D_next - D_m| threshold for the adaptive lookahead report",
    )
    ap.add_argument("--grade", action="store_true", help="run RQ4/Direction D (needs answers)")
    ap.add_argument("--assumed-eps-tail", type=float, default=0.0, help="eps_tail used when building C_s for RQ4")
    ap.add_argument("--r-max", type=float, default=1.0)
    ap.add_argument("--default-bf", type=int, default=6)
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.horizons = [int(x) for x in str(args.horizons).split(",") if x]
    args.depths = [int(x) for x in str(args.depths).split(",") if x != ""]
    args.quantiles = [float(x) for x in str(args.quantiles).split(",") if x]
    selected_runtime_horizon(args)
    return args


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

    # Imported lazily so the aggregation helpers stay importable without httpx.
    from treetune.gear.vllm_scorer import VLLMLogprobClient

    client = VLLMLogprobClient(api_base=args.api_base, model=args.model)
    sampler = Sampler(client, temperature=args.temperature)
    records: List[Dict[str, Any]] = []
    try:
        for row_idx, row in enumerate(rows):
            prompt = str(row[args.prompt_field])
            answer = row.get(args.answer_field)
            for depth in args.depths:
                rec = await build_node_record(
                    sampler,
                    prompt=prompt,
                    answer=answer,
                    depth=depth,
                    args=args,
                    rng=rng,
                )
                if rec is not None:
                    rec["prompt_index"] = row_idx
                    records.append(rec)
                print(
                    f"[calibrate] node {len(records)}: prompt={row_idx} depth={depth}",
                    file=sys.stderr,
                )
    finally:
        await client.aclose()

    summary = summarize(records, args)
    runtime_horizon = selected_runtime_horizon(args)
    out = {
        "metadata": build_metadata(args, runtime_horizon),
        "args": {k: v for k, v in vars(args).items()},
        "summary": summary,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\n[calibrate] wrote {args.out}", file=sys.stderr)


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
