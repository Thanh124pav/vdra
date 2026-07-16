#!/usr/bin/env python
"""RQ6: does better value estimation improve segment-gradient quality?

Offline, self-contained (HF causal LM, CPU-friendly with a small model):

  1. per prompt, sample K segment children s_1..s_K (M tokens each) from the
     model — these are the internal nodes whose values must be estimated;
  2. per child, sample ``--n-ref`` full continuations and grade them; the pool
     mean is V_ref(s_i) and the reference segment gradient is

         g_ref = sum_i (V_ref(s_i) - b) * grad log pi(s_i | prompt);

  3. estimate the child dispersion bound C_s from the §9 tanh TV between the
     children's own continuation blocks (scored by the same model), allocate
     the MC budget (``default_bf x K``) per method (uniform / vdra / random /
     empirical_variance / oracle), simulate V_hat by subsampling each pool,
     and compare

         cos(g_hat, g_ref),  ||g_hat - g_ref||^2,  Var_seeds[g_hat]

     per method (Summary.md RQ6).

Example (tiny model, CPU):
    python scripts/eval_gradient_quality.py \
        --hf-model sshleifer/tiny-gpt2 --prompts-file data/math_train.jsonl \
        --num-prompts 4 --children 4 --n-ref 8 --seeds 16 \
        --out results/gradient_quality.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vdra_core import value_gap_bound  # noqa: E402

_RQ5_SPEC = importlib.util.spec_from_file_location(
    "_vdra_rq5", Path(__file__).resolve().parent / "eval_value_mse.py"
)
_rq5 = importlib.util.module_from_spec(_RQ5_SPEC)
sys.modules.setdefault("_vdra_rq5", _rq5)
_RQ5_SPEC.loader.exec_module(_rq5)

METHODS = _rq5.METHODS


# --------------------------------------------------------------------------- #
# Pure helpers (CPU-testable without torch).
# --------------------------------------------------------------------------- #
def flat_cosine(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return None
    return dot / (na * nb)


def l2_sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def tanh_tv(logps_i: Sequence[float], logps_j: Sequence[float]) -> float:
    vals = [
        abs(math.tanh((x - y) / 2.0))
        for x, y in zip(logps_i, logps_j)
        if math.isfinite(x) and math.isfinite(y)
    ]
    return sum(vals) / len(vals) if vals else 0.0


def subsample_value(pool: Sequence[float], k: int, rng: random.Random) -> float:
    pool = [float(v) for v in pool]
    k = max(int(k), 1)
    if k <= len(pool):
        sample = rng.sample(pool, k)
    else:
        sample = [rng.choice(pool) for _ in range(k)]
    return sum(sample) / len(sample)


# --------------------------------------------------------------------------- #
# HF model plumbing.
# --------------------------------------------------------------------------- #
def _load_model(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    torch.manual_seed(0)
    return model, tok


def _generate(model, tok, prompt_ids: List[int], max_new_tokens: int, temperature: float) -> List[int]:
    import torch

    with torch.no_grad():
        out = model.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            pad_token_id=tok.pad_token_id,
        )
    return out[0].tolist()[len(prompt_ids):]


def _sequence_logprob(model, prefix_ids: List[int], cont_ids: List[int], *, grad: bool):
    """log pi(cont | prefix): sum of chosen-token logprobs (differentiable)."""

    import torch

    ids = torch.tensor([list(prefix_ids) + list(cont_ids)])
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(ids).logits[0]
        logps = torch.log_softmax(logits, dim=-1)
        start = len(prefix_ids)
        total = logps[
            torch.arange(start - 1, start - 1 + len(cont_ids)),
            torch.tensor(list(cont_ids)),
        ].sum()
    return total


def _gradient_for_values(
    model, prompt_ids: List[int], children_ids: List[List[int]], values: List[float]
):
    """g = sum_i (V_i - mean V) * grad log pi(s_i | prompt), flattened."""

    import torch

    model.zero_grad(set_to_none=True)
    baseline = sum(values) / len(values)
    loss = None
    for child_ids, value in zip(children_ids, values):
        weight = float(value) - baseline
        term = weight * _sequence_logprob(model, prompt_ids, child_ids, grad=True)
        loss = term if loss is None else loss + term
    loss.backward()
    flat = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in model.parameters()
        ]
    )
    model.zero_grad(set_to_none=True)
    return flat


def estimate_child_dispersion(
    model,
    child_full_ids: List[int],
    continuation_ids: List[List[int]],
    *,
    k0: int,
    first_phase_tokens: int,
    short_horizon: int,
    r_max: float,
    eps_tail: float,
) -> float:
    """C_s for one child from its own continuations (§9, no extra sampling).

    Pilot prefixes are the first ``first_phase_tokens`` of ``k0`` continuations;
    each pilot's scored block is its own next ``short_horizon`` tokens, scored
    under every pilot prefix by the same model.
    """

    pilots = [c[:first_phase_tokens] for c in continuation_ids[:k0] if len(c) > first_phase_tokens]
    blocks = [
        c[first_phase_tokens : first_phase_tokens + short_horizon]
        for c in continuation_ids[:k0]
        if len(c) > first_phase_tokens
    ]
    if len(pilots) < 2:
        return 0.0
    logp = [
        [
            float(
                _sequence_logprob(
                    model, child_full_ids + pilots[i], blocks[b], grad=False
                )
            )
            for b in range(len(blocks))
        ]
        for i in range(len(pilots))
    ]
    total = 0.0
    for i in range(len(pilots)):
        for j in range(i + 1, len(pilots)):
            cols = [i, j]
            tv = tanh_tv([logp[i][c] for c in cols], [logp[j][c] for c in cols])
            gap = value_gap_bound(tv, r_max=r_max, eps_tail=eps_tail)
            total += gap * gap
    return total / (len(pilots) * len(pilots))


# --------------------------------------------------------------------------- #
# Main experiment.
# --------------------------------------------------------------------------- #
def evaluate_prompt(
    model,
    tok,
    *,
    prompt: str,
    answer: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    prompt_ids = tok.encode(prompt)
    children_ids: List[List[int]] = []
    records: List[Dict[str, Any]] = []
    all_conts: List[List[List[int]]] = []
    for child_idx in range(args.children):
        child_ids = _generate(model, tok, prompt_ids, args.segment_tokens, args.temperature)
        if not child_ids:
            continue
        child_full = prompt_ids + child_ids
        conts = [
            _generate(model, tok, child_full, args.full_tokens, args.temperature)
            for _ in range(args.n_ref)
        ]
        rewards = [
            _rq5._cal.simple_math_grade(
                tok.decode(child_full + cont), answer, args.answer_prefix
            )
            for cont in conts
        ]
        c_s = estimate_child_dispersion(
            model,
            child_full,
            conts,
            k0=args.k0,
            first_phase_tokens=args.first_phase_tokens,
            short_horizon=args.short_horizon,
            r_max=args.r_max,
            eps_tail=args.assumed_eps_tail,
        )
        children_ids.append(child_ids)
        all_conts.append(conts)
        records.append(
            {"node_id": f"c{child_idx}", "rewards": rewards, "c_s": c_s}
        )
    if len(children_ids) < 2:
        return None

    v_ref = [sum(r["rewards"]) / len(r["rewards"]) for r in records]
    g_ref = _gradient_for_values(model, prompt_ids, children_ids, v_ref)
    g_ref_list = g_ref.tolist()

    budget = args.default_bf * len(records)
    out: Dict[str, Any] = {"per_method": {}, "num_children": len(records)}
    for method in METHODS:
        weights = {
            rec["node_id"]: _rq5.method_weight(method, rec, k0=args.k0)
            for rec in records
        }
        allocations = _rq5.allocate_by_weights(
            weights, budget=budget, n_min=args.n_min
        )
        cosines: List[float] = []
        l2s: List[float] = []
        grad_sum = None
        grad_sq_sum = 0.0
        for seed in range(args.seeds):
            rng = random.Random(f"rq6:{method}:{seed}")
            values = [
                subsample_value(rec["rewards"], allocations[rec["node_id"]], rng)
                for rec in records
            ]
            g_hat = _gradient_for_values(model, prompt_ids, children_ids, values)
            g_list = g_hat.tolist()
            cos = flat_cosine(g_list, g_ref_list)
            if cos is not None:
                cosines.append(cos)
            l2s.append(l2_sq(g_list, g_ref_list))
            grad_sum = g_hat if grad_sum is None else grad_sum + g_hat
            grad_sq_sum += float((g_hat * g_hat).sum())
        n = max(args.seeds, 1)
        mean_norm_sq = float((grad_sum / n).pow(2).sum()) if grad_sum is not None else 0.0
        out["per_method"][method] = {
            "cos_mean": sum(cosines) / len(cosines) if cosines else None,
            "l2_mean": sum(l2s) / len(l2s) if l2s else None,
            # E||g||^2 - ||E g||^2: gradient variability across rollout seeds.
            "grad_variance": grad_sq_sum / n - mean_norm_sq,
            "allocations": allocations,
        }
    return out


def aggregate(per_prompt: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_prompts": len(per_prompt), "per_method": {}}
    for method in METHODS:
        cos = [p["per_method"][method]["cos_mean"] for p in per_prompt if p["per_method"][method]["cos_mean"] is not None]
        l2 = [p["per_method"][method]["l2_mean"] for p in per_prompt]
        var = [p["per_method"][method]["grad_variance"] for p in per_prompt]
        summary["per_method"][method] = {
            "cos_mean": sum(cos) / len(cos) if cos else None,
            "l2_mean": sum(l2) / len(l2) if l2 else None,
            "grad_variance_mean": sum(var) / len(var) if var else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = []
    with open(args.prompts_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng.shuffle(rows)
    rows = rows[: args.num_prompts]

    model, tok = _load_model(args.hf_model)
    per_prompt: List[Dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt") or row.get("problem") or row.get("question")
        answer = row.get("answer") or row.get("solution")
        if not prompt or answer is None:
            continue
        result = evaluate_prompt(
            model, tok, prompt=str(prompt), answer=str(answer), args=args
        )
        if result is not None:
            per_prompt.append(result)
            print(f"[rq6] prompt done ({len(per_prompt)}/{len(rows)})", flush=True)

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hf_model": args.hf_model,
            "children": args.children,
            "n_ref": args.n_ref,
            "default_bf": args.default_bf,
            "seeds": args.seeds,
            "seed": args.seed,
        },
        "summary": aggregate(per_prompt),
        "per_prompt": per_prompt,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[rq6] wrote {args.out}")
    else:
        print(text)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-model", required=True, help="small HF causal LM")
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--answer-prefix", default="# Answer\n")
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--children", type=int, default=4, help="segment children per prompt")
    ap.add_argument("--n-ref", type=int, default=8, help="reference rollouts per child")
    ap.add_argument("--segment-tokens", type=int, default=32)
    ap.add_argument("--full-tokens", type=int, default=128)
    ap.add_argument("--k0", type=int, default=4)
    ap.add_argument("--first-phase-tokens", type=int, default=16)
    ap.add_argument("--short-horizon", type=int, default=16)
    ap.add_argument("--default-bf", type=int, default=4, help="MC rollouts per child under uniform")
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--assumed-eps-tail", type=float, default=0.0)
    ap.add_argument("--r-max", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    main()
