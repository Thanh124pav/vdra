"""Offline diagnostic helpers for rollout-vs-actor log-prob parity.

The runtime-heavy part of this diagnostic is producing records with identical
prompt/response token ids and two log-prob arrays: one from the rollout server
and one from the actor. This module owns the deterministic comparison and CLI
so the validation fails loudly with max/mean deltas instead of hiding drift
behind a large tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_REQUIRED_METADATA = (
    "model",
    "tokenizer",
    "prompt_token_ids",
    "response_token_ids",
    "temperature",
)


def compare_logprob_arrays(
    rollout_logprobs: Sequence[float],
    actor_logprobs: Sequence[float],
    *,
    atol: float,
) -> dict[str, float]:
    if len(rollout_logprobs) != len(actor_logprobs):
        raise ValueError(
            f"logprob length mismatch: rollout={len(rollout_logprobs)} actor={len(actor_logprobs)}"
        )
    if not rollout_logprobs:
        raise ValueError("empty logprob arrays cannot validate parity")
    deltas = [abs(float(a) - float(b)) for a, b in zip(rollout_logprobs, actor_logprobs)]
    max_delta = max(deltas)
    mean_delta = sum(deltas) / len(deltas)
    if not math.isfinite(max_delta) or not math.isfinite(mean_delta):
        raise ValueError("non-finite logprob parity delta")
    if max_delta > float(atol):
        raise AssertionError(
            f"logprob parity failed: max_delta={max_delta:.6g} mean_delta={mean_delta:.6g} atol={atol:.6g}"
        )
    return {"max_delta": float(max_delta), "mean_delta": float(mean_delta), "num_tokens": float(len(deltas))}


def validate_parity_record(record: Mapping[str, object], *, atol: float) -> dict[str, float]:
    missing = [key for key in _REQUIRED_METADATA if key not in record]
    if missing:
        raise ValueError(f"parity record missing required metadata: {missing}")
    rollout = record.get("rollout_logprobs")
    actor = record.get("actor_logprobs")
    if not isinstance(rollout, list) or not isinstance(actor, list):
        raise ValueError("record must contain rollout_logprobs and actor_logprobs lists")
    response_ids = record.get("response_token_ids")
    if not isinstance(response_ids, list) or len(response_ids) != len(rollout):
        raise ValueError("response_token_ids must align one-to-one with rollout_logprobs")
    return compare_logprob_arrays(rollout, actor, atol=atol)


def validate_jsonl(path: str | Path, *, atol: float) -> dict[str, float]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = validate_parity_record(json.loads(line), atol=atol)
        except Exception as exc:
            raise RuntimeError(f"logprob parity record {line_no} failed") from exc
        rows.append(row)
    if not rows:
        raise ValueError("no parity records found")
    return {
        "records": float(len(rows)),
        "max_delta": max(row["max_delta"] for row in rows),
        "mean_delta": sum(row["mean_delta"] for row in rows) / len(rows),
        "num_tokens": sum(row["num_tokens"] for row in rows),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate rollout-vs-actor log-prob parity records")
    parser.add_argument("records_jsonl")
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(validate_jsonl(args.records_jsonl, atol=args.atol), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
