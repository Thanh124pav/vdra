"""Tree-agnostic pruning diagnostics for GEAR-style k algorithms."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from treetune.gear.budget_allocation import reward_variance_from_pair_tvs
from treetune.gear.thresholds import ThresholdConfig, tv_to_value_bound


PairKey = Tuple[int, int]


def _pair_key_to_str(pair: PairKey) -> str:
    return f"{pair[0]},{pair[1]}"


def _pair_key_from_any(value: Any) -> Optional[PairKey]:
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        cleaned = value.replace("(", "").replace(")", "").replace(" ", "")
        sep = "," if "," in cleaned else "-"
        parts = cleaned.split(sep)
        if len(parts) == 2 and all(part.lstrip("-").isdigit() for part in parts):
            return int(parts[0]), int(parts[1])
    return None


def normalize_pair_tvs(raw: Mapping[Any, Any]) -> Dict[PairKey, float]:
    out: Dict[PairKey, float] = {}
    for key, value in raw.items():
        pair = _pair_key_from_any(key)
        if pair is None:
            continue
        out[pair] = float(value)
    return out


@dataclass
class PruningTraceRecord:
    node_id: str
    depth: int
    default_branch_factor: int
    predicted_k: int
    pair: Optional[str]
    p_x: Optional[List[float]]
    p_y: Optional[List[float]]
    value_gap: Optional[float]
    tv: Optional[float]
    value_upper_bound: Optional[float]
    reward_variance: Optional[float]
    sigma4: Optional[float]
    duplicate: Optional[bool]
    prune_candidate: Optional[bool]
    keep: Optional[bool]
    unavailable_fields: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_unavailable_record(
    *,
    node_id: str,
    depth: int,
    default_branch_factor: int,
    predicted_k: int,
    unavailable_fields: Sequence[str],
) -> PruningTraceRecord:
    return PruningTraceRecord(
        node_id=node_id,
        depth=int(depth),
        default_branch_factor=int(default_branch_factor),
        predicted_k=int(predicted_k),
        pair=None,
        p_x=None,
        p_y=None,
        value_gap=None,
        tv=None,
        value_upper_bound=None,
        reward_variance=None,
        sigma4=None,
        duplicate=None,
        prune_candidate=None,
        keep=None,
        unavailable_fields=list(unavailable_fields),
    )


def trace_records_from_matrices(
    *,
    node_id: str,
    depth: int,
    default_branch_factor: int,
    predicted_k: int,
    prob_matrix: Sequence[Sequence[float]],
    pair_tvs: Mapping[Any, Any],
    threshold_cfg: ThresholdConfig,
    duplicate_tv_threshold: Optional[float] = None,
    value_gaps: Optional[Mapping[Any, Any]] = None,
    reward_variance: Optional[float] = None,
) -> List[PruningTraceRecord]:
    normalized_pairs = normalize_pair_tvs(pair_tvs)
    duplicate_tv_threshold = (
        float(duplicate_tv_threshold)
        if duplicate_tv_threshold is not None
        else float(threshold_cfg.epsilon)
    )
    if reward_variance is None:
        reward_variance = reward_variance_from_pair_tvs(
            normalized_pairs, n=len(prob_matrix), gamma=threshold_cfg.gamma
        )
    sigma4 = float(reward_variance) * float(reward_variance)
    normalized_value_gaps = normalize_pair_tvs(value_gaps or {})
    records: List[PruningTraceRecord] = []
    for pair, tv in sorted(normalized_pairs.items()):
        i, j = pair
        p_x = list(prob_matrix[i]) if i < len(prob_matrix) else None
        p_y = list(prob_matrix[j]) if j < len(prob_matrix) else None
        value_gap = normalized_value_gaps.get(pair)
        upper_bound = tv_to_value_bound(float(tv), threshold_cfg)
        duplicate = float(tv) < duplicate_tv_threshold
        prune_candidate = (
            bool(value_gap is not None and float(value_gap) <= upper_bound)
            if value_gap is not None
            else duplicate
        )
        records.append(
            PruningTraceRecord(
                node_id=node_id,
                depth=int(depth),
                default_branch_factor=int(default_branch_factor),
                predicted_k=int(predicted_k),
                pair=_pair_key_to_str(pair),
                p_x=p_x,
                p_y=p_y,
                value_gap=float(value_gap) if value_gap is not None else None,
                tv=float(tv),
                value_upper_bound=float(upper_bound),
                reward_variance=float(reward_variance),
                sigma4=float(sigma4),
                duplicate=bool(duplicate),
                prune_candidate=bool(prune_candidate),
                keep=not bool(prune_candidate),
                unavailable_fields=[],
            )
        )
    return records


def records_from_analysis_node(
    node: Mapping[str, Any],
    *,
    threshold_cfg: ThresholdConfig,
) -> List[PruningTraceRecord]:
    evidence = (
        node.get("prunning_analysis")
        or node.get("pruning_analysis")
        or node.get("gear_prunning_analysis")
        or {}
    )
    prob_matrix = evidence.get("prob_matrix") or node.get("gear_prob_matrix")
    pair_tvs = evidence.get("pair_tvs") or node.get("gear_pair_tvs")
    node_id = str(node.get("gear_segment_id") or node.get("segment_id") or "root")
    depth = int(node.get("gear_depth", node.get("depth", 0)) or 0)
    default_branch_factor = int(
        evidence.get(
            "default_branch_factor",
            node.get("gear_default_branch_factor", len(node.get("children") or [])),
        )
        or 0
    )
    predicted_k = int(
        evidence.get(
            "predicted_k",
            node.get("gear_predicted_k", default_branch_factor),
        )
        or 0
    )
    if prob_matrix is None or pair_tvs is None:
        missing = []
        if prob_matrix is None:
            missing.append("prob_matrix")
        if pair_tvs is None:
            missing.append("pair_tvs")
        return [
            make_unavailable_record(
                node_id=node_id,
                depth=depth,
                default_branch_factor=default_branch_factor,
                predicted_k=predicted_k,
                unavailable_fields=missing,
            )
        ]
    return trace_records_from_matrices(
        node_id=node_id,
        depth=depth,
        default_branch_factor=default_branch_factor,
        predicted_k=predicted_k,
        prob_matrix=prob_matrix,
        pair_tvs=pair_tvs,
        threshold_cfg=threshold_cfg,
        duplicate_tv_threshold=evidence.get("duplicate_tv_threshold"),
        value_gaps=evidence.get("value_gaps"),
        reward_variance=evidence.get("reward_variance", node.get("gear_reward_variance")),
    )


def summarize_records(records: Iterable[PruningTraceRecord]) -> Dict[str, Any]:
    records = list(records)
    available = [rec for rec in records if not rec.unavailable_fields]
    pruned = [rec for rec in available if rec.prune_candidate]
    duplicates = [rec for rec in available if rec.duplicate]
    tvs = [rec.tv for rec in available if rec.tv is not None]
    variances = [
        rec.reward_variance for rec in available if rec.reward_variance is not None
    ]
    return {
        "num_records": len(records),
        "num_available_records": len(available),
        "num_unavailable_records": len(records) - len(available),
        "num_prune_candidates": len(pruned),
        "num_duplicate_pairs": len(duplicates),
        "prune_candidate_rate": len(pruned) / len(available) if available else 0.0,
        "duplicate_rate": len(duplicates) / len(available) if available else 0.0,
        "tv_mean": sum(tvs) / len(tvs) if tvs else None,
        "tv_max": max(tvs) if tvs else None,
        "reward_variance_mean": (
            sum(variances) / len(variances) if variances else None
        ),
        "reward_variance_max": max(variances) if variances else None,
    }


def format_trace_record(record: PruningTraceRecord) -> str:
    first = (
        f"[prunning] node={record.node_id} depth={record.depth} "
        f"default_b={record.default_branch_factor} predicted_k={record.predicted_k}"
    )
    if record.unavailable_fields:
        return (
            first
            + "\n"
            + f"[prunning] unavailable_fields={','.join(record.unavailable_fields)}"
        )
    return "\n".join(
        [
            first,
            f"[prunning] pair=({record.pair})",
            f"[prunning] p_x={json.dumps(record.p_x)}",
            f"[prunning] p_y={json.dumps(record.p_y)}",
            f"[prunning] |v_x-v_y|={record.value_gap}",
            f"[prunning] tv={record.tv}",
            f"[prunning] value_upper_bound={record.value_upper_bound}",
            f"[prunning] reward_variance_sigma2={record.reward_variance}",
            f"[prunning] sigma4={record.sigma4}",
            (
                f"[prunning] duplicate={record.duplicate} "
                f"prune_candidate={record.prune_candidate} keep={record.keep}"
            ),
        ]
    )
