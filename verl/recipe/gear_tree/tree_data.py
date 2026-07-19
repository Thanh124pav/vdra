"""Assemble tree edges into a verl ``DataProto`` training batch.

Each SPO/GEAR edge becomes one training row: ``query = parent trajectory tokens``
(left-padded prompt) and ``response = this segment's generated tokens``
(right-padded), following verl's standard layout (see
``vLLMRollout.generate_sequences``). Per-token advantages, old log-probs, values
and returns are broadcast from the edge scalars by
``tree_advantage.token_fields_for_edges`` so the numbers stay identical to
treetune's per-token broadcast.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

from recipe.gear_tree.tree_advantage import token_fields_for_edges


# PLAN.md P0.N4: deterministic string -> int64 mapping for group tensors.
# blake2b keeps collision-probability negligible while staying reproducible
# across processes and container restarts. Signed int64 so torch tensors of
# dtype int64 hold the whole range.
_ID_MASK = (1 << 63) - 1


def _stable_int_id(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) & _ID_MASK


def group_tensors_for_edges(edges: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """PLAN.md P0.N4 + P0.2: build row-level group tensors from tree edges.

    Returns int64 tensors ``tree_group_ids``, ``parent_group_ids``,
    ``queue_group_ids``, ``allocated_k``, ``tree_total_segment_count`` and
    float32 ``sample_multiplicity``, all shaped ``[batch]``. Missing metadata
    falls back to safe defaults (id 0, allocated_k = 1, multiplicity = 1,
    tree_total_segment_count = 1) so legacy edges that were generated before
    this migration still tensorize; strict main runs must additionally check
    the group-integrity invariants below.
    """
    bsz = len(edges)
    tree_ids = torch.empty(bsz, dtype=torch.int64)
    parent_ids = torch.empty(bsz, dtype=torch.int64)
    queue_ids = torch.empty(bsz, dtype=torch.int64)
    allocated = torch.empty(bsz, dtype=torch.int64)
    multiplicities = torch.empty(bsz, dtype=torch.float32)
    tree_total = torch.empty(bsz, dtype=torch.int64)
    for row, edge in enumerate(edges):
        tree_ids[row] = _stable_int_id(edge.get("tree_id"))
        parent_ids[row] = _stable_int_id(edge.get("parent_group_id"))
        queue_ids[row] = _stable_int_id(edge.get("queue_flush_id", 0))
        allocated[row] = int(edge.get("allocated_k", 1) or 1)
        multiplicities[row] = float(edge.get("sample_multiplicity", 1) or 1)
        # PLAN.md P0.2: read the pre-filter N_seg(T) stamped by
        # extract_edges_from_tree. Fall back to the tree_summary or the
        # retained row count so legacy edges keep loading.
        raw_total = edge.get("tree_total_segment_count")
        if raw_total is None:
            summary = edge.get("tree_summary") or {}
            raw_total = summary.get("tree_total_segment_count")
        try:
            total_int = int(raw_total) if raw_total is not None else 0
        except (TypeError, ValueError):
            total_int = 0
        tree_total[row] = max(total_int, 1)
    return {
        "tree_group_ids": tree_ids,
        "parent_group_ids": parent_ids,
        "queue_group_ids": queue_ids,
        "allocated_k": allocated,
        "sample_multiplicity": multiplicities,
        "tree_total_segment_count": tree_total,
    }


def compute_segment_objective_weights(edges: Sequence[Dict[str, Any]]) -> List[float]:
    """PLAN.md P0.4: precompute the segment-average weight for every row.

    For every retained segment ``s`` in tree ``T`` in a batch of ``N_T`` trees::

        w_s = 1 / (N_T * N_seg(T))

    where ``N_seg(T)`` is the pre-filter ``tree_total_segment_count`` stamped
    by ``extract_edges_from_tree``. The full loss is then::

        L = sum_row w_row * L_row

    which reproduces exactly ``(1/N_T) sum_T (1/N_seg(T)) sum_s L_s`` and is
    invariant under mini/microbatch splits (the weights partition, not the
    denominator). Passing ``segment_token_reduction`` changes only how ``L_row``
    is computed inside the loss; the weight is the same for both modes.
    """
    if not edges:
        return []
    tree_ids: set[str] = set()
    for edge in edges:
        tree_ids.add(str(edge.get("tree_id", "")))
    n_tree = max(len(tree_ids), 1)
    weights: List[float] = []
    for edge in edges:
        raw_total = edge.get("tree_total_segment_count")
        if raw_total is None:
            summary = edge.get("tree_summary") or {}
            raw_total = summary.get("tree_total_segment_count")
        try:
            total_int = int(raw_total) if raw_total is not None else 0
        except (TypeError, ValueError):
            total_int = 0
        n_seg = max(total_int, 1)
        weights.append(1.0 / (n_tree * n_seg))
    return weights


def validate_segment_objective_weights(
    edges: Sequence[Dict[str, Any]],
    weights: Sequence[float],
    *,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """PLAN.md P0.4: enforce the two normalization invariants for the
    segment-average objective:

        sum_{row in tree T} w_row == (retained_in_T / N_seg(T)) / N_T
        sum_all_rows w_row <= 1

    (Equality holds when every realized segment is retained. Zero-advantage
    filtering strictly shrinks the total mass.) Raises ``ValueError`` on any
    failure and returns a small diagnostics dict.
    """
    from collections import defaultdict

    if len(edges) != len(weights):
        raise ValueError(
            f"segment objective_weights length {len(weights)} != edges length {len(edges)}"
        )
    if not edges:
        return {
            "vdra/segment_weight_sum": 0.0,
            "vdra/segment_weight_tree_count": 0,
        }

    trees: Dict[str, List[int]] = defaultdict(list)
    for row, edge in enumerate(edges):
        trees[str(edge.get("tree_id", ""))].append(row)
    n_tree = len(trees)
    failures: List[str] = []
    total = 0.0
    max_per_tree_err = 0.0
    for tid, rows in trees.items():
        raw_total = edges[rows[0]].get("tree_total_segment_count")
        if raw_total is None:
            summary = edges[rows[0]].get("tree_summary") or {}
            raw_total = summary.get("tree_total_segment_count")
        try:
            total_int = int(raw_total) if raw_total is not None else 0
        except (TypeError, ValueError):
            total_int = 0
        n_seg = max(total_int, 1)
        expected = len(rows) / (n_tree * n_seg)
        got = sum(weights[r] for r in rows)
        if abs(got - expected) > atol:
            failures.append(
                f"tree {tid!r}: sum(w) = {got!r}, expected retained/({n_tree}*{n_seg}) = {expected!r}"
            )
        max_per_tree_err = max(max_per_tree_err, abs(got - expected))
        total += got
    if total - 1.0 > atol:
        failures.append(f"batch sum(w) = {total!r} > 1")
    if failures:
        raise ValueError(
            "segment objective_weights normalization failed (PLAN.md P0.4):\n  "
            + "\n  ".join(failures)
        )
    return {
        "vdra/segment_weight_sum": float(total),
        "vdra/segment_weight_tree_count": int(n_tree),
        "vdra/segment_weight_per_tree_max_err": float(max_per_tree_err),
    }


def compute_objective_weights(edges: Sequence[Dict[str, Any]]) -> List[float]:
    """PLAN.md P0.3: precompute the exact objective weight for every row.

    For every realized child ``j`` of parent ``p`` in tree ``T``:

        w_{p,j} = (1 / N_tree) * (1 / |P(T)|) * (m_{p,j} / sum_j' m_{p,j'})

    where ``N_tree`` is the number of distinct ``tree_id`` in the batch,
    ``|P(T)|`` is the number of distinct realized parent groups in tree
    ``T``, and ``m_{p,j}`` is the child's ``sample_multiplicity`` (``1`` under
    ``fresh_iid``). The returned list is aligned with ``edges`` row-for-row
    and sums to ``1`` over the whole batch.
    """
    from collections import defaultdict

    if not edges:
        return []

    # Group edges by tree, then by parent group.
    trees: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for row, edge in enumerate(edges):
        tid = str(edge.get("tree_id", ""))
        pgid = str(edge.get("parent_group_id", ""))
        trees[tid][pgid].append(row)

    n_tree = len(trees)
    weights = [0.0] * len(edges)
    for tid, parents in trees.items():
        p_count = len(parents)
        for pgid, rows in parents.items():
            mults = [
                max(int(edges[r].get("sample_multiplicity", 1) or 1), 1)
                for r in rows
            ]
            total_m = float(sum(mults))
            for r, m in zip(rows, mults):
                weights[r] = (1.0 / n_tree) * (1.0 / p_count) * (m / total_m)
    return weights


def validate_objective_weights(
    edges: Sequence[Dict[str, Any]],
    weights: Sequence[float],
    *,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """PLAN.md P0.3: enforce the three normalization invariants.

        sum_j local_child_weight[p, j] == 1 for every parent
        sum_p parent_weight[T, p]      == 1 for every tree
        sum_all_rows objective_weights == 1

    Raises ``ValueError`` on any failure. Returns a small diagnostics dict.
    """
    from collections import defaultdict

    if len(edges) != len(weights):
        raise ValueError(
            f"objective_weights length {len(weights)} != edges length {len(edges)}"
        )
    if not edges:
        return {
            "vdra/objective_weight_sum": 0.0,
            "vdra/objective_weight_tree_count": 0,
        }

    trees: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for row, edge in enumerate(edges):
        trees[str(edge.get("tree_id", ""))][str(edge.get("parent_group_id", ""))].append(row)

    n_tree = len(trees)
    failures: List[str] = []
    total = 0.0
    max_parent_err = 0.0
    max_tree_err = 0.0
    for tid, parents in trees.items():
        p_count = len(parents)
        tree_mass = 0.0
        for pgid, rows in parents.items():
            local_sum = sum(weights[r] for r in rows)
            expected_local = 1.0 / (n_tree * p_count)
            if abs(local_sum - expected_local) > atol:
                failures.append(
                    f"parent {pgid!r} in tree {tid!r}: sum(w) = {local_sum!r}, "
                    f"expected 1/(N_tree*|P(T)|) = {expected_local!r}"
                )
            # Local child fractions must sum to 1 per parent.
            if local_sum > 0:
                mults = [max(int(edges[r].get("sample_multiplicity", 1) or 1), 1) for r in rows]
                total_m = float(sum(mults))
                for r, m in zip(rows, mults):
                    local_frac = weights[r] / local_sum
                    if abs(local_frac - (m / total_m)) > atol:
                        max_parent_err = max(
                            max_parent_err, abs(local_frac - (m / total_m))
                        )
            tree_mass += local_sum
        expected_tree = 1.0 / n_tree
        if abs(tree_mass - expected_tree) > atol:
            failures.append(
                f"tree {tid!r}: sum(w) = {tree_mass!r}, expected 1/N_tree = {expected_tree!r}"
            )
        max_tree_err = max(max_tree_err, abs(tree_mass - expected_tree))
        total += tree_mass
    if abs(total - 1.0) > atol:
        failures.append(f"batch sum(w) = {total!r} != 1")
    if failures:
        raise ValueError(
            "objective_weights normalization failed (PLAN.md P0.3):\n  "
            + "\n  ".join(failures)
        )
    return {
        "vdra/objective_weight_sum": float(total),
        "vdra/objective_weight_tree_count": int(n_tree),
        "vdra/objective_weight_parent_max_err": float(max_parent_err),
        "vdra/objective_weight_tree_max_err": float(max_tree_err),
    }


def compute_group_metrics(edges: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """PLAN.md P0.N7 runtime metrics.

    Computes:
      * vdra/parent_groups_per_tree (mean over trees)
      * vdra/children_per_parent_mean / _std
      * vdra/empty_token_mask_children — reported as 0 when no per-token
        length metadata is available; the actor loss also emits the exact
        per-step count.
      * vdra/queue_parent_mass_sum — sum of |Q_r|/|P(T)| over queue partitions
        per tree; a well-formed queue partition sums to 1.
      * vdra/parent_weight_sum_per_tree — always 1 for the canonical
        node-balanced aggregation.
      * vdra/child_weight_sum_per_parent — 1 under fresh_iid; sum(m_j)/sum(m_j)
        under weighted_reuse.
      * vdra/effective_segment_weight_vs_branch_factor_corr — Pearson
        correlation between a segment's effective weight (1/(|P(T)|*k_p) for
        fresh_iid) and its parent's branch factor k_p. For the canonical
        node-balanced loss this correlation must be strongly negative
        (-1 in the simplest single-tree case); a positive correlation flags
        that the legacy edge-mean has crept back in.
    """
    from collections import defaultdict
    import math

    if not edges:
        return {}

    # Group by tree.
    trees: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        trees[str(edge.get("tree_id", ""))].append(edge)

    parent_groups_per_tree: List[int] = []
    children_per_parent: List[int] = []
    queue_parent_mass_sum: List[float] = []
    parent_weight_sum: List[float] = []
    child_weight_sum: List[float] = []
    seg_weights: List[float] = []
    branch_factors: List[float] = []
    empty_mask_children = 0

    for tid, tree_edges in trees.items():
        # parents in this tree
        parents: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in tree_edges:
            parents[str(edge.get("parent_group_id", ""))].append(edge)
            if int(edge.get("response_length", edge.get("num_tokens", 1)) or 0) <= 0:
                empty_mask_children += 1
        parent_groups_per_tree.append(len(parents))
        p_total = max(len(parents), 1)
        parent_weight_sum.append(1.0)  # canonical aggregation always sums to 1

        # child weights per parent (== 1 under fresh_iid)
        for pgid, group in parents.items():
            children_per_parent.append(len(group))
            mults = [max(int(e.get("sample_multiplicity", 1) or 1), 1) for e in group]
            total_mult = float(sum(mults))
            if total_mult > 0:
                child_weight_sum.append(total_mult / total_mult)  # == 1
            k_p = float(len(group))
            for _ in group:
                # segment weight = 1 / (|P(T)| * k_p) under fresh_iid.
                seg_weights.append(1.0 / (p_total * max(k_p, 1.0)))
                branch_factors.append(k_p)

        # queue partition
        queues: Dict[Any, set] = defaultdict(set)
        for edge in tree_edges:
            queues[edge.get("queue_flush_id", 0)].add(str(edge.get("parent_group_id", "")))
        mass = sum(len(pset) / p_total for pset in queues.values())
        queue_parent_mass_sum.append(mass)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(xs: List[float]) -> float:
        if not xs:
            return 0.0
        mu = _mean(xs)
        return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))

    def _pearson(xs: List[float], ys: List[float]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx = _mean(xs)
        my = _mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if dx == 0.0 or dy == 0.0:
            return 0.0
        return num / (dx * dy)

    return {
        "vdra/parent_groups_per_tree": _mean([float(x) for x in parent_groups_per_tree]),
        "vdra/children_per_parent_mean": _mean([float(x) for x in children_per_parent]),
        "vdra/children_per_parent_std": _std([float(x) for x in children_per_parent]),
        "vdra/empty_token_mask_children": float(empty_mask_children),
        "vdra/queue_parent_mass_sum": _mean(queue_parent_mass_sum),
        "vdra/parent_weight_sum_per_tree": _mean(parent_weight_sum),
        "vdra/child_weight_sum_per_parent": _mean(child_weight_sum),
        "vdra/effective_segment_weight_vs_branch_factor_corr": _pearson(
            seg_weights, branch_factors
        ),
        "vdra/trees_in_batch": float(len(trees)),
    }


def validate_group_integrity(
    edges: Sequence[Dict[str, Any]],
    *,
    strict_fresh_iid: bool = True,
) -> Dict[str, Any]:
    """PLAN.md P0.N4: enforce grouping invariants before the actor update.

    Invariants (all failures raise ``ValueError`` when ``strict_fresh_iid``):
      * every row sharing a ``parent_group_id`` shares one ``tree_group_id``;
      * every row sharing a ``parent_group_id`` shares one ``allocated_k``;
      * fresh_iid groups (sample_multiplicity == 1 across every row of the
        group) have row_count == allocated_k;
      * no parent group is silently split or partially dropped.

    Returns a small diagnostics dict for logging.
    """
    from collections import defaultdict

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        pgid = str(edge.get("parent_group_id", ""))
        groups[pgid].append(edge)

    failures: List[str] = []
    fresh_iid_group_count = 0
    weighted_group_count = 0
    for pgid, group in groups.items():
        tree_ids = {str(e.get("tree_id")) for e in group}
        if len(tree_ids) != 1:
            failures.append(f"parent_group_id={pgid!r} spans multiple tree_ids={tree_ids}")
        allocated_values = {int(e.get("allocated_k", 0)) for e in group}
        if len(allocated_values) != 1:
            failures.append(
                f"parent_group_id={pgid!r} has inconsistent allocated_k={allocated_values}"
            )
        mults = [int(e.get("sample_multiplicity", 1) or 1) for e in group]
        if all(m == 1 for m in mults):
            fresh_iid_group_count += 1
            # PLAN.md P0.N4: always detect a partial fresh_iid parent group;
            # ``strict_fresh_iid`` only decides whether to raise.
            expected = next(iter(allocated_values), 0)
            if expected and len(group) != expected:
                failures.append(
                    f"fresh_iid parent_group_id={pgid!r} has {len(group)} rows "
                    f"but allocated_k={expected}"
                )
        else:
            weighted_group_count += 1
    if failures and strict_fresh_iid:
        raise ValueError(
            "Group-integrity check failed (PLAN.md P0.N4):\n  " + "\n  ".join(failures)
        )
    return {
        "vdra/group_integrity_failures": len(failures),
        "vdra/fresh_iid_parent_groups": fresh_iid_group_count,
        "vdra/weighted_reuse_parent_groups": weighted_group_count,
        "vdra/parent_groups_total": len(groups),
    }


def normalize_generated_edges(
    edges: Sequence[Dict[str, Any]],
    *,
    snapshot_id: str,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """PLAN.md P0.H: normalize freshly generated edges and assign edge IDs.

    Strict mode requires the unique tree identity stamped by
    ``make_tree_instance_id`` (snapshot + rollout iteration + question +
    per-tree uuid/counter) plus explicit parent/child identities, and
    derives ``edge_id`` from exactly (tree identity | parent group | child
    segment). Legacy fallback chains (generic ``gear_segment_id`` /
    ``child_index``) survive only in non-strict mode for old fixtures.
    """
    normalized: List[Dict[str, Any]] = []
    for idx, edge in enumerate(edges):
        record = dict(edge)
        tree_id = record.get("tree_instance_id") or record.get("tree_id")
        parent_group = record.get("parent_group_id")
        child_seg = record.get("child_segment_id")
        if strict:
            missing = [
                name
                for name, value in (
                    ("tree_instance_id/tree_id", tree_id),
                    ("parent_group_id", parent_group),
                    ("child_segment_id", child_seg),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Strict VDRA edge identity is incomplete; missing "
                    f"{missing} on generated edge {idx} (PLAN.md P0.H). "
                    "Generation must stamp make_tree_instance_id-derived "
                    "tree identities and explicit parent/child segment ids."
                )
            key = f"{tree_id}|{parent_group}|{child_seg}"
        else:
            tree_id = tree_id or record.get("gear_segment_id", "")
            parent_group = (
                parent_group
                or record.get("parent_path")
                or record.get("gear_parent_segment_id", "")
            )
            child_seg = (
                child_seg
                or record.get("gear_segment_id")
                or str(record.get("child_index", idx))
            )
            qid = record.get("question_id", "")
            key = f"{snapshot_id}|{qid}|{tree_id}|{parent_group}|{child_seg}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        record.setdefault("edge_id", f"{snapshot_id}:{digest}")
        record.setdefault("policy_snapshot_id", snapshot_id)
        if record["policy_snapshot_id"] != snapshot_id:
            raise ValueError(
                "Generated edge policy_snapshot_id mismatches rollout snapshot"
            )
        response = list(record.get("response_token_ids") or [])
        log_probs = record.get("actor_shifted_log_probs")
        if log_probs is None:
            raise ValueError(
                "Generated edge is missing generation-time actor_shifted_log_probs"
            )
        if len(log_probs) != len(response):
            raise ValueError(
                "Generated edge log-probs do not align with response tokens"
            )
        record.setdefault("depth", int(record.get("depth", 0) or 0))
        record.setdefault("leaf", bool(record.get("leaf", False)))
        record.setdefault("pruned", bool(record.get("pruned", False)))
        record.setdefault("tree_update_mode", record.get("tree_update_mode", "spo"))
        normalized.append(record)
    return normalized


def verify_tree_instance_id_uniqueness(
    edges: Sequence[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """PLAN.md P0.H: real tree-identity verification for a generated batch.

    Stronger than ``bool(set(tree_ids))``:

    * a collision between two stochastic trees merged under one ``tree_id``
      shows up as duplicate ``(tree_id, child_segment_id)`` pairs;
    * a ``tree_id`` equal to the ambiguous ``snapshot:question`` fallback of
      its own edge is a forbidden identity in main runs.

    Returns ``(ok, failure_details)``.
    """
    from collections import Counter

    details: List[str] = []
    pair_counts = Counter(
        (str(e.get("tree_id", "")), str(e.get("child_segment_id", "")))
        for e in edges
    )
    for (tid, child_seg), count in sorted(pair_counts.items()):
        # Legacy fixtures without child_segment_id cannot express a
        # collision; edge_id uniqueness (construction validation) covers
        # duplicates for them.
        if count > 1 and tid and child_seg:
            details.append(
                f"tree_id {tid!r} carries child_segment_id {child_seg!r} "
                f"{count} times — two stochastic trees collided under one id"
            )
    for e in edges:
        tid = str(e.get("tree_id", ""))
        fallback = (
            f"{e.get('policy_snapshot_id', '')}:{e.get('question_id', '')}"
        )
        if tid and tid == fallback:
            details.append(
                f"tree_id {tid!r} equals the ambiguous snapshot:question "
                "fallback — not a unique tree identity"
            )
            break
    return (not details), details


def _queue_segment_identity_failures(
    edges: Sequence[Dict[str, Any]],
) -> Tuple[int, List[str]]:
    """PLAN.md P0.B: per-tree identity
    ``sum_q queue_released_segment_count[q] == tree_total_segment_count``.

    Only meaningful on a COMPLETE generated tree — a partial replay sample
    is missing queues by design, so this must never run on sampled batches.
    """
    from collections import defaultdict

    tree_totals: Dict[str, int] = {}
    queue_totals: Dict[str, int] = defaultdict(int)
    tree_queue_seen: Dict[Tuple[str, str], bool] = {}
    for edge in edges:
        tid = str(edge.get("tree_id", ""))
        total = int(edge.get("tree_total_segment_count", 0) or 0)
        if total > 0:
            tree_totals[tid] = total
        qid = str(edge.get("queue_flush_id", "0"))
        q_key = (tid, qid)
        if not tree_queue_seen.get(q_key):
            tree_queue_seen[q_key] = True
            queue_totals[tid] += int(edge.get("queue_released_segment_count", 0) or 0)
    details: List[str] = []
    for tid, total in tree_totals.items():
        got = queue_totals.get(tid, total)
        if got != total:
            details.append(
                f"tree {tid!r}: sum_q queue_released_segment_count = {got} "
                f"!= tree_total_segment_count = {total}"
            )
    return len(details), details


def validate_tree_construction(
    edges: Sequence[Dict[str, Any]],
    *,
    strict_fresh_iid: bool = True,
) -> Dict[str, Any]:
    """PLAN.md P0.B: full generated-tree construction validation.

    Runs once on the COMPLETE batch of edges extracted from freshly
    generated trees, immediately after normalization and before the edges
    are inserted into replay. It is the only place allowed to require
    complete parent groups and full-tree queue identities:

      * every parent group: one tree_id, one allocated_k;
      * fresh_iid parent groups: row_count == allocated_k and
        sample_multiplicity == 1 on every row;
      * edge IDs unique within the generated batch;
      * no pruned placeholder appears as a trainable edge;
      * per tree: sum_q queue_released_segment_count[q] ==
        tree_total_segment_count;
      * stored old log-probs align with response tokens.

    Raises ``ValueError`` on any failure when ``strict_fresh_iid``; returns
    a diagnostics dict either way.
    """
    from collections import Counter

    failures: List[str] = []

    # Parent-group invariants (complete trees only).
    try:
        metrics = validate_group_integrity(edges, strict_fresh_iid=False)
    except ValueError as exc:  # pragma: no cover - non-strict never raises
        metrics = {"vdra/group_integrity_failures": 1}
        failures.append(str(exc))
    group_failures = int(metrics.get("vdra/group_integrity_failures", 0) or 0)
    if group_failures:
        failures.append(
            f"{group_failures} parent-group integrity failure(s) in generated batch"
        )

    # Edge-ID uniqueness within the generated batch.
    id_counts = Counter(str(e.get("edge_id", "")) for e in edges)
    duplicate_ids = sorted(
        eid for eid, count in id_counts.items() if count > 1 and eid
    )
    missing_ids = sum(1 for e in edges if not str(e.get("edge_id", "")))
    if duplicate_ids:
        failures.append(f"duplicate edge_id values: {duplicate_ids[:5]}")
    if missing_ids:
        failures.append(f"{missing_ids} generated edge(s) missing edge_id")

    # Pruned placeholders must never be emitted as trainable edges.
    pruned_rows = sum(1 for e in edges if bool(e.get("pruned", False)))
    if pruned_rows:
        failures.append(
            f"{pruned_rows} pruned placeholder row(s) present in generated batch"
        )

    # Full-tree queue identity.
    queue_failures, queue_details = _queue_segment_identity_failures(edges)
    failures.extend(queue_details)

    # Stored old log-probs must align with response tokens row by row.
    misaligned = 0
    for e in edges:
        log_probs = e.get("actor_shifted_log_probs")
        response = e.get("response_token_ids")
        if log_probs is not None and response is not None:
            if len(list(log_probs)) != len(list(response)):
                misaligned += 1
    if misaligned:
        failures.append(
            f"{misaligned} generated edge(s) with old log-probs misaligned "
            "with response tokens"
        )

    metrics.update(
        {
            "vdra/construction_failures": len(failures),
            "vdra/queue_segment_identity_failures": float(queue_failures),
            "vdra/generated_duplicate_edge_ids": float(len(duplicate_ids)),
            "vdra/generated_pruned_rows": float(pruned_rows),
        }
    )
    if failures and strict_fresh_iid:
        raise ValueError(
            "Tree-construction validation failed (PLAN.md P0.B):\n  "
            + "\n  ".join(failures)
        )
    return metrics


def validate_replay_batch(
    edges: Sequence[Dict[str, Any]],
    *,
    target_edges_per_iteration: Optional[int] = None,
    max_edges_per_question_per_iteration: Optional[int] = None,
    max_edge_age_iterations: Optional[int] = None,
    current_rollout_iteration: Optional[int] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """PLAN.md P0.B: sampled replay-batch validation (row-local only).

    Edge-level replay intentionally splits trees and parent groups, so this
    validator must NEVER require complete trees, complete parent groups,
    ``row_count == allocated_k``, or queue totals reconstructed from the
    partial sample. It checks only:

      * required row metadata exists (edge_id, question_id,
        generation_rollout_iteration, training advantage);
      * edge_id values are unique in the sampled batch;
      * stored old log-probs align with response tokens;
      * ages are in ``[0, max_edge_age_iterations)`` (when age inputs given);
      * per-question selected count <= resolved cap (when given);
      * selected count <= target (when given).

    Raises ``ValueError`` on failure when ``strict``; returns diagnostics.
    """
    from collections import Counter, defaultdict

    failures: List[str] = []

    missing_edge_id = 0
    missing_question = 0
    missing_generation_iteration = 0
    missing_advantage = 0
    misaligned_log_probs = 0
    per_question: Dict[str, int] = defaultdict(int)
    ages: List[int] = []
    for e in edges:
        if not str(e.get("edge_id", "")):
            missing_edge_id += 1
        qid = str(e.get("question_id", ""))
        if not qid:
            missing_question += 1
        per_question[qid] += 1
        if e.get("generation_rollout_iteration") is None:
            missing_generation_iteration += 1
        elif current_rollout_iteration is not None:
            ages.append(
                int(current_rollout_iteration)
                - int(e.get("generation_rollout_iteration"))
            )
        if e.get("advantage") is None:
            missing_advantage += 1
        log_probs = e.get("actor_shifted_log_probs")
        response = e.get("response_token_ids")
        if log_probs is not None and response is not None:
            if len(list(log_probs)) != len(list(response)):
                misaligned_log_probs += 1

    if missing_edge_id:
        failures.append(f"{missing_edge_id} sampled edge(s) missing edge_id")
    if missing_question:
        failures.append(f"{missing_question} sampled edge(s) missing question_id")
    if missing_generation_iteration:
        failures.append(
            f"{missing_generation_iteration} sampled edge(s) missing "
            "generation_rollout_iteration"
        )
    if missing_advantage:
        failures.append(
            f"{missing_advantage} sampled edge(s) missing training advantage"
        )
    if misaligned_log_probs:
        failures.append(
            f"{misaligned_log_probs} sampled edge(s) with old log-probs "
            "misaligned with response tokens"
        )

    id_counts = Counter(str(e.get("edge_id", "")) for e in edges)
    duplicate_ids = sorted(
        eid for eid, count in id_counts.items() if count > 1 and eid
    )
    if duplicate_ids:
        failures.append(f"duplicate sampled edge_id values: {duplicate_ids[:5]}")

    invalid_ages = 0
    if ages and max_edge_age_iterations is not None:
        invalid_ages = sum(
            1 for a in ages if a < 0 or a >= int(max_edge_age_iterations)
        )
        if invalid_ages:
            failures.append(
                f"{invalid_ages} sampled edge(s) with age outside "
                f"[0, {int(max_edge_age_iterations)})"
            )

    max_per_question = max(per_question.values()) if per_question else 0
    if (
        max_edges_per_question_per_iteration is not None
        and max_per_question > int(max_edges_per_question_per_iteration)
    ):
        failures.append(
            f"per-question selected count {max_per_question} exceeds resolved "
            f"cap {int(max_edges_per_question_per_iteration)}"
        )

    if (
        target_edges_per_iteration is not None
        and len(edges) > int(target_edges_per_iteration)
    ):
        failures.append(
            f"selected edge count {len(edges)} exceeds "
            f"target_edges_per_iteration {int(target_edges_per_iteration)}"
        )

    metrics = {
        "vdra/replay_batch_failures": len(failures),
        "vdra/replay_selected_edges": float(len(edges)),
        "vdra/replay_unique_questions": float(len(per_question)),
        "vdra/replay_max_per_question": float(max_per_question),
        "vdra/replay_invalid_ages": float(invalid_ages),
    }
    if failures and strict:
        raise ValueError(
            "Replay-batch validation failed (PLAN.md P0.B):\n  "
            + "\n  ".join(failures)
        )
    return metrics


def _compute_position_id_with_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Local narrow equivalent of verl.utils.model.compute_position_id_with_mask.

    Avoid importing the broad upstream verl model utility in CPU VDRA tests; that
    module eagerly imports optional HF model classes unrelated to tree data.
    """

    return (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0)


def _left_pad(ids: Sequence[int], length: int, pad_id: int) -> List[int]:
    ids = list(ids)
    return [pad_id] * (length - len(ids)) + ids


def _right_pad(ids: Sequence[int], length: int, pad_id: int) -> List[int]:
    ids = list(ids)
    return ids + [pad_id] * (length - len(ids))


# PLAN.md P0.C: loss modes whose actor loss consumes the float objective
# weight tensors. The canonical ``vdra_segment_mean_ppo`` batch-slot mean
# does NOT — it must receive neither tensor.
_LOSS_MODES_WITH_OBJECTIVE_WEIGHTS = ("vdra_node_balanced_ppo",)


def edges_to_dataproto(
    edges: List[Dict[str, Any]],
    tokenizer,
    *,
    max_prompt_length: int,
    max_response_length: int,
    include_old_log_probs: bool = True,
    loss_mode: str = "vdra_segment_mean_ppo",
) -> DataProto:
    """Build a DataProto whose rows are tree edges.

    Tensor fields (all right/left padded to fixed lengths):
      ``prompts, responses, input_ids, attention_mask, position_ids,
      response_mask, advantages, returns, values, token_level_rewards`` and
      (optionally) ``old_log_probs``.
    Non-tensor fields: ``uid`` (per source question, for grouping/logging),
    ``question_id``, ``reward_model``, ``extra_info``.

    PLAN.md P0.C: ``loss_mode`` selects which float weight tensors are
    attached. The canonical ``vdra_segment_mean_ppo`` mode attaches NO
    ``objective_weights`` / ``segment_objective_weights`` — the main loss
    uses the equal replay-slot mean (``original_optimizer_batch_slot_count``
    supplied by the actor). Only the explicit ``vdra_node_balanced_ppo``
    ablation still receives its precomputed weights. Integer group/identity
    tensors are attached unconditionally for diagnostics and validation.
    """
    if not edges:
        raise ValueError("edges_to_dataproto received an empty edge list")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    bsz = len(edges)
    prompts = torch.empty((bsz, max_prompt_length), dtype=torch.long)
    responses = torch.empty((bsz, max_response_length), dtype=torch.long)
    prompt_mask = torch.zeros((bsz, max_prompt_length), dtype=torch.long)
    response_mask = torch.zeros((bsz, max_response_length), dtype=torch.long)

    # Stable uid per source question so group-relative logging/estimators work.
    qid_to_uid: Dict[Any, str] = {}
    uids: List[str] = []
    question_ids: List[Any] = []
    reward_models: List[dict] = []
    extra_infos: List[dict] = []

    # Truncated per-token old-logprobs, aligned to the (truncated) response.
    for row, edge in enumerate(edges):
        q_ids = edge.get("query_token_ids") or []
        r_ids = edge.get("response_token_ids") or []
        if not r_ids:
            raise ValueError(f"edge {row} has no response_token_ids")
        if len(q_ids) > max_prompt_length:
            raise ValueError(
                f"edge {row} query_token_ids length {len(q_ids)} exceeds max_prompt_length "
                f"{max_prompt_length}; strict VDRA forbids silent context truncation"
            )
        if len(r_ids) > max_response_length:
            raise ValueError(
                f"edge {row} response_token_ids length {len(r_ids)} exceeds max_response_length "
                f"{max_response_length}; strict VDRA forbids silent response truncation"
            )

        valid_r = len(r_ids)
        valid_q = len(q_ids)

        prompts[row] = torch.tensor(_left_pad(q_ids, max_prompt_length, pad_id), dtype=torch.long)
        responses[row] = torch.tensor(_right_pad(r_ids, max_response_length, pad_id), dtype=torch.long)
        prompt_mask[row, max_prompt_length - valid_q :] = 1
        response_mask[row, :valid_r] = 1

        qid = edge.get("question_id")
        if qid not in qid_to_uid:
            qid_to_uid[qid] = str(uuid.uuid4())
        uids.append(qid_to_uid[qid])
        question_ids.append(qid)
        instance = edge.get("instance", {}) or {}
        reward_models.append(
            instance.get("reward_model", {"ground_truth": instance.get("answer")})
        )
        extra_infos.append({"problem": instance.get("problem")})

    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=-1)
    position_ids = _compute_position_id_with_mask(attention_mask)

    batch = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
        },
        batch_size=bsz,
    )

    # Broadcast edge scalars -> per-token tensors (advantages/values/returns/
    # rewards/old_log_probs) using the shared, treetune-faithful helper.
    token_fields = token_fields_for_edges(
        edges, response_mask, include_old_log_probs=include_old_log_probs
    )
    for key, value in token_fields.items():
        batch[key] = value

    # PLAN.md P0.N4: attach the canonical row-level group tensors so the
    # node-balanced actor loss can reduce token -> child -> parent -> tree
    # without a second pass over non-tensor metadata.
    for key, value in group_tensors_for_edges(edges).items():
        batch[key] = value

    # PLAN.md P0.C: float objective-weight tensors are attached ONLY for the
    # explicit node-balanced ablation. The canonical segment-mean loss uses
    # the equal replay-slot mean and must receive neither tensor; tree- and
    # parent-normalized weights would silently re-couple the batch to
    # complete-tree assumptions.
    if str(loss_mode) in _LOSS_MODES_WITH_OBJECTIVE_WEIGHTS:
        obj_weights = compute_objective_weights(edges)
        validate_objective_weights(edges, obj_weights)
        batch["objective_weights"] = torch.tensor(obj_weights, dtype=torch.float32)

        seg_weights = compute_segment_objective_weights(edges)
        validate_segment_objective_weights(edges, seg_weights)
        batch["segment_objective_weights"] = torch.tensor(
            seg_weights, dtype=torch.float32
        )

    non_tensor_batch = {
        "uid": np.array(uids, dtype=object),
        "question_id": np.array(question_ids, dtype=object),
        "reward_model": np.array(reward_models, dtype=object),
        "extra_info": np.array(extra_infos, dtype=object),
        # Keep the raw string ids for logging / manifest validation.
        "tree_id": np.array([str(e.get("tree_id", "")) for e in edges], dtype=object),
        "parent_group_id": np.array(
            [str(e.get("parent_group_id", "")) for e in edges], dtype=object
        ),
        "child_segment_id": np.array(
            [str(e.get("child_segment_id", "")) for e in edges], dtype=object
        ),
        "queue_flush_id": np.array(
            [str(e.get("queue_flush_id", "0")) for e in edges], dtype=object
        ),
        # PLAN.md P0.2: keep the pre-filter counts as string metadata for
        # manifests / logging in addition to the int64 tensor above.
        "tree_total_segment_count_str": np.array(
            [str(int(e.get("tree_total_segment_count", 0) or 0)) for e in edges],
            dtype=object,
        ),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
