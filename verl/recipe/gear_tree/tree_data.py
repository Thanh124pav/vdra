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
    """Precompute the TREE-BALANCED weight for every row (PLAN.md P0.4).

    This is the ``tree_balanced_segment_mean`` ABLATION weight, NOT a
    canonical objective: ``segment_mean`` / ``token_mean`` use the
    trainer-stamped logical denominators ``M_B`` / ``T_B`` instead.

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
        group) satisfy the PRE-FILTER construction contract
        ``realized_child_count == allocated_k`` and the retained-row bound
        ``row_count <= allocated_k``.

    The retained row count is deliberately NOT required to equal
    ``allocated_k``: exact-zero-advantage edges are intentionally removed by
    the Stage 1 zero filter, so a fresh_iid group may legitimately retain a
    strict subset of its realized children. Rows stamped with
    ``realized_child_count`` at extraction time carry the pre-filter fact;
    legacy rows without the stamp fall back to the retained row count
    (retained == realized by construction for pre-zero-filter fixtures).

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
            expected = next(iter(allocated_values), 0)
            # Zero-filter contract: validate the PRE-FILTER realized count
            # against the allocation and bound the retained rows by it —
            # never require retained == allocated_k, because exact-zero
            # advantage edges are intentionally removed.
            realized_values = {
                int(e["realized_child_count"])
                for e in group
                if e.get("realized_child_count") is not None
            }
            if len(realized_values) > 1:
                failures.append(
                    f"fresh_iid parent_group_id={pgid!r} has inconsistent "
                    f"realized_child_count={sorted(realized_values)}"
                )
            realized = next(iter(realized_values), len(group))
            if expected and realized != expected:
                failures.append(
                    f"fresh_iid parent_group_id={pgid!r} realized {realized} "
                    f"children but allocated_k={expected}"
                )
            if expected and len(group) > expected:
                failures.append(
                    f"fresh_iid parent_group_id={pgid!r} retained {len(group)} "
                    f"rows which exceeds allocated_k={expected}"
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


def derive_edge_id(
    *,
    snapshot_id: str,
    tree_instance_id: str,
    parent_group_id: str,
    child_segment_id: str,
) -> str:
    """PLAN.md M3: deterministic canonical edge identity.

    Derived from exactly (tree_instance_id | parent_group_id |
    child_segment_id), digest-prefixed by the policy snapshot. The digest
    formula is byte-identical to the historical strict-path derivation, so
    well-formed edge IDs do not change value.
    """
    key = f"{tree_instance_id}|{parent_group_id}|{child_segment_id}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
    return f"{snapshot_id}:{digest}"


def is_canonical_tree_instance_id(
    tree_instance_id: Any, *, snapshot_id: Any
) -> bool:
    """PLAN.md M3: True iff ``tree_instance_id`` has the exact structure the
    canonical builder (:func:`tree_rollout.make_tree_instance_id`) produces:

        "{policy_snapshot}|iter:{rollout_iteration}|q:{question}|{tiebreaker}"

    A generic value such as ``"t0"`` — truthy but structureless — is NOT a
    canonical identity. The policy-snapshot component must match the rollout
    snapshot, the rollout-iteration and question markers must be present, and
    the per-tree tiebreaker must be non-empty. This validates structure only;
    it does not change the ``derive_edge_id`` digest for well-formed ids.
    """
    if not tree_instance_id:
        return False
    parts = str(tree_instance_id).split("|")
    if len(parts) < 4:
        return False
    if parts[0] != str(snapshot_id):
        return False
    if not parts[1].startswith("iter:") or parts[1] == "iter:":
        return False
    if not parts[2].startswith("q:") or parts[2] == "q:":
        return False
    tiebreaker = "|".join(parts[3:])
    return bool(tiebreaker)


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
        parent_group = record.get("parent_group_id")
        child_seg = record.get("child_segment_id")
        if strict:
            # PLAN.md M3: strict identity requires tree_instance_id
            # specifically — a legacy tree_id alone is not sufficient.
            tree_instance_id = record.get("tree_instance_id")
            missing = [
                name
                for name, value in (
                    ("tree_instance_id", tree_instance_id),
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
            # PLAN.md M3: reject a generic tree_instance_id (e.g. "t0"). Strict
            # mode requires the canonical builder structure — policy snapshot +
            # rollout iteration + stable question + unique tiebreaker.
            if not is_canonical_tree_instance_id(
                tree_instance_id, snapshot_id=snapshot_id
            ):
                raise ValueError(
                    f"Strict VDRA requires a canonical tree_instance_id on "
                    f"generated edge {idx}; got {tree_instance_id!r} (PLAN.md "
                    "M3). It must be produced by make_tree_instance_id: "
                    "'{snapshot}|iter:{n}|q:{qid}|{tiebreaker}' with the "
                    "policy snapshot matching the rollout snapshot."
                )
            derived = derive_edge_id(
                snapshot_id=snapshot_id,
                tree_instance_id=str(tree_instance_id),
                parent_group_id=str(parent_group),
                child_segment_id=str(child_seg),
            )
            supplied = record.get("edge_id")
            if supplied is not None and str(supplied) != derived:
                raise ValueError(
                    f"Supplied edge_id {supplied!r} does not match the "
                    f"identity-derived id {derived!r} for generated edge "
                    f"{idx} (PLAN.md P0.H/M3)."
                )
            record["edge_id"] = derived
        else:
            # Legacy non-strict compatibility path: fallback chains and a
            # caller-supplied edge_id are honored as-is.
            tree_id = (
                record.get("tree_instance_id")
                or record.get("tree_id")
                or record.get("gear_segment_id", "")
            )
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
            digest = hashlib.blake2b(
                key.encode("utf-8"), digest_size=16
            ).hexdigest()
            record.setdefault("edge_id", f"{snapshot_id}:{digest}")
        record.setdefault("policy_snapshot_id", snapshot_id)
        if record["policy_snapshot_id"] != snapshot_id:
            raise ValueError(
                "Generated edge policy_snapshot_id mismatches rollout snapshot"
            )
        is_slot = (
            "trainable_edge_id" in record and record["trainable_edge_id"] is None
        )
        if is_slot:
            # PLAN.md §1.2: metadata-only logical slot — same identity
            # derivation as trainable edges, no payload checks. Its
            # pre-filter response_token_count must already be stamped.
            if int(record.get("response_token_count", 0) or 0) <= 0:
                raise ValueError(
                    "Generated logical slot is missing a positive "
                    "response_token_count (PLAN.md §1.2)."
                )
            if float(record.get("advantage", 1.0) or 0.0) != 0.0:
                raise ValueError(
                    "Generated logical slot must carry exactly zero "
                    "advantage (PLAN.md §1.2)."
                )
        else:
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
            # PLAN.md §1.2: a trainable record that participates in the
            # sparse ledger points at itself once its edge_id is final.
            if "advantage_is_zero" in record:
                record.setdefault("trainable_edge_id", record["edge_id"])
        record.setdefault("depth", int(record.get("depth", 0) or 0))
        record.setdefault("leaf", bool(record.get("leaf", False)))
        record.setdefault("pruned", bool(record.get("pruned", False)))
        record.setdefault("tree_update_mode", record.get("tree_update_mode", "spo"))
        normalized.append(record)
    return normalized


def _parse_canonical_tree_instance_id(tree_id: str) -> Tuple[str, int, str]:
    """Parse ``snapshot|iter:<n>|q:<question>|...`` tree identities."""
    parts = str(tree_id).split("|")
    if len(parts) < 4:
        raise ValueError(
            f"tree_id {tree_id!r} is not a canonical tree_instance_id"
        )
    snapshot = parts[0]
    iteration_part = parts[1]
    question_part = parts[2]
    if not snapshot:
        raise ValueError(f"tree_id {tree_id!r} has an empty snapshot")
    if not iteration_part.startswith("iter:"):
        raise ValueError(
            f"tree_id {tree_id!r} is missing the iter:<n> component"
        )
    if not question_part.startswith("q:"):
        raise ValueError(
            f"tree_id {tree_id!r} is missing the q:<question> component"
        )
    try:
        rollout_iteration = int(iteration_part.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(
            f"tree_id {tree_id!r} has a non-integer rollout iteration"
        ) from exc
    return snapshot, rollout_iteration, question_part.split(":", 1)[1]


def _summary_tree_identity_failures(record: Dict[str, Any]) -> List[str]:
    """Validate summary-only tree identity against rollout metadata."""
    summary = record.get("tree_summary") or {}
    if not summary:
        return []
    tree_id = str(record.get("tree_id") or summary.get("tree_id") or "")
    missing = [
        name
        for name in (
            "tree_id",
            "policy_snapshot_id",
            "rollout_iteration",
            "question_id",
        )
        if summary.get(name) is None and not (name == "tree_id" and tree_id)
    ]
    if missing:
        return [
            "tree_summary is missing canonical identity metadata "
            f"{missing}; cannot verify summary-only tree identity"
        ]
    try:
        tid_snapshot, tid_iteration, tid_question = _parse_canonical_tree_instance_id(
            tree_id
        )
    except ValueError as exc:
        return [str(exc)]

    failures: List[str] = []
    expected_snapshot = str(summary["policy_snapshot_id"])
    if tid_snapshot != expected_snapshot:
        failures.append(
            f"tree_id snapshot {tid_snapshot!r} != summary.policy_snapshot_id "
            f"{expected_snapshot!r}"
        )
    expected_iteration = int(summary["rollout_iteration"])
    if tid_iteration != expected_iteration:
        failures.append(
            f"tree_id rollout iteration {tid_iteration!r} != "
            f"summary.rollout_iteration {expected_iteration!r}"
        )
    expected_question = str(summary["question_id"])
    if tid_question != expected_question:
        failures.append(
            f"tree_id question {tid_question!r} != summary.question_id "
            f"{expected_question!r}"
        )
    return failures


def verify_tree_instance_id_uniqueness(
    edges: Sequence[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """PLAN.md P0.H: real tree-identity verification for a generated batch.

    Stronger than ``bool(set(tree_ids))``:

    * a collision between two stochastic trees merged under one ``tree_id``
      shows up as duplicate ``(tree_id, child_segment_id)`` pairs;
    * summary-only tree records must have a canonical ``tree_id`` whose
      snapshot, rollout iteration and question match ``tree_summary``;
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
        details.extend(_summary_tree_identity_failures(e))
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

    Zero-filter contract: when the tree carries its extraction-time
    ``tree_summary`` the full PRE-FILTER queue map from that summary is used,
    because a queue whose every edge had exactly zero advantage retains no
    rows at all and would otherwise be miscounted as a construction failure.
    """
    from collections import defaultdict

    tree_totals: Dict[str, int] = {}
    queue_totals: Dict[str, int] = defaultdict(int)
    tree_queue_seen: Dict[Tuple[str, str], bool] = {}
    summary_trees: set = set()
    for edge in edges:
        tid = str(edge.get("tree_id", ""))
        total = int(edge.get("tree_total_segment_count", 0) or 0)
        if total > 0:
            tree_totals[tid] = total
        summary = edge.get("tree_summary")
        if tid not in summary_trees and isinstance(summary, dict):
            queue_map = summary.get("queue_released_segment_count")
            if isinstance(queue_map, dict) and queue_map:
                summary_trees.add(tid)
                queue_totals[tid] = sum(int(v or 0) for v in queue_map.values())
        if tid in summary_trees:
            continue
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
    construction_summaries: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """PLAN.md P0.B: full generated-tree construction validation.

    Runs once on the COMPLETE batch of edges extracted from freshly
    generated trees, immediately after normalization and before the edges
    are inserted into replay. It is the only place allowed to require
    complete parent groups and full-tree queue identities:

      * every parent group: one tree_id, one allocated_k;
      * fresh_iid parent groups: pre-filter realized_child_count ==
        allocated_k, retained row_count <= allocated_k, and
        sample_multiplicity == 1 on every row (zero-filtered subsets are
        legitimate — retained == allocated_k is NOT required);
      * edge IDs unique within the generated batch;
      * no pruned placeholder appears as a trainable edge;
      * per tree: sum_q queue_released_segment_count[q] ==
        tree_total_segment_count;
      * stored old log-probs align with response tokens.

    ``construction_summaries``: extraction-time per-tree summaries (see
    ``extract_edges_from_tree(collect_construction_summaries=...)``). They
    are the only record of a parent — or an entire tree — whose every child
    had exactly zero advantage: such rows are intentionally absent from
    ``edges`` and must never be re-inserted into replay. When provided,
    the summaries' pre-filter ``realized == allocated_k`` facts are
    validated even for parents with zero retained rows.

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

    # Extraction-time construction summaries: the only construction record
    # for parents/trees whose every child was zero-filtered away.
    summary_failure_count = 0
    if construction_summaries:
        summary_details = _construction_summary_failures(construction_summaries)
        summary_failure_count = len(summary_details)
        failures.extend(summary_details)

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
            "vdra/construction_summary_failures": float(summary_failure_count),
        }
    )
    if failures and strict_fresh_iid:
        raise ValueError(
            "Tree-construction validation failed (PLAN.md P0.B):\n  "
            + "\n  ".join(failures)
        )
    return metrics


def _construction_summary_failures(
    construction_summaries: Sequence[Dict[str, Any]],
) -> List[str]:
    """Validate extraction-time construction summaries.

    Covers the facts that retained edges cannot represent: a parent (or a
    whole tree) whose every child had exactly zero advantage retains no rows,
    so its pre-filter ``realized == allocated_k`` contract and its queue
    identity can only be checked here. Zero-filtered rows themselves are
    never re-inserted into replay.
    """
    details: List[str] = []
    for summary in construction_summaries:
        tid = str(summary.get("tree_id", ""))
        parent_facts = summary.get("parent_construction") or {}
        for pgid, facts in parent_facts.items():
            realized = int(facts.get("realized", 0) or 0)
            allocated = int(facts.get("allocated_k", 0) or 0)
            retained = int(facts.get("retained", 0) or 0)
            if allocated and realized != allocated:
                details.append(
                    f"tree {tid!r} parent_group_id={pgid!r} realized "
                    f"{realized} children but allocated_k={allocated}"
                )
            if allocated and retained > allocated:
                details.append(
                    f"tree {tid!r} parent_group_id={pgid!r} retained "
                    f"{retained} rows which exceeds allocated_k={allocated}"
                )
        total = int(summary.get("tree_total_segment_count", 0) or 0)
        queue_map = summary.get("queue_released_segment_count") or {}
        if total and queue_map:
            queue_sum = sum(int(v or 0) for v in queue_map.values())
            if queue_sum != total:
                details.append(
                    f"tree {tid!r}: summary sum_q queue_released_segment_count"
                    f" = {queue_sum} != tree_total_segment_count = {total}"
                )
    return details


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


# PLAN.md P0.C: loss modes whose actor loss consumes precomputed float
# objective-weight tensors. The canonical ``vdra_segment_mean_ppo`` path
# (segment_mean / token_mean) needs NONE: it uses the trainer-stamped logical
# batch denominators M_B / T_B. Node-balanced weights are only for the
# explicit ablation below.
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

    PLAN.md P0.C/§1.3: ``loss_mode`` selects which float weight tensors are
    attached. The canonical ``vdra_segment_mean_ppo`` path attaches NO
    tree-balanced objective weights: ``segment_mean`` / ``token_mean`` use
    the trainer-stamped logical batch denominators ``M_B`` / ``T_B``. The
    formula ``1/(N_T * N_seg(T))`` belongs only to the
    ``tree_balanced_segment_mean`` ablation. Only the explicit
    ``vdra_node_balanced_ppo`` ablation still receives its precomputed
    weights. Integer group/identity tensors are attached unconditionally for
    diagnostics and validation.
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
    # explicit node-balanced ablation. The CANONICAL paper objectives
    # (segment_mean / token_mean) weigh rows uniformly by the trainer-stamped
    # pre-filter logical denominators M_B / T_B and must receive neither
    # tensor; tree- and parent-normalized weights would silently re-couple the
    # batch to complete-tree assumptions. (The 1/(N_T * N_seg(T)) formula
    # belongs to the tree_balanced_segment_mean ABLATION only.)
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


def build_logical_update_batch(
    slots: List[Dict[str, Any]],
    tokenizer,
    *,
    max_prompt_length: int,
    max_response_length: int,
    ppo_mini_batch_size: int,
    dp_size: int,
    loss_mode: str = "vdra_segment_mean_ppo",
    include_old_log_probs: bool = True,
    use_prob_mask: bool = False,
    probability_mask_threshold: float = 0.9,
) -> Tuple[Optional[DataProto], Dict[str, float]]:
    """PLAN.md §1.2/§1.3/§5/§6 (2026-07-21): tensorize a reservation of LOGICAL slots.

    ``slots`` is the reservation in RESERVATION ORDER: metadata-only zero
    slots (``is_ledger_slot``) and trainable rows interleaved. This function:

    1. partitions the slots into logical optimizer batches of
       ``ppo_mini_batch_size`` BEFORE any tensor filtering;
    2. computes the exact PRE-FILTER denominators per batch ``B``:
       ``M_B`` = logical slot count,
       ``T_B_response`` = sum of ``response_token_count``,
       ``T_B_prob_mask`` = sum of ``prob_mask_token_count``,
       each over EVERY slot in ``B`` (zero slots included) — failing fast
       when a record lacks the count the SELECTED objective needs (no
       ``tree_total_segment_count`` approximation, no retained-row fallback);
    3. classifies each logical batch (PLAN.md §6):
       ``all_zero_advantage`` (no trainable rows),
       ``zero_active_tokens`` (trainable rows exist but no effective active
       token on any of them), otherwise ``trainable``;
    4. tensorizes ONLY the trainable rows of TRAINABLE batches, padding each
       to a multiple of ``dp_size`` with collective-safe dummy rows
       (exact-zero advantage, one pad token, explicitly masked and counted in
       NO denominator or replay metric);
    5. orders rows RANK-MAJOR (all of rank 0's rows for batches 0..K-1, then
       rank 1's, ...) so verl's contiguous ``DataProto.chunk`` hands every
       rank the same number of rows for every logical batch — no FSDP rank
       can skip a collective another rank enters;
    6. stamps ``logical_batch_index``/``is_dummy`` row tensors and the
       per-batch ``original_logical_segment_count`` /
       ``original_logical_response_token_count`` /
       ``original_logical_prob_mask_token_count`` /
       ``logical_batch_status`` lists (one value per logical batch, never one
       per call) plus ``logical_dp_size`` into ``meta_info``.

    Returns ``(None, stats)`` when EVERY logical batch is skipped — the
    trainer records an explicit skipped update (no update_actor RPC, no
    global_step, no scheduler.step; PLAN.md §1.3/§7).
    """
    from recipe.gear_tree.prob_mask import count_prob_mask_active_tokens
    from recipe.gear_tree.replay_buffer import is_ledger_slot

    n = len(slots)
    mini = int(ppo_mini_batch_size)
    dp = int(dp_size)
    if n <= 0:
        raise ValueError("build_logical_update_batch received an empty reservation")
    if mini <= 0 or dp <= 0:
        raise ValueError(
            f"ppo_mini_batch_size={ppo_mini_batch_size!r} and dp_size={dp_size!r} "
            "must be positive"
        )
    if n % mini != 0:
        raise ValueError(
            f"reservation of {n} logical slots is not divisible by "
            f"ppo_mini_batch_size={mini}; the postpone_until_divisible policy "
            "must gate the update BEFORE tensorization (PLAN.md P0.D/§1.2)."
        )

    def _slot_counts(slot: Dict[str, Any]) -> Tuple[int, int]:
        """(response_token_count, prob_mask_token_count) — stamped, never
        recomputed for a zero slot whose log-prob payload was removed."""
        raw_resp = slot.get("response_token_count")
        if raw_resp is None:
            raw_resp = len(slot.get("response_token_ids") or []) or None
        if raw_resp is None or int(raw_resp) <= 0:
            raise ValueError(
                "logical slot/edge is missing a positive "
                "response_token_count; the exact pre-filter denominator "
                "cannot be reconstructed (PLAN.md §1.2 — never approximate "
                "with tree_total_segment_count). Offending record edge_id="
                f"{slot.get('edge_id')!r}."
            )
        resp = int(raw_resp)
        # PLAN.md §4: second line of defence for direct/test callers that
        # bypass replay insertion — a stamped count from another threshold
        # must never be consumed under this one.
        raw_thr = slot.get("probability_mask_threshold")
        if raw_thr is not None and abs(
            float(raw_thr) - float(probability_mask_threshold)
        ) > 1e-12:
            raise ValueError(
                f"logical record {slot.get('edge_id')!r} stamps "
                f"probability_mask_threshold={float(raw_thr)} but this batch "
                f"is being built with {float(probability_mask_threshold)}; "
                "its prob_mask_token_count was computed under a different "
                "objective and must not be reused (PLAN.md §4)."
            )
        raw_mask = slot.get("prob_mask_token_count")
        if not is_ledger_slot(slot):
            # PLAN.md §1: a TRAINABLE row still carries its payload, so its
            # stamped counts must AGREE with it — never merely be present.
            response_ids = slot.get("response_token_ids") or []
            if response_ids and resp != len(response_ids):
                raise ValueError(
                    f"logical edge {slot.get('edge_id')!r} stamps "
                    f"response_token_count={resp} but carries "
                    f"{len(response_ids)} response tokens (PLAN.md §1)."
                )
            log_probs = slot.get("actor_shifted_log_probs")
            if log_probs is not None:
                recomputed = count_prob_mask_active_tokens(
                    log_probs, probability_mask_threshold
                )
                if raw_mask is None:
                    raw_mask = recomputed
                elif int(raw_mask) != recomputed:
                    raise ValueError(
                        f"logical edge {slot.get('edge_id')!r} stamps "
                        f"prob_mask_token_count={int(raw_mask)} but its stored "
                        f"old log-probs give {recomputed} active tokens at "
                        f"threshold={float(probability_mask_threshold)} "
                        "(PLAN.md §1)."
                    )
        if use_prob_mask and raw_mask is None:
            raise ValueError(
                "policy_aggregation='token_mean' with use_prob_mask=true "
                "requires a stamped prob_mask_token_count on every logical "
                "record; a zero slot's active count can NEVER be recomputed "
                "because its old-log-prob payload was removed (PLAN.md §4). "
                f"Offending record edge_id={slot.get('edge_id')!r}."
            )
        mask_count = 0 if raw_mask is None else int(raw_mask)
        if not (0 <= mask_count <= resp):
            raise ValueError(
                "prob_mask_token_count must satisfy 0 <= count <= "
                f"response_token_count; got {mask_count} > {resp} for "
                f"edge_id={slot.get('edge_id')!r} (PLAN.md §4)."
            )
        return resp, mask_count

    STATUS_TRAINABLE = "trainable"
    STATUS_ALL_ZERO = "all_zero_advantage"
    STATUS_ZERO_ACTIVE = "zero_active_tokens"

    n_batches = n // mini
    segment_counts: List[float] = []
    response_token_counts: List[float] = []
    prob_mask_token_counts: List[float] = []
    statuses: List[str] = []
    trainable_active_tokens: List[float] = []
    per_batch_rows: List[List[Dict[str, Any]]] = []
    all_zero_batches = 0
    zero_active_batches = 0
    for k in range(n_batches):
        batch_slots = slots[k * mini : (k + 1) * mini]
        t_resp = 0
        t_mask = 0
        for slot in batch_slots:
            resp, mask_count = _slot_counts(slot)
            t_resp += resp
            t_mask += mask_count
        rows = [s for s in batch_slots if not is_ledger_slot(s)]
        # PLAN.md §6: executable signal counts EFFECTIVE active tokens on the
        # nonzero-advantage trainable rows only.
        active = 0
        for row in rows:
            resp, mask_count = _slot_counts(row)
            active += mask_count if use_prob_mask else resp
        if not rows:
            status = STATUS_ALL_ZERO
            all_zero_batches += 1
            rows = []
        elif active <= 0:
            status = STATUS_ZERO_ACTIVE
            zero_active_batches += 1
            rows = []  # nothing executable: no tensor rows for this batch
        else:
            status = STATUS_TRAINABLE
        segment_counts.append(float(mini))
        response_token_counts.append(float(t_resp))
        prob_mask_token_counts.append(float(t_mask))
        statuses.append(status)
        trainable_active_tokens.append(float(active))
        per_batch_rows.append(rows)

    total_rows = sum(len(rows) for rows in per_batch_rows)
    total_response_tokens = sum(response_token_counts)
    total_mask_tokens = sum(prob_mask_token_counts)
    stats: Dict[str, float] = {
        "vdra/logical_slots": float(n),
        "vdra/logical_batches": float(n_batches),
        "vdra/all_zero_advantage_logical_batches": float(all_zero_batches),
        "vdra/zero_active_token_logical_batches": float(zero_active_batches),
        "vdra/trainable_logical_batches": float(
            n_batches - all_zero_batches - zero_active_batches
        ),
        "vdra/tensor_rows": float(total_rows),
        "vdra/prob_mask_active_token_fraction": (
            float(total_mask_tokens / total_response_tokens)
            if total_response_tokens > 0
            else 0.0
        ),
    }
    if total_rows == 0:
        stats["vdra/skipped_zero_gradient_updates"] = 1.0
        stats["vdra/dummy_rows"] = 0.0
        return None, stats

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def _dummy_row(batch_idx: int, i: int) -> Dict[str, Any]:
        return {
            "edge_id": f"__vdra_dummy__|{batch_idx}|{i}",
            "question_id": "__vdra_dummy__",
            "tree_id": "__vdra_dummy__",
            "parent_group_id": "__vdra_dummy__/p",
            "child_segment_id": f"__vdra_dummy__/{batch_idx}/{i}",
            "allocated_k": 1,
            "sample_multiplicity": 1,
            "tree_total_segment_count": 1,
            "queue_flush_id": "0",
            "queue_released_segment_count": 1,
            "query_token_ids": [int(pad_id)],
            "response_token_ids": [int(pad_id)],
            "actor_shifted_log_probs": [0.0],
            "advantage": 0.0,
            "value": 0.0,
            "reward": 0.0,
            "is_dummy": True,
        }

    # Round-robin rank assignment per logical batch, dummies appended last so
    # real rows spread as evenly as possible.
    n_dummy = 0
    per_rank_rows: List[List[Dict[str, Any]]] = [[] for _ in range(dp)]
    per_rank_batch_idx: List[List[int]] = [[] for _ in range(dp)]
    for k, rows in enumerate(per_batch_rows):
        if not rows:
            # Skipped logical batch (all_zero_advantage or zero_active_tokens):
            # zero rows on EVERY rank, so all ranks skip optimizer step k
            # consistently (PLAN.md §1.3/§7).
            continue
        padded = list(rows)
        while len(padded) % dp != 0:
            padded.append(_dummy_row(k, len(padded)))
            n_dummy += 1
        for i, row in enumerate(padded):
            rank = i % dp
            per_rank_rows[rank].append(row)
            per_rank_batch_idx[rank].append(k)

    ordered_rows: List[Dict[str, Any]] = []
    ordered_batch_idx: List[int] = []
    rows_per_rank = {len(rows) for rows in per_rank_rows}
    if len(rows_per_rank) != 1:
        raise AssertionError(
            "rank shares diverged after dummy padding; this is a bug in "
            f"build_logical_update_batch (shares={sorted(rows_per_rank)})"
        )
    for rank in range(dp):
        ordered_rows.extend(per_rank_rows[rank])
        ordered_batch_idx.extend(per_rank_batch_idx[rank])

    batch = edges_to_dataproto(
        ordered_rows,
        tokenizer,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        include_old_log_probs=include_old_log_probs,
        loss_mode=loss_mode,
    )
    batch.batch["logical_batch_index"] = torch.tensor(
        ordered_batch_idx, dtype=torch.int64
    )
    batch.batch["is_dummy"] = torch.tensor(
        [1 if row.get("is_dummy") else 0 for row in ordered_rows],
        dtype=torch.int64,
    )
    # PLAN.md §5/§6: aligned per-logical-batch lists (one value per batch,
    # never one for the whole update_actor call).
    batch.meta_info["original_logical_segment_count"] = segment_counts
    batch.meta_info["original_logical_response_token_count"] = response_token_counts
    batch.meta_info["original_logical_prob_mask_token_count"] = prob_mask_token_counts
    batch.meta_info["logical_batch_status"] = statuses
    batch.meta_info["logical_batch_trainable_active_tokens"] = trainable_active_tokens
    batch.meta_info["logical_batch_count"] = int(n_batches)
    batch.meta_info["logical_dp_size"] = int(dp)
    # The objective-mask identity the denominators were computed under, so the
    # actor can assert it matches its own configuration.
    batch.meta_info["logical_use_prob_mask"] = bool(use_prob_mask)
    batch.meta_info["logical_probability_mask_threshold"] = float(
        probability_mask_threshold
    )
    stats["vdra/dummy_rows"] = float(n_dummy)
    return batch, stats
