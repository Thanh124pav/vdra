"""Trainer-owned replay buffer for tree-family VERL edge updates.

PLAN.md P0.A: canonical VDRA replay is EDGE-level — the trainer reserves
individual edges via :meth:`reserve_for_update` (per-question cap, hard
target cap, transactional commit/rollback). Complete-tree replay via
:meth:`reserve_complete_trees_for_update` is retained only as the explicit
``replay_sampling_unit: complete_tree`` ablation; it is never selected by
``strict_group_integrity``. Use :func:`reserve_replay_edges` to dispatch on
the configured sampling unit.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


_REQUIRED_EDGE_FIELDS = (
    "edge_id",
    "question_id",
    "query_token_ids",
    "response_token_ids",
    "actor_shifted_log_probs",
    "advantage",
    "value",
    "reward",
)

# PLAN.md §1.2 (2026-07-21): a metadata-only logical slot for an exact-zero
# advantage segment. It has no trainable payload but MUST carry enough
# bookkeeping for reservation, per-question caps, divisibility and the
# pre-filter M_B / T_B denominators.
_REQUIRED_SLOT_FIELDS = (
    "edge_id",
    "question_id",
    "tree_id",
    "parent_group_id",
    "advantage",
    "response_token_count",
)

_SLOT_FORBIDDEN_PAYLOAD_FIELDS = (
    "query_token_ids",
    "response_token_ids",
    "actor_shifted_log_probs",
)


def is_ledger_slot(edge: Mapping[str, Any]) -> bool:
    """True for metadata-only zero-advantage logical slots (PLAN.md §1.2).

    A slot is identified by an EXPLICIT ``trainable_edge_id=None`` marker —
    dense zero-advantage rows (full payload, ``trainable_edge_id`` set to
    their own id) are ordinary trainable records, never slots.
    """
    return "trainable_edge_id" in edge and edge["trainable_edge_id"] is None


def compute_max_edges_per_question(
    tree_shape: Sequence[int],
    *,
    trees_per_question: int = 1,
    max_edge_age_iterations: int,
) -> int:
    """PLAN.md P0.2 auto per-question cap.

    Given a per-depth branch factor ``tree_shape = [b_1, ..., b_D]``, the
    maximum number of non-root edges in one full tree is

        E_max = sum_{d=1..D} prod_{l=1..d} b_l.

    With ``R`` stochastic trees per question per rollout iteration,

        E_max^{q/iter} = R * E_max,
        C_question    = ceil(E_max^{q/iter} / max_edge_age_iterations).

    Examples (R=1, age=8):
        [6,6,6]  → E_max=258  → cap=33
        [8,8,8]  → E_max=584  → cap=73
    """

    if max_edge_age_iterations <= 0:
        raise ValueError(
            "max_edge_age_iterations must be > 0 to resolve auto per-question cap"
        )
    if trees_per_question <= 0:
        raise ValueError("trees_per_question must be > 0")
    shape = [int(b) for b in tree_shape if int(b) > 0]
    if not shape:
        raise ValueError("tree_shape must contain at least one positive branch")
    e_max = 0
    prod = 1
    for b in shape:
        prod *= b
        e_max += prod
    e_max *= int(trees_per_question)
    return max(1, math.ceil(e_max / int(max_edge_age_iterations)))


@dataclass(frozen=True)
class ReplayReservation:
    reservation_id: int
    edge_ids: Tuple[str, ...]
    edges: Tuple[Dict[str, Any], ...]
    stats: Dict[str, Any]


VALID_REPLAY_SAMPLING_UNITS = ("edge", "complete_tree")

VALID_UNDERFILLED_UPDATE_POLICIES = ("postpone_until_divisible", "use_available")


def should_postpone_sampled_update(
    *,
    selected_count: int,
    target_edges_per_iteration: int,
    ppo_mini_batch_size: int,
    underfilled_update_policy: str = "postpone_until_divisible",
) -> bool:
    """PLAN.md P0.D: exact optimizer-batch cardinality for one iteration.

    * ``selected_count > target`` is a sampler bug and raises
      ``AssertionError`` — both reservation paths cap at the target, so an
      oversized batch must never reach the actor.
    * Canonical ``postpone_until_divisible``: postpone whenever the count is
      not divisible by ``ppo_mini_batch_size`` (under- OR over-filled), so no
      tail optimizer batch can form.
    * ``use_available``: ablation-only; runs whatever was sampled.
    """
    n = int(selected_count)
    target = int(target_edges_per_iteration)
    if n > target:
        raise AssertionError(
            f"replay sampler returned {n} edges, exceeding "
            f"target_edges_per_iteration={target} (PLAN.md P0.D). This is a "
            "sampler bug; the reservation path must cap at the target."
        )
    policy = str(underfilled_update_policy)
    if policy == "use_available":
        return False
    if policy != "postpone_until_divisible":
        raise ValueError(f"Unknown underfilled_update_policy: {policy!r}")
    if n == 0:
        return False
    return n % int(ppo_mini_batch_size) != 0


def batch_has_zero_learning_signal(edges: Sequence[Mapping[str, Any]]) -> bool:
    """Diagnostic predicate for experiments; canonical trainer does not skip.

    Uses the exact ``advantage`` scalar tensorization broadcasts into the
    policy ``advantages`` tensor. Missing advantages are invalid replay rows
    and must fail instead of being interpreted as zero.
    """
    edge_list = list(edges)
    if not edge_list:
        return False
    for edge in edge_list:
        if "advantage" not in edge or edge["advantage"] is None:
            raise ValueError("sampled edge is missing training advantage")
    return all(float(edge["advantage"]) == 0.0 for edge in edge_list)


def expected_optimizer_steps(
    *,
    selected_count: int,
    ppo_mini_batch_size: int,
    ppo_epochs: int = 1,
) -> int:
    """PLAN.md P0.D: N_steps = N_selected / ppo_mini_batch_size * ppo_epochs.

    Valid ONLY after divisibility has been enforced; raises otherwise so the
    floor formula can never silently hide a tail batch.
    """
    n = int(selected_count)
    mini = int(ppo_mini_batch_size)
    if mini <= 0:
        raise ValueError("ppo_mini_batch_size must be > 0")
    if n % mini != 0:
        raise ValueError(
            f"expected_optimizer_steps requires selected_count divisible by "
            f"ppo_mini_batch_size; got {n} % {mini} != 0 (PLAN.md P0.D)."
        )
    return n // mini * max(int(ppo_epochs), 1)


def reserve_replay_edges(
    replay_buffer: "GearTreeReplayBuffer",
    *,
    replay_sampling_unit: str,
    current_rollout_iteration: int,
) -> ReplayReservation:
    """PLAN.md P0.A: production reservation dispatch.

    The sampling unit comes from config only. ``edge`` is canonical;
    ``complete_tree`` is an explicit non-canonical ablation. Strictness
    (``tree_policy.strict_group_integrity``) controls validation and must
    never select the reservation path.
    """

    unit = str(replay_sampling_unit)
    if unit == "edge":
        return replay_buffer.reserve_for_update(
            current_rollout_iteration=current_rollout_iteration
        )
    if unit == "complete_tree":
        return replay_buffer.reserve_complete_trees_for_update(
            current_rollout_iteration=current_rollout_iteration
        )
    raise ValueError(
        f"Unknown replay_sampling_unit={unit!r}; expected one of "
        f"{VALID_REPLAY_SAMPLING_UNITS} (PLAN.md P0.A)."
    )


class GearTreeReplayBuffer:
    """CPU-native edge replay buffer shared by SPO/VDRA tree methods."""

    schema_version = 1

    def __init__(
        self,
        *,
        # PLAN.md P0.2 canonical names
        target_edges_per_iteration: Optional[int] = None,
        max_edge_age_iterations: Optional[int] = None,
        max_edges_per_question_per_iteration: Union[int, str, None] = None,
        replay_sampling_unit: str = "edge",
        tree_shape: Optional[Sequence[int]] = None,
        trees_per_question: int = 1,
        # PLAN.md P0.2 deprecated aliases — kept for one release for migration.
        target_edges_per_update: Optional[int] = None,
        max_edge_age: Optional[int] = None,
        max_edges_per_question: Union[int, str, None] = None,
        underfill_policy: str = "use_available",
        sampling_seed: int = 0,
    ) -> None:
        # PLAN.md P0.2: accept both the new canonical names and the deprecated
        # aliases. Prefer the new name if both are set to a non-default value.
        resolved_target = self._resolve_alias(
            new=target_edges_per_iteration,
            old=target_edges_per_update,
            new_name="target_edges_per_iteration",
            old_name="target_edges_per_update",
            default=512,
        )
        resolved_age = self._resolve_alias(
            new=max_edge_age_iterations,
            old=max_edge_age,
            new_name="max_edge_age_iterations",
            old_name="max_edge_age",
            default=8,
        )
        resolved_cap_raw = self._resolve_alias(
            new=max_edges_per_question_per_iteration,
            old=max_edges_per_question,
            new_name="max_edges_per_question_per_iteration",
            old_name="max_edges_per_question",
            default="auto",
        )

        self.target_edges_per_iteration = max(int(resolved_target), 1)
        self.max_edge_age_iterations = max(int(resolved_age), 1)
        self.replay_sampling_unit = str(replay_sampling_unit)
        if self.replay_sampling_unit not in VALID_REPLAY_SAMPLING_UNITS:
            raise ValueError(
                "replay_sampling_unit must be 'edge' (canonical) or "
                "'complete_tree' (explicit ablation), got "
                f"{self.replay_sampling_unit!r} (PLAN.md P0.A)."
            )
        self.tree_shape: Tuple[int, ...] = tuple(int(b) for b in (tree_shape or ()))
        self.trees_per_question = max(int(trees_per_question), 1)
        # Resolve auto/int for per-question cap.
        if isinstance(resolved_cap_raw, str) and str(resolved_cap_raw).strip().lower() == "auto":
            if not self.tree_shape:
                raise ValueError(
                    "max_edges_per_question_per_iteration='auto' requires "
                    "tree_shape to be passed to GearTreeReplayBuffer "
                    "(PLAN.md P0.2)."
                )
            self.max_edges_per_question_per_iteration = compute_max_edges_per_question(
                self.tree_shape,
                trees_per_question=self.trees_per_question,
                max_edge_age_iterations=self.max_edge_age_iterations,
            )
            self.max_edges_per_question_cap_source = "auto"
        else:
            self.max_edges_per_question_per_iteration = max(int(resolved_cap_raw), 1)
            self.max_edges_per_question_cap_source = "override"
        self.resolved_max_edges_per_question_per_iteration = (
            self.max_edges_per_question_per_iteration
        )

        # Backward-compat attribute aliases so old callers (and JSON meta) still
        # read the same value under the old name.
        self.target_edges_per_update = self.target_edges_per_iteration
        self.max_edge_age = self.max_edge_age_iterations
        self.max_edges_per_question = self.max_edges_per_question_per_iteration

        self.underfill_policy = str(underfill_policy)
        if self.underfill_policy != "use_available":
            raise ValueError("Only underfill_policy='use_available' is currently supported")
        self.sampling_seed = int(sampling_seed)
        self._edges: Dict[str, Dict[str, Any]] = {}
        self._reserved: Dict[str, int] = {}
        self._next_reservation_id = 1
        self.metrics: Dict[str, int] = {
            "added_edges": 0,
            "expired_edges": 0,
            "sampled_edges": 0,
            "reserved_edges": 0,
            "committed_edges": 0,
            "rolled_back_edges": 0,
        }

    @staticmethod
    def _resolve_alias(
        *,
        new: Any,
        old: Any,
        new_name: str,
        old_name: str,
        default: Any,
    ) -> Any:
        """Prefer the new PLAN.md P0.2 name; accept the deprecated one with a warning."""
        if new is not None and old is not None and new != old:
            raise ValueError(
                f"{new_name}={new!r} conflicts with deprecated alias "
                f"{old_name}={old!r}; drop {old_name} and use {new_name} only."
            )
        if new is not None:
            return new
        if old is not None:
            warnings.warn(
                f"{old_name} is deprecated; use {new_name} (PLAN.md P0.2).",
                DeprecationWarning,
                stacklevel=3,
            )
            return old
        return default

    def __len__(self) -> int:
        return len(self._edges)

    def edges(self) -> List[Dict[str, Any]]:
        return [dict(self._edges[key]) for key in sorted(self._edges)]

    def add(
        self,
        edges: Iterable[Mapping[str, Any]],
        *,
        generation_rollout_iteration: Optional[int] = None,
        generation_step: Optional[int] = None,  # deprecated alias
        policy_snapshot_id: str,
    ) -> int:
        """PLAN.md P0.2/P0.3: add a batch of edges to the replay buffer.

        ``generation_rollout_iteration`` stamps replay-age semantics on every
        new edge; ``generation_step`` is kept as a deprecated alias so callers
        that still pass ``self.global_steps`` continue to work while they
        migrate. The buffer stores BOTH keys on each record: the canonical
        ``generation_rollout_iteration`` (used for age) and the legacy
        ``generation_step`` (kept for JSON round-trip stability).
        """
        if generation_rollout_iteration is None and generation_step is None:
            raise ValueError(
                "GearTreeReplayBuffer.add requires generation_rollout_iteration "
                "(PLAN.md P0.2)."
            )
        if generation_rollout_iteration is None:
            warnings.warn(
                "generation_step is deprecated; pass generation_rollout_iteration "
                "instead (PLAN.md P0.2).",
                DeprecationWarning,
                stacklevel=2,
            )
            generation_rollout_iteration = generation_step
        gri = int(generation_rollout_iteration)

        prepared: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for edge in edges:
            record = dict(edge)
            self._validate_edge(record)
            # Canonical name (PLAN.md P0.2) — used for age. Fall back to the
            # deprecated per-edge ``generation_step`` alias when the edge did
            # not stamp the new key itself, so pre-migration record producers
            # keep working during the alias window.
            if "generation_rollout_iteration" in record:
                record["generation_rollout_iteration"] = int(
                    record["generation_rollout_iteration"]
                )
            elif "generation_step" in record:
                record["generation_rollout_iteration"] = int(record["generation_step"])
            else:
                record["generation_rollout_iteration"] = gri
            # Legacy alias kept for JSON round-trip stability.
            record["generation_step"] = int(
                record.get("generation_step", record["generation_rollout_iteration"])
            )
            record["policy_snapshot_id"] = str(
                record.get("policy_snapshot_id", policy_snapshot_id)
            )
            if record["policy_snapshot_id"] != str(policy_snapshot_id):
                raise ValueError(
                    "new edge policy_snapshot_id does not match current rollout snapshot"
                )
            edge_id = str(record["edge_id"])
            # PLAN.md P0.3: duplicate edge_ids indicate that two rollouts
            # collapsed to the same (question, snapshot, path) tuple — either
            # tree_instance_id was not made unique or a caller re-sent the
            # same edge. Silently overwriting a live row would corrupt the
            # per-parent group and mask a rollout-side bug.
            if edge_id in self._edges or edge_id in seen_ids:
                raise ValueError(
                    f"Replay edge {edge_id!r} is already in the buffer; "
                    "duplicate edge_ids indicate the rollout did not assign a "
                    "unique tree_instance_id (PLAN.md P0.3)."
                )
            seen_ids.add(edge_id)
            prepared.append(record)
        # All-or-nothing commit.
        for record in prepared:
            self._edges[str(record["edge_id"])] = record
        self.metrics["added_edges"] += len(prepared)
        return len(prepared)

    @staticmethod
    def _edge_generation_iteration(edge: Mapping[str, Any], default: int) -> int:
        """Return the edge's generation rollout iteration.

        Prefers the canonical ``generation_rollout_iteration`` field
        (PLAN.md P0.2); falls back to the deprecated ``generation_step`` alias
        for JSON round-trips of pre-migration checkpoints.
        """
        value = edge.get("generation_rollout_iteration")
        if value is None:
            value = edge.get("generation_step", default)
        return int(value)

    def expire(
        self,
        *,
        current_rollout_iteration: Optional[int] = None,
        current_step: Optional[int] = None,  # deprecated alias
    ) -> List[str]:
        """PLAN.md P0.2/P0.3: expire an edge when its age (in rollout
        iterations) reaches ``max_edge_age_iterations``. Age is measured in
        rollout iterations only — never in ``global_step``.
        """
        if current_rollout_iteration is None and current_step is None:
            raise ValueError(
                "GearTreeReplayBuffer.expire requires current_rollout_iteration "
                "(PLAN.md P0.2)."
            )
        if current_rollout_iteration is None:
            warnings.warn(
                "current_step is deprecated; pass current_rollout_iteration "
                "instead (PLAN.md P0.2).",
                DeprecationWarning,
                stacklevel=2,
            )
            current_rollout_iteration = current_step
        cri = int(current_rollout_iteration)
        expired = [
            edge_id
            for edge_id, edge in self._edges.items()
            if cri - self._edge_generation_iteration(edge, cri)
            >= self.max_edge_age_iterations
            and edge_id not in self._reserved
        ]
        for edge_id in expired:
            self._edges.pop(edge_id, None)
        self.metrics["expired_edges"] += len(expired)
        return sorted(expired)

    def sample_for_update(
        self,
        *,
        current_rollout_iteration: Optional[int] = None,
        current_step: Optional[int] = None,  # deprecated alias
        remove: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if current_rollout_iteration is None and current_step is None:
            raise ValueError(
                "sample_for_update requires current_rollout_iteration "
                "(PLAN.md P0.2)."
            )
        if current_rollout_iteration is None:
            warnings.warn(
                "current_step is deprecated; pass current_rollout_iteration "
                "instead (PLAN.md P0.2).",
                DeprecationWarning,
                stacklevel=2,
            )
            current_rollout_iteration = current_step
        cri = int(current_rollout_iteration)
        size_before = len(self._edges)
        expired_ids = self.expire(current_rollout_iteration=cri)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge_id, edge in self._edges.items():
            if edge_id in self._reserved:
                continue
            grouped[str(edge["question_id"])].append(edge)

        rng = random.Random(self.sampling_seed + cri)
        candidates: List[Dict[str, Any]] = []
        edges_per_question: List[int] = []
        cap = self.max_edges_per_question_per_iteration
        for question_id in sorted(grouped):
            group = sorted(grouped[question_id], key=lambda item: str(item["edge_id"]))
            if len(group) > cap:
                group = rng.sample(group, cap)
                group = sorted(group, key=lambda item: str(item["edge_id"]))
            candidates.extend(group)
            edges_per_question.append(len(group))

        candidates = sorted(candidates, key=lambda item: str(item["edge_id"]))
        if len(candidates) > self.target_edges_per_iteration:
            sampled = rng.sample(candidates, self.target_edges_per_iteration)
            sampled = sorted(sampled, key=lambda item: str(item["edge_id"]))
        else:
            sampled = candidates

        sampled_ids = {str(edge["edge_id"]) for edge in sampled}
        if remove:
            self.remove(sampled_ids)
        self.metrics["sampled_edges"] += len(sampled)

        ages = [
            cri - self._edge_generation_iteration(edge, cri) for edge in sampled
        ]
        # PLAN.md P0.7: log the observed age histogram instead of asserting a
        # uniform 1/8 composition upstream.
        age_hist: Dict[int, int] = defaultdict(int)
        for a in ages:
            age_hist[int(a)] += 1
        stats = {
            "buffer/size_before": size_before,
            "buffer/expired_edges": len(expired_ids),
            "buffer/candidate_edges": len(candidates),
            "buffer/sampled_edges": len(sampled),
            "buffer/size_after": len(self._edges),
            "buffer/reserved_edges": len(self._reserved),
            "buffer/underfilled": float(len(sampled) < self.target_edges_per_iteration),
            "buffer/unique_questions": len({edge["question_id"] for edge in sampled}),
            "buffer/mean_edge_age": sum(ages) / len(ages) if ages else 0.0,
            "buffer/max_edge_age": max(ages) if ages else 0,
            "buffer/edges_per_question_mean": (
                sum(edges_per_question) / len(edges_per_question) if edges_per_question else 0.0
            ),
            "buffer/edges_per_question_max": max(edges_per_question) if edges_per_question else 0,
            "buffer/edge_age_histogram": dict(age_hist),
            "buffer/resolved_max_edges_per_question_per_iteration": cap,
            "removed_edge_ids": sorted(sampled_ids),
        }
        for depth in (0, 1, 2):
            stats[f"buffer/depth_{depth}_edges"] = sum(
                1 for edge in sampled if int(edge.get("depth", -1)) == depth
            )
        return [dict(edge) for edge in sampled], stats

    def reserve_for_update(
        self,
        *,
        current_rollout_iteration: Optional[int] = None,
        current_step: Optional[int] = None,  # deprecated alias
    ) -> ReplayReservation:
        if current_rollout_iteration is None and current_step is None:
            raise ValueError(
                "reserve_for_update requires current_rollout_iteration "
                "(PLAN.md P0.2)."
            )
        if current_rollout_iteration is None:
            warnings.warn(
                "current_step is deprecated; pass current_rollout_iteration "
                "instead (PLAN.md P0.2).",
                DeprecationWarning,
                stacklevel=2,
            )
            current_rollout_iteration = current_step
        sampled, stats = self.sample_for_update(
            current_rollout_iteration=int(current_rollout_iteration), remove=False
        )
        reservation_id = self._next_reservation_id
        self._next_reservation_id += 1
        edge_ids = tuple(str(edge["edge_id"]) for edge in sampled)
        for edge_id in edge_ids:
            if edge_id in self._reserved:
                raise RuntimeError(f"Replay edge {edge_id!r} is already reserved")
            self._reserved[edge_id] = reservation_id
        self.metrics["reserved_edges"] += len(edge_ids)
        stats["buffer/reservation_id"] = reservation_id
        stats["removed_edge_ids"] = list(edge_ids)
        stats["buffer/reserved_edges"] = len(self._reserved)
        return ReplayReservation(
            reservation_id=reservation_id,
            edge_ids=edge_ids,
            edges=tuple(dict(edge) for edge in sampled),
            stats=stats,
        )

    def reserve_complete_trees_for_update(
        self,
        *,
        current_rollout_iteration: Optional[int] = None,
        current_step: Optional[int] = None,  # deprecated alias
    ) -> ReplayReservation:
        """Non-canonical ``replay_sampling_unit=complete_tree`` ablation.

        Trees are added whole (all edges sharing a ``tree_id``) while the
        cumulative edge count stays within ``target_edges_per_iteration`` —
        the reservation NEVER exceeds the target (PLAN.md P0.A/P0.D); a tree
        that does not fit is skipped, so the reservation may be underfilled.
        ``max_edges_per_question_per_iteration`` is applied per question by
        picking a subset of that question's trees, never by dropping an
        individual edge from one of its trees. A tree is never split across
        two reservations, and no partial parent group is ever returned.
        """
        if current_rollout_iteration is None and current_step is None:
            raise ValueError(
                "reserve_complete_trees_for_update requires "
                "current_rollout_iteration (PLAN.md P0.2)."
            )
        if current_rollout_iteration is None:
            warnings.warn(
                "current_step is deprecated; pass current_rollout_iteration "
                "instead (PLAN.md P0.2).",
                DeprecationWarning,
                stacklevel=2,
            )
            current_rollout_iteration = current_step
        cri = int(current_rollout_iteration)
        size_before = len(self._edges)
        expired_ids = self.expire(current_rollout_iteration=cri)
        rng = random.Random(self.sampling_seed + cri)

        # Group available (non-reserved) edges by (question_id, tree_id).
        by_question: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for edge_id, edge in self._edges.items():
            if edge_id in self._reserved:
                continue
            qid = str(edge.get("question_id", ""))
            tid = str(
                edge.get("tree_id")
                or edge.get("gear_segment_id")
                or edge_id
            )
            by_question[qid][tid].append(edge)

        # Materialise per-question tree lists, respecting per-question edge caps.
        picked_trees: List[List[Dict[str, Any]]] = []
        edges_per_question: List[int] = []
        for qid in sorted(by_question):
            trees = by_question[qid]
            # Deterministic ordering within a question by tree_id.
            tree_ids = sorted(trees)
            # Pack whole trees until we exceed the resolved per-question cap,
            # keeping the last one added if it's the first one for this
            # question so we never starve a question that only has one large
            # tree.
            cap = self.max_edges_per_question_per_iteration
            selected: List[List[Dict[str, Any]]] = []
            cumulative = 0
            for tid in tree_ids:
                tree_edges = sorted(
                    trees[tid], key=lambda e: str(e["edge_id"])
                )
                if not selected:
                    selected.append(tree_edges)
                    cumulative += len(tree_edges)
                    continue
                if cumulative + len(tree_edges) > cap:
                    continue
                selected.append(tree_edges)
                cumulative += len(tree_edges)
            picked_trees.extend(selected)
            edges_per_question.append(cumulative)

        # Shuffle whole trees, then pack whole trees WITHOUT ever exceeding
        # the per-iteration target (PLAN.md P0.A/P0.D: never return more than
        # target_edges_per_iteration). A tree that does not fit is skipped so
        # a smaller tree later in the shuffle may still fill the remainder;
        # trees are never split.
        rng.shuffle(picked_trees)
        packed_trees: List[List[Dict[str, Any]]] = []
        skipped_oversized_trees = 0
        cumulative = 0
        for tree_edges in picked_trees:
            if cumulative + len(tree_edges) > self.target_edges_per_iteration:
                skipped_oversized_trees += 1
                continue
            packed_trees.append(tree_edges)
            cumulative += len(tree_edges)

        # Flatten and reserve.
        sampled: List[Dict[str, Any]] = [e for tree in packed_trees for e in tree]
        sampled = sorted(sampled, key=lambda e: str(e["edge_id"]))
        # PLAN.md P0.N6: assert no partial parent group.
        _assert_complete_parent_groups(sampled)

        reservation_id = self._next_reservation_id
        self._next_reservation_id += 1
        edge_ids = tuple(str(edge["edge_id"]) for edge in sampled)
        for edge_id in edge_ids:
            if edge_id in self._reserved:
                raise RuntimeError(f"Replay edge {edge_id!r} is already reserved")
            self._reserved[edge_id] = reservation_id
        self.metrics["reserved_edges"] += len(edge_ids)
        self.metrics["sampled_edges"] += len(sampled)

        ages = [
            cri - self._edge_generation_iteration(edge, cri)
            for edge in sampled
        ]
        age_hist: Dict[int, int] = defaultdict(int)
        for a in ages:
            age_hist[int(a)] += 1
        stats = {
            "buffer/size_before": size_before,
            "buffer/expired_edges": len(expired_ids),
            "buffer/candidate_trees": len(picked_trees),
            "buffer/packed_trees": len(packed_trees),
            "buffer/skipped_oversized_trees": skipped_oversized_trees,
            "buffer/sampled_edges": len(sampled),
            "buffer/size_after": len(self._edges),
            "buffer/reserved_edges": len(self._reserved),
            "buffer/underfilled": float(len(sampled) < self.target_edges_per_iteration),
            "buffer/edge_age_histogram": dict(age_hist),
            "buffer/resolved_max_edges_per_question_per_iteration": (
                self.max_edges_per_question_per_iteration
            ),
            "buffer/unique_questions": len({edge["question_id"] for edge in sampled}),
            "buffer/unique_trees": len(packed_trees),
            "buffer/mean_edge_age": sum(ages) / len(ages) if ages else 0.0,
            "buffer/max_edge_age": max(ages) if ages else 0,
            "buffer/edges_per_question_mean": (
                sum(edges_per_question) / len(edges_per_question)
                if edges_per_question
                else 0.0
            ),
            "buffer/edges_per_question_max": (
                max(edges_per_question) if edges_per_question else 0
            ),
            "buffer/reservation_id": reservation_id,
            "removed_edge_ids": list(edge_ids),
        }
        return ReplayReservation(
            reservation_id=reservation_id,
            edge_ids=edge_ids,
            edges=tuple(dict(edge) for edge in sampled),
            stats=stats,
        )

    def commit(self, reservation: ReplayReservation) -> List[str]:
        self._check_reservation(reservation)
        removed = self.remove(reservation.edge_ids)
        for edge_id in reservation.edge_ids:
            self._reserved.pop(edge_id, None)
        self.metrics["committed_edges"] += len(removed)
        if sorted(removed) != sorted(reservation.edge_ids):
            raise RuntimeError("Replay commit did not remove exactly the reserved edges")
        return removed

    def rollback(self, reservation: ReplayReservation) -> None:
        self._check_reservation(reservation)
        for edge_id in reservation.edge_ids:
            self._reserved.pop(edge_id, None)
        self.metrics["rolled_back_edges"] += len(reservation.edge_ids)

    def remove(self, edge_ids: Iterable[str]) -> List[str]:
        removed: List[str] = []
        for edge_id in sorted(str(edge_id) for edge_id in edge_ids):
            if self._edges.pop(edge_id, None) is not None:
                self._reserved.pop(edge_id, None)
                removed.append(edge_id)
        return removed

    def _check_reservation(self, reservation: ReplayReservation) -> None:
        mismatched = [
            edge_id
            for edge_id in reservation.edge_ids
            if self._reserved.get(edge_id) != reservation.reservation_id
        ]
        if mismatched:
            raise RuntimeError(f"Replay reservation is not active for edges: {mismatched}")

    def save(self, checkpoint_dir: str | Path) -> None:
        target = Path(checkpoint_dir)
        target.mkdir(parents=True, exist_ok=True)
        edge_path = target / "gear_tree_replay_buffer.jsonl"
        meta_path = target / "gear_tree_replay_buffer_meta.json"
        tmp_edge = edge_path.with_suffix(edge_path.suffix + ".tmp")
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with tmp_edge.open("w", encoding="utf-8") as handle:
            for edge in self.edges():
                handle.write(json.dumps(edge, sort_keys=True) + "\n")
        tmp_meta.write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    # PLAN.md P0.2 canonical names.
                    "target_edges_per_iteration": self.target_edges_per_iteration,
                    "max_edge_age_iterations": self.max_edge_age_iterations,
                    "max_edges_per_question_per_iteration": (
                        self.max_edges_per_question_per_iteration
                    ),
                    "resolved_max_edges_per_question_per_iteration": (
                        self.resolved_max_edges_per_question_per_iteration
                    ),
                    "max_edges_per_question_cap_source": (
                        self.max_edges_per_question_cap_source
                    ),
                    "tree_shape": list(self.tree_shape),
                    "trees_per_question": self.trees_per_question,
                    "replay_sampling_unit": self.replay_sampling_unit,
                    # Deprecated aliases kept for reader tooling.
                    "target_edges_per_update": self.target_edges_per_update,
                    "max_edges_per_question": self.max_edges_per_question,
                    "max_edge_age": self.max_edge_age,
                    "underfill_policy": self.underfill_policy,
                    "sampling_seed": self.sampling_seed,
                    "metrics": self.metrics,
                    "next_reservation_id": self._next_reservation_id,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        tmp_edge.replace(edge_path)
        tmp_meta.replace(meta_path)

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "GearTreeReplayBuffer":
        source = Path(checkpoint_dir)
        meta = json.loads((source / "gear_tree_replay_buffer_meta.json").read_text(encoding="utf-8"))
        # PLAN.md P0.2: prefer the new canonical field names when present so
        # that a resume-after-migration reads the new schema first; fall back
        # to the deprecated aliases for older checkpoints.
        target_iter = meta.get(
            "target_edges_per_iteration", meta.get("target_edges_per_update")
        )
        max_age_iter = meta.get(
            "max_edge_age_iterations", meta.get("max_edge_age")
        )
        cap = meta.get(
            "max_edges_per_question_per_iteration",
            meta.get("max_edges_per_question"),
        )
        buffer = cls(
            target_edges_per_iteration=target_iter,
            max_edge_age_iterations=max_age_iter,
            max_edges_per_question_per_iteration=cap,
            replay_sampling_unit=meta.get("replay_sampling_unit", "edge"),
            tree_shape=meta.get("tree_shape") or None,
            trees_per_question=int(meta.get("trees_per_question", 1) or 1),
            underfill_policy=meta.get("underfill_policy", "use_available"),
            sampling_seed=meta.get("sampling_seed", 0),
        )
        buffer.metrics = dict(meta.get("metrics", {}))
        buffer._next_reservation_id = int(meta.get("next_reservation_id", 1))
        edge_file = source / "gear_tree_replay_buffer.jsonl"
        if edge_file.exists():
            for line in edge_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                edge = json.loads(line)
                buffer._validate_edge(edge)
                buffer._edges[str(edge["edge_id"])] = edge
        return buffer

    @staticmethod
    def _validate_edge(edge: MutableMapping[str, Any]) -> None:
        if is_ledger_slot(edge):
            # PLAN.md §1.2: metadata-only logical slot for an exact-zero
            # advantage segment. Strict bookkeeping, no trainable payload.
            missing = [f for f in _REQUIRED_SLOT_FIELDS if f not in edge]
            if missing:
                raise ValueError(
                    f"Replay logical slot is missing required fields: {missing}"
                )
            if float(edge["advantage"]) != 0.0:
                raise ValueError(
                    "Replay logical slot must carry exactly zero advantage "
                    f"(PLAN.md §1.2); got {edge['advantage']!r}."
                )
            if not bool(edge.get("advantage_is_zero", False)):
                raise ValueError(
                    "Replay logical slot must stamp advantage_is_zero=True "
                    "(PLAN.md §1.2)."
                )
            if int(edge.get("response_token_count", 0) or 0) <= 0:
                raise ValueError(
                    "Replay logical slot must record its positive pre-filter "
                    "response_token_count — T_B cannot be reconstructed "
                    "otherwise (PLAN.md §1.2)."
                )
            payload = [
                f for f in _SLOT_FORBIDDEN_PAYLOAD_FIELDS if edge.get(f)
            ]
            if payload:
                raise ValueError(
                    "Replay logical slot must be metadata-only; found "
                    f"trainable payload fields {payload} (PLAN.md §1.2)."
                )
            return
        missing = [field for field in _REQUIRED_EDGE_FIELDS if field not in edge]
        if missing:
            raise ValueError(f"Replay edge is missing required fields: {missing}")
        response_len = len(edge.get("response_token_ids") or [])
        logprob_len = len(edge.get("actor_shifted_log_probs") or [])
        if response_len <= 0:
            raise ValueError("Replay edge has no response_token_ids")
        if logprob_len != response_len:
            raise ValueError(
                "Replay edge actor_shifted_log_probs must align one-to-one with response_token_ids"
            )


def _assert_complete_parent_groups(edges: Iterable[Mapping[str, Any]]) -> None:
    """PLAN.md P0.N6: reservation must not split a parent group.

    We consider a parent group complete when every edge belonging to that
    ``parent_group_id`` in the currently available snapshot is present. The
    check is intentionally cheap: it groups by parent_group_id + allocated_k
    and verifies row_count matches the stamped allocated_k (fresh_iid) or
    that every row of the group has the same allocated_k (weighted_reuse).
    """

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        pgid = str(edge.get("parent_group_id", ""))
        if not pgid:
            continue
        groups[pgid].append(edge)
    for pgid, group in groups.items():
        allocated_values = {int(e.get("allocated_k", 0) or 0) for e in group}
        if len(allocated_values) != 1:
            raise RuntimeError(
                f"replay reservation split parent group {pgid!r}: "
                f"inconsistent allocated_k={allocated_values}"
            )
        allocated_k = next(iter(allocated_values), 0)
        multiplicities = [int(e.get("sample_multiplicity", 1) or 1) for e in group]
        if all(m == 1 for m in multiplicities) and allocated_k > 0:
            if len(group) != allocated_k:
                raise RuntimeError(
                    f"replay reservation returned a partial parent group "
                    f"{pgid!r}: {len(group)} rows for allocated_k={allocated_k}"
                )


def pack_edges_into_microbatches(
    edges: Iterable[Mapping[str, Any]],
    *,
    micro_batch_size: int,
) -> List[List[Mapping[str, Any]]]:
    """PLAN.md P0.N6: group-aware microbatch packing.

    Guarantees:
      * a parent group (all edges with the same ``parent_group_id``) is
        placed entirely in one microbatch;
      * where possible, edges of the same ``tree_id`` are kept together;
      * a microbatch never exceeds ``micro_batch_size`` unless a single
        parent group is larger than the limit — in that case, the parent
        group is placed in its own microbatch (a well-formed VDRA config
        cannot have this happen; we surface it explicitly instead).
    """

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    # First group by parent, then by tree (parent -> tree map).
    parents: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    parent_to_tree: Dict[str, str] = {}
    for edge in edges:
        pgid = str(edge.get("parent_group_id", ""))
        if not pgid:
            # A row without a parent group cannot be packed group-aware; put
            # it in a synthetic single-row group keyed by its own edge_id so
            # sorting stays deterministic.
            pgid = f"__row__:{edge.get('edge_id')}"
        parents[pgid].append(edge)
        parent_to_tree.setdefault(pgid, str(edge.get("tree_id", "")))

    # Then order parents by tree so trees stay together in a microbatch.
    ordered_parents = sorted(
        parents.keys(), key=lambda pid: (parent_to_tree.get(pid, ""), pid)
    )

    packed: List[List[Mapping[str, Any]]] = []
    current: List[Mapping[str, Any]] = []
    for pgid in ordered_parents:
        group = parents[pgid]
        if len(group) > micro_batch_size:
            # Emit any pending microbatch, then place the oversize group alone.
            if current:
                packed.append(current)
                current = []
            packed.append(list(group))
            continue
        if len(current) + len(group) > micro_batch_size:
            packed.append(current)
            current = []
        current.extend(group)
    if current:
        packed.append(current)
    return packed
