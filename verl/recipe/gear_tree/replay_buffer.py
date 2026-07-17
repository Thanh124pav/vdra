"""Trainer-owned replay buffer for tree-family VERL edge updates."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


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


@dataclass(frozen=True)
class ReplayReservation:
    reservation_id: int
    edge_ids: Tuple[str, ...]
    edges: Tuple[Dict[str, Any], ...]
    stats: Dict[str, Any]


class GearTreeReplayBuffer:
    """CPU-native edge replay buffer shared by SPO/VDRA tree methods."""

    schema_version = 1

    def __init__(
        self,
        *,
        target_edges_per_update: int = 512,
        max_edges_per_question: int = 32,
        max_edge_age: int = 8,
        underfill_policy: str = "use_available",
        sampling_seed: int = 0,
    ) -> None:
        self.target_edges_per_update = max(int(target_edges_per_update), 1)
        self.max_edges_per_question = max(int(max_edges_per_question), 1)
        self.max_edge_age = max(int(max_edge_age), 1)
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

    def __len__(self) -> int:
        return len(self._edges)

    def edges(self) -> List[Dict[str, Any]]:
        return [dict(self._edges[key]) for key in sorted(self._edges)]

    def add(
        self,
        edges: Iterable[Mapping[str, Any]],
        *,
        generation_step: int,
        policy_snapshot_id: str,
    ) -> int:
        count = 0
        for edge in edges:
            record = dict(edge)
            self._validate_edge(record)
            record["generation_step"] = int(record.get("generation_step", generation_step))
            record["policy_snapshot_id"] = str(record.get("policy_snapshot_id", policy_snapshot_id))
            if record["policy_snapshot_id"] != str(policy_snapshot_id):
                raise ValueError(
                    "new edge policy_snapshot_id does not match current rollout snapshot"
                )
            self._edges[str(record["edge_id"])] = record
            count += 1
        self.metrics["added_edges"] += count
        return count

    def expire(self, *, current_step: int) -> List[str]:
        expired = [
            edge_id
            for edge_id, edge in self._edges.items()
            if int(current_step) - int(edge.get("generation_step", current_step)) >= self.max_edge_age
            and edge_id not in self._reserved
        ]
        for edge_id in expired:
            self._edges.pop(edge_id, None)
        self.metrics["expired_edges"] += len(expired)
        return sorted(expired)

    def sample_for_update(
        self, *, current_step: int, remove: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        size_before = len(self._edges)
        expired_ids = self.expire(current_step=current_step)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge_id, edge in self._edges.items():
            if edge_id in self._reserved:
                continue
            grouped[str(edge["question_id"])].append(edge)

        rng = random.Random(self.sampling_seed + int(current_step))
        candidates: List[Dict[str, Any]] = []
        edges_per_question: List[int] = []
        for question_id in sorted(grouped):
            group = sorted(grouped[question_id], key=lambda item: str(item["edge_id"]))
            if len(group) > self.max_edges_per_question:
                group = rng.sample(group, self.max_edges_per_question)
                group = sorted(group, key=lambda item: str(item["edge_id"]))
            candidates.extend(group)
            edges_per_question.append(len(group))

        candidates = sorted(candidates, key=lambda item: str(item["edge_id"]))
        if len(candidates) > self.target_edges_per_update:
            sampled = rng.sample(candidates, self.target_edges_per_update)
            sampled = sorted(sampled, key=lambda item: str(item["edge_id"]))
        else:
            sampled = candidates

        sampled_ids = {str(edge["edge_id"]) for edge in sampled}
        if remove:
            self.remove(sampled_ids)
        self.metrics["sampled_edges"] += len(sampled)

        ages = [int(current_step) - int(edge.get("generation_step", current_step)) for edge in sampled]
        stats = {
            "buffer/size_before": size_before,
            "buffer/expired_edges": len(expired_ids),
            "buffer/candidate_edges": len(candidates),
            "buffer/sampled_edges": len(sampled),
            "buffer/size_after": len(self._edges),
            "buffer/reserved_edges": len(self._reserved),
            "buffer/underfilled": float(len(sampled) < self.target_edges_per_update),
            "buffer/unique_questions": len({edge["question_id"] for edge in sampled}),
            "buffer/mean_edge_age": sum(ages) / len(ages) if ages else 0.0,
            "buffer/max_edge_age": max(ages) if ages else 0,
            "buffer/edges_per_question_mean": (
                sum(edges_per_question) / len(edges_per_question) if edges_per_question else 0.0
            ),
            "buffer/edges_per_question_max": max(edges_per_question) if edges_per_question else 0,
            "removed_edge_ids": sorted(sampled_ids),
        }
        for depth in (0, 1, 2):
            stats[f"buffer/depth_{depth}_edges"] = sum(
                1 for edge in sampled if int(edge.get("depth", -1)) == depth
            )
        return [dict(edge) for edge in sampled], stats

    def reserve_for_update(self, *, current_step: int) -> ReplayReservation:
        sampled, stats = self.sample_for_update(current_step=current_step, remove=False)
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
        buffer = cls(
            target_edges_per_update=meta["target_edges_per_update"],
            max_edges_per_question=meta["max_edges_per_question"],
            max_edge_age=meta["max_edge_age"],
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
