"""Online VDRA queue and shared residual-budget helpers."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, List, MutableMapping, Optional

from .core import AllocationSummary, allocate_branch_factors
from .logging_schema import node_id, write_node_accounting


@dataclass
class SharedReservePool:
    queue_count: int
    value: int = 0
    contributed: int = 0
    consumed: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, amount: int) -> int:
        amount = max(int(amount), 0)
        async with self._lock:
            self.value += amount
            self.contributed += amount
            return self.value

    async def draw_queue_share(self, max_amount: Optional[int] = None) -> int:
        async with self._lock:
            if self.value <= 0:
                return 0
            queue_share = int(math.ceil(self.value / max(self.queue_count, 1)))
            amount = min(queue_share, self.value)
            if max_amount is not None:
                amount = min(amount, max(int(max_amount), 0))
            self.value -= amount
            return amount

    async def consume(self, amount: int) -> int:
        amount = max(int(amount), 0)
        async with self._lock:
            self.consumed += amount
            return self.consumed


@dataclass
class OnlineQueueItem:
    node: MutableMapping[str, Any]
    default_branch_factor: int
    depth: int
    weight_key: Optional[str] = None
    policy_snapshot_id: Optional[str] = None
    completion_future: Optional[asyncio.Future] = None


@dataclass
class OnlineBudgetQueue:
    queue_id: int
    capacity: int
    policy_snapshot_id: str
    items: List[OnlineQueueItem] = field(default_factory=list)
    first_enqueued_at: Optional[float] = None

    def append(self, item: OnlineQueueItem, now: Optional[float] = None) -> None:
        snapshot = item.policy_snapshot_id or self.policy_snapshot_id
        if snapshot != self.policy_snapshot_id:
            raise RuntimeError(
                f"VDRA queue {self.queue_id} cannot mix policy snapshots: "
                f"{self.policy_snapshot_id!r} != {snapshot!r}"
            )
        if not self.items:
            self.first_enqueued_at = time.monotonic() if now is None else now
        self.items.append(item)
        item.node["vdra_queue_id"] = self.queue_id
        item.node["vdra_policy_snapshot_id"] = self.policy_snapshot_id

    def flush_reason(
        self, timeout_seconds: float, now: Optional[float] = None
    ) -> Optional[str]:
        if not self.items:
            return None
        if len(self.items) >= self.capacity:
            return "capacity"
        current = time.monotonic() if now is None else now
        if (
            timeout_seconds > 0
            and self.first_enqueued_at is not None
            and current - self.first_enqueued_at >= timeout_seconds
        ):
            return "timeout"
        return None

    def pop_all(self) -> List[OnlineQueueItem]:
        items = list(self.items)
        self.items.clear()
        self.first_enqueued_at = None
        return items


@dataclass
class QueueFlushResult:
    queue_id: int
    items: List[OnlineQueueItem]
    summary: AllocationSummary
    reserve_draw: int
    base_budget: int
    total_budget: int
    flush_reason: str
    queue_wait_seconds: float
    reserve_available_at_flush: int
    reserve_after_flush: int
    allocation_seconds: float

    @property
    def timed_out(self) -> bool:
        return self.flush_reason == "timeout"

    @property
    def reserve_consumed(self) -> int:
        return int(sum(self.summary.additional_allocations.values()))

    @property
    def unallocated_residual_budget(self) -> int:
        return max(int(self.reserve_draw) - self.reserve_consumed, 0)

    def to_record(self) -> dict:
        return {
            "queue_id": self.queue_id,
            "policy_snapshot_id": self.items[0].policy_snapshot_id if self.items else None,
            "flush_reason": self.flush_reason,
            "queue_wait_seconds": self.queue_wait_seconds,
            "queue_size_at_flush": len(self.items),
            "default_queue_budget": self.base_budget,
            "total_saved_budget": int(sum(self.summary.saved_allocations.values())),
            "total_unmet_demand": int(sum(self.summary.unmet_demands.values())),
            "reserve_before_flush": self.reserve_available_at_flush,
            "reserve_drawn": self.reserve_draw,
            "reserve_after_flush": self.reserve_after_flush,
            "allocated_residual_budget": self.reserve_consumed,
            "unallocated_residual_budget": self.unallocated_residual_budget,
            "allocation_seconds": self.allocation_seconds,
        }


class RootQueueManager:
    """Long-lived online queues for one frozen-policy rollout iteration."""

    def __init__(
        self,
        *,
        queue_count: int,
        queue_capacity: int = 8,
        timeout_seconds: float,
        reserve_pool: SharedReservePool,
        n_min: int = 1,
        use_residual_budget: bool = True,
        policy_snapshot_id: str = "current",
        strict_vdra: bool = True,
        rounding_strategy: str = "integer_marginal",
        rounding_seed: int = 0,
        lambda_: Optional[float] = None,
    ):
        self.queue_count = max(int(queue_count), 1)
        self.queue_capacity = max(int(queue_capacity), 1)
        self.timeout_seconds = max(float(timeout_seconds), 0.0)
        self.reserve_pool = reserve_pool
        self.n_min = max(int(n_min), 0)
        self.use_residual_budget = bool(use_residual_budget)
        self.policy_snapshot_id = str(policy_snapshot_id)
        self.strict_vdra = bool(strict_vdra)
        self.rounding_strategy = rounding_strategy
        self.rounding_seed = int(rounding_seed)
        self.queues = [
            OnlineBudgetQueue(i, self.queue_capacity, self.policy_snapshot_id)
            for i in range(self.queue_count)
        ]
        self.flush_count = 0
        self.timeout_flush_count = 0
        self.capacity_flush_count = 0
        self.final_drain_count = 0
        self.reserve_consumed = 0

    def enqueue(self, item: OnlineQueueItem, now: Optional[float] = None) -> None:
        queue = min(self.queues, key=lambda q: (len(q.items), q.queue_id))
        queue.append(item, now=now)

    async def flush_ready(self, now: Optional[float] = None) -> List[QueueFlushResult]:
        results: List[QueueFlushResult] = []
        for queue in self.queues:
            reason = queue.flush_reason(self.timeout_seconds, now=now)
            if reason:
                result = await self._flush_queue(queue, reason=reason, now=now)
                if result:
                    results.append(result)
        return results

    async def drain(self, now: Optional[float] = None) -> List[QueueFlushResult]:
        results: List[QueueFlushResult] = []
        for queue in self.queues:
            result = await self._flush_queue(queue, reason="final_drain", now=now)
            if result:
                results.append(result)
        return results

    async def _flush_queue(
        self, queue: OnlineBudgetQueue, *, reason: str, now: Optional[float]
    ) -> Optional[QueueFlushResult]:
        if not queue.items:
            return None
        current = time.monotonic() if now is None else now
        wait = max(current - (queue.first_enqueued_at or current), 0.0)
        items = queue.pop_all()
        reserve_before = self.reserve_pool.value
        for item in items:
            item.node["vdra_default_k"] = int(item.default_branch_factor)
            if "vdra_predicted_k" not in item.node:
                item.node["vdra_predicted_k"] = int(
                    item.node.get("gear_predicted_k", item.default_branch_factor)
                )
        base_budget = sum(max(int(item.default_branch_factor), self.n_min) for item in items)
        # Unified allocation keeps the queue budget exact. Pruning and expansion
        # are outputs of the same integer solve, so the old reserve pool is no
        # longer an input to allocation.
        reserve_draw = 0
        total_budget = base_budget
        weight_key = (
            "gear_allocation_weight_override"
            if any(item.weight_key == "gear_allocation_weight_override" for item in items)
            else None
        )
        t0 = time.perf_counter()
        summary = allocate_branch_factors(
            [item.node for item in items],
            total_budget=total_budget,
            n_min=self.n_min,
            weight_key=weight_key,
            strict=self.strict_vdra,
            rounding_strategy=self.rounding_strategy,
            rounding_seed=self.rounding_seed,
        )
        allocation_seconds = time.perf_counter() - t0
        for idx, item in enumerate(items):
            key = node_id(item.node, idx)
            write_node_accounting(
                item.node,
                default_k=item.default_branch_factor,
                predicted_k=int(item.node["vdra_predicted_k"]),
                allocated_k=summary.allocations[key],
                k_min=self.n_min,
                dispersion_C=float(item.node.get("vdra_dispersion_C", item.node.get("gear_reward_variance", 0.0)) or 0.0),
                allocation_weight=summary.weights[key],
            )
            item.node["vdra_queue_wait_seconds"] = wait
            item.node["vdra_flush_reason"] = reason
        self.flush_count += 1
        self.timeout_flush_count += int(reason == "timeout")
        self.capacity_flush_count += int(reason == "capacity")
        self.final_drain_count += int(reason == "final_drain")
        residual_used = int(summary.transferred_budget)
        self.reserve_consumed += residual_used
        reserve_after = self.reserve_pool.value
        return QueueFlushResult(
            queue_id=queue.queue_id,
            items=items,
            summary=summary,
            reserve_draw=reserve_draw,
            base_budget=base_budget,
            total_budget=total_budget,
            flush_reason=reason,
            queue_wait_seconds=wait,
            reserve_available_at_flush=reserve_before,
            reserve_after_flush=reserve_after,
            allocation_seconds=allocation_seconds,
        )
