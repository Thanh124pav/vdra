"""Online reserve and queue helpers for GEAR tree construction."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional, Sequence

from treetune.gear.budget_allocation import AllocationSummary, allocate_branch_factors


@dataclass
class SharedReservePool:
    """Minibatch-level reserve pool shared by concurrently-built trees."""

    queue_count: int
    value: int = 0
    contributed: int = 0
    consumed: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, amount: int) -> int:
        amount = max(int(amount), 0)
        if amount <= 0:
            return self.value
        async with self._lock:
            self.value += amount
            self.contributed += amount
            return self.value

    async def draw_queue_share(self) -> int:
        async with self._lock:
            if self.value <= 0:
                return 0
            requested = int(math.ceil(self.value / max(int(self.queue_count), 1)))
            drawn = min(requested, self.value)
            self.value -= drawn
            self.consumed += drawn
            return drawn


@dataclass
class OnlineQueueItem:
    node: MutableMapping[str, Any]
    default_branch_factor: int
    depth: int
    weight_key: Optional[str] = None


@dataclass
class OnlineBudgetQueue:
    queue_id: int
    items: List[OnlineQueueItem] = field(default_factory=list)
    first_enqueued_at: Optional[float] = None

    def append(self, item: OnlineQueueItem, now: Optional[float] = None) -> None:
        if not self.items:
            self.first_enqueued_at = time.monotonic() if now is None else now
        self.items.append(item)
        item.node["gear_budget_queue_id"] = self.queue_id

    def should_flush(self, timeout_seconds: float, now: Optional[float] = None) -> bool:
        if not self.items:
            return False
        if timeout_seconds <= 0:
            return True
        if self.first_enqueued_at is None:
            return False
        current = time.monotonic() if now is None else now
        return current - self.first_enqueued_at >= timeout_seconds

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
    timed_out: bool = False


class RootQueueManager:
    """Queue manager for one root prompt.

    Queues are local to a tree/root, so they can mix depths for that root but
    cannot mix nodes from different prompts.  Reserve accounting is shared via
    ``SharedReservePool``.
    """

    def __init__(
        self,
        *,
        queue_count: int,
        timeout_seconds: float,
        reserve_pool: SharedReservePool,
        lambda_: float,
        n_min: int = 0,
        use_residual_budget: bool = True,
    ):
        self.queue_count = max(int(queue_count), 1)
        self.timeout_seconds = max(float(timeout_seconds), 0.0)
        self.reserve_pool = reserve_pool
        self.lambda_ = float(lambda_)
        self.n_min = max(int(n_min), 0)
        self.use_residual_budget = bool(use_residual_budget)
        self.queues = [OnlineBudgetQueue(queue_id=i) for i in range(self.queue_count)]
        self.flush_count = 0
        self.timeout_flush_count = 0
        self.reserve_consumed = 0

    def enqueue(self, item: OnlineQueueItem, now: Optional[float] = None) -> None:
        queue = min(self.queues, key=lambda q: (len(q.items), q.queue_id))
        queue.append(item, now=now)

    async def flush_ready(self, now: Optional[float] = None) -> List[QueueFlushResult]:
        results: List[QueueFlushResult] = []
        for queue in self.queues:
            if queue.should_flush(self.timeout_seconds, now=now):
                result = await self._flush_queue(queue, timed_out=True)
                if result is not None:
                    results.append(result)
        return results

    async def drain(self) -> List[QueueFlushResult]:
        results: List[QueueFlushResult] = []
        for queue in self.queues:
            result = await self._flush_queue(queue, timed_out=False)
            if result is not None:
                results.append(result)
        return results

    async def _flush_queue(
        self, queue: OnlineBudgetQueue, *, timed_out: bool
    ) -> Optional[QueueFlushResult]:
        items = queue.pop_all()
        if not items:
            return None
        reserve_draw = (
            await self.reserve_pool.draw_queue_share()
            if self.use_residual_budget
            else 0
        )
        base_budget = sum(max(int(item.default_branch_factor), 0) for item in items)
        total_budget = base_budget + reserve_draw
        nodes = [item.node for item in items]
        weight_key = (
            "gear_allocation_weight_override"
            if any(item.weight_key == "gear_allocation_weight_override" for item in items)
            else None
        )
        summary = allocate_branch_factors(
            nodes,
            total_budget=total_budget,
            lambda_=self.lambda_,
            n_min=self.n_min,
            distribute_remainder=True,
            weight_key=weight_key,
            fallback_uniform=True,
        )
        self.flush_count += 1
        if timed_out:
            self.timeout_flush_count += 1
        self.reserve_consumed += reserve_draw
        return QueueFlushResult(
            queue_id=queue.queue_id,
            items=items,
            summary=summary,
            reserve_draw=reserve_draw,
            base_budget=base_budget,
            total_budget=total_budget,
            timed_out=timed_out,
        )
