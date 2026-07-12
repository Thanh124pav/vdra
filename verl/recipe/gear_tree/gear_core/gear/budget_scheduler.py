"""Small flexible queue scheduler for budget-allocation nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Sequence

from .budget_allocation import AllocationSummary, allocate_branch_factors


@dataclass
class BudgetQueue:
    queue_id: int
    nodes: List[Any] = field(default_factory=list)
    active: bool = False


class FlexibleBudgetScheduler:
    """Allocate ready nodes in small queues.

    This synchronous scheduler models the flexible policy at allocation time:
    ready nodes are packed into the least-loaded non-active queue and each
    queue receives a proportional depth budget.  The async waiting/timeout is
    handled by the caller; this class keeps the budget math and queue metadata
    deterministic and testable.
    """

    def __init__(
        self,
        *,
        queue_count: int = 2,
        lambda_: float = 0.02,
        n_min: int = 0,
    ):
        self.queues = [BudgetQueue(queue_id=i) for i in range(max(int(queue_count), 1))]
        self.lambda_ = float(lambda_)
        self.n_min = max(int(n_min), 0)

    def allocate(
        self,
        nodes: Sequence[Any],
        *,
        total_depth_budget: int,
    ) -> List[AllocationSummary]:
        for queue in self.queues:
            queue.nodes.clear()
            queue.active = False
        if not nodes:
            return []
        for node in nodes:
            queue = min(self.queues, key=lambda q: (q.active, len(q.nodes), q.queue_id))
            queue.nodes.append(node)
            if isinstance(node, dict):
                node["gear_budget_queue_id"] = queue.queue_id

        summaries: List[AllocationSummary] = []
        total_nodes = len(nodes)
        for queue in self.queues:
            if not queue.nodes:
                continue
            queue_budget = int(math.floor(total_depth_budget * len(queue.nodes) / max(total_nodes, 1)))
            summary = allocate_branch_factors(
                queue.nodes,
                total_budget=queue_budget,
                lambda_=self.lambda_,
                n_min=self.n_min,
            )
            summaries.append(summary)
        return summaries
