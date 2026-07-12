"""Deterministic queue partitioning for synchronous VDRA callers."""

from dataclasses import dataclass, field
from typing import Any, List, Sequence

from .core import AllocationSummary, allocate_branch_factors


@dataclass
class BudgetQueue:
    queue_id: int
    nodes: List[Any] = field(default_factory=list)


class FlexibleBudgetScheduler:
    def __init__(self, *, queue_count: int = 2, n_min: int = 1, strict_vdra: bool = True):
        self.queues = [BudgetQueue(i) for i in range(max(int(queue_count), 1))]
        self.n_min = max(int(n_min), 0)
        self.strict_vdra = bool(strict_vdra)

    def allocate(
        self, nodes: Sequence[Any], *, total_depth_budget: int
    ) -> List[AllocationSummary]:
        for queue in self.queues:
            queue.nodes.clear()
        for node in nodes:
            queue = min(self.queues, key=lambda item: (len(item.nodes), item.queue_id))
            queue.nodes.append(node)
            if isinstance(node, dict):
                node["vdra_queue_id"] = queue.queue_id
        summaries: List[AllocationSummary] = []
        remaining_budget = max(int(total_depth_budget), 0)
        remaining_nodes = len(nodes)
        for queue in self.queues:
            if not queue.nodes:
                continue
            queue_budget = round(
                remaining_budget * len(queue.nodes) / max(remaining_nodes, 1)
            )
            summaries.append(
                allocate_branch_factors(
                    queue.nodes,
                    total_budget=queue_budget,
                    n_min=self.n_min,
                    strict=self.strict_vdra,
                )
            )
            remaining_budget -= queue_budget
            remaining_nodes -= len(queue.nodes)
        return summaries
