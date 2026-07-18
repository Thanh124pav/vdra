import pytest

from treetune.gear.budget_scheduler import FlexibleBudgetScheduler


def _node(name, default_k, predicted_k, dispersion_C):
    return {
        "vdra_node_id": name,
        "vdra_default_k": default_k,
        "vdra_predicted_k": predicted_k,
        "vdra_dispersion_C": dispersion_C,
    }


def test_flexible_scheduler_marks_queue_ids_and_preserves_budget():
    nodes = [
        _node("a", 1, 5, 0.1),
        _node("b", 1, 5, 0.5),
        _node("c", 1, 5, 1.0),
    ]
    summaries = FlexibleBudgetScheduler(queue_count=2).allocate(
        nodes, total_depth_budget=9
    )
    assert all("vdra_queue_id" in node for node in nodes)
    assert sum(summary.allocated_budget for summary in summaries) == 9


def test_flexible_scheduler_reports_slack_on_infeasible_budget():
    # PLAN.md P0.R3: the underlying allocator now spends
    # min(budget, sum u_p) and reports the residual via
    # underallocated_budget; the scheduler must surface that slack instead
    # of aborting the queue.
    nodes = [_node("a", 4, 1, 0.0), _node("b", 4, 2, 0.0)]
    summaries = FlexibleBudgetScheduler(queue_count=1, n_min=1).allocate(
        nodes, total_depth_budget=8
    )
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.allocated_budget == sum(summary.upper_bounds.values())
    assert summary.underallocated_budget == 8 - summary.allocated_budget
    assert summary.allocated_budget < 8
